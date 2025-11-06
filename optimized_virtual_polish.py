import argparse
import os
import time
import threading
import queue
import traceback
import multiprocessing

import cv2
import numpy as np
import tensorflow as tf


# ---------------- CONFIG ----------------
DEFAULT_MODEL = r"C:\\Users\\M.I TECH\\Desktop\\final\\nails_seg_s_yolov8_v1_float16.tflite"
TARGET_CAM = 0
DISPLAY_WINDOW = "Virtual Nail Polish. Press Q to quit."
DESIRED_FPS = 25             # UI/display FPS (what you see)
INFER_FPS = 15               # how often we actually run inference (lower = faster/less CPU)
TFLITE_THREADS = max(1, multiprocessing.cpu_count() - 1)
NAIL_COLOR = (199, 21, 133)  # fallback RGB
TEXTURE_DEFAULT = 0.35
MIN_CONTOUR = 60
DILATION_PIXELS = 4
FEATHER_IN = 3
FEATHER_OUT = 6
MAX_OUT_ALPHA = 0.08
MASK_DOWNSCALE = 0.5


cv2.setUseOptimized(True)


# ---------------- helper functions ----------------
_SMALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def adaptive_smooth_contour(pts, iterations=1):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new = []
        for i in range(len(pts)):
            p_prev = pts[(i - 1) % len(pts)]
            p_curr = pts[i]
            p_next = pts[(i + 1) % len(pts)]
            smoothed = 0.5 * p_curr + 0.25 * p_prev + 0.25 * p_next
            new.append(smoothed)
        pts = np.asarray(new, dtype=np.float32)
    return pts.astype(np.int32)


def refine_nail_mask(bin_mask, min_area=60, kernel=_SMALL_KERNEL):
    bin_mask = (bin_mask > 127).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return bin_mask
    m = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(m)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) >= 3:
            smooth = adaptive_smooth_contour(approx[:, 0, :], iterations=1)
            if len(smooth) >= 3:
                cv2.fillPoly(refined, [smooth], 255)
    refined = cv2.GaussianBlur(refined, (3, 3), 0)
    _, refined = cv2.threshold(refined, 100, 255, cv2.THRESH_BINARY)
    return refined


def create_feathered_alpha(mask, inner_feather=3, outer_feather=6, max_out_alpha=0.10):
    m = (mask > 127).astype(np.uint8) * 255
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)

    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(255 - m, cv2.DIST_L2, 5).astype(np.float32)

    alpha = np.zeros_like(dist_in, dtype=np.float32)
    alpha[m > 0] = 1.0

    if outer_feather > 0:
        falloff = np.clip(1.0 - (dist_out / float(outer_feather)), 0.0, 1.0)
        alpha = np.where(m > 0, alpha, falloff * float(max_out_alpha))

    if inner_feather > 0:
        inside = np.clip(dist_in / float(inner_feather), 0.0, 1.0)
        alpha_inside = 0.90 + 0.10 * inside
        alpha = np.where(m > 0, alpha_inside, alpha)

    k = max(3, int(outer_feather) * 2 + 1)
    if k % 2 == 0:
        k += 1
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def colorize_nail_texture(orig_bgr, alpha, color_rgb, texture_strength=0.35, brightness_adjust=1.0):
    img_f = orig_bgr.astype(np.float32) * brightness_adjust
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    highlights = np.sqrt(gray)
    color = np.asarray((color_rgb[2], color_rgb[1], color_rgb[0]), dtype=np.float32).reshape(1, 1, 3)
    colored_layer = img_f * texture_strength + color * (1.0 - texture_strength)
    highlight_boost = highlights[:, :, None] * 0.2
    colored_layer = colored_layer * (1 + highlight_boost)
    alpha_3 = alpha[:, :, None]
    out = img_f * (1.0 - alpha_3) + colored_layer * alpha_3
    return np.clip(out, 0, 255).astype(np.uint8)


def add_natural_sheen(img_rgb, hard_mask, intensity=0.06):
    gloss = np.zeros(hard_mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx = x + int(w * 0.4)
        cy = y + int(h * 0.25)
        rx = max(1, int(w * 0.35))
        ry = max(1, int(h * 0.15))
        cv2.ellipse(gloss, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    g = (gloss.astype(np.float32) / 255.0) * intensity
    g3 = np.stack([g, g, g], axis=2)
    out = img_rgb.astype(np.float32) * (1 - g3) + 255.0 * g3
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------- TFLite helpers ----------------
def load_tflite_interpreter(path, num_threads=TFLITE_THREADS):
    if not os.path.exists(path):
        raise FileNotFoundError(f"TFLite model not found at: {path}")

    interpreter = tf.lite.Interpreter(model_path=path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def _resolve_mask_outputs(outputs, cached_det_idx, cached_proto_idx):
    if cached_det_idx is not None and cached_proto_idx is not None:
        if cached_det_idx < len(outputs) and cached_proto_idx < len(outputs):
            return cached_det_idx, cached_proto_idx
    from_idx, proto_idx, _ = parse_tflite_outputs(outputs)
    return from_idx, proto_idx


# ---------------- Video reader thread ----------------
class CameraReader(threading.Thread):
    def __init__(self, src=0, width=640, height=480):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if not self.q.empty():
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass
        self.cap.release()

    def read(self, timeout=0.01):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False


# ---------------- Inference worker ----------------
class InferenceWorker(threading.Thread):
    shared_color_rgb = (199, 21, 133)
    shared_texture = TEXTURE_DEFAULT
    shared_sheen = False
    shared_sheen_intensity = 0.06

    def __init__(self, model_path, num_threads=TFLITE_THREADS,
                 infer_fps=INFER_FPS, mask_downscale=MASK_DOWNSCALE):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.infer_fps = float(infer_fps)
        self.mask_downscale = float(mask_downscale)
        self.in_q = queue.Queue(maxsize=1)
        self.out_lock = threading.Lock()
        self.latest_result = None
        self.running = True

        self.interpreter, input_details, output_details = load_tflite_interpreter(
            self.model_path, num_threads=num_threads
        )
        self.input_detail = input_details[0]
        self.output_details = output_details

        input_shape = self.input_detail['shape']
        if len(input_shape) != 4:
            raise ValueError("Expected NHWC input shape for TFLite model")
        self.in_h, self.in_w = int(input_shape[1]), int(input_shape[2])
        self.input_dtype = self.input_detail['dtype']

        if self.input_dtype == np.uint8:
            self.input_tensor = np.empty(input_shape, dtype=np.uint8)
            self._float_buffer = None
        else:
            self.input_tensor = np.empty(input_shape, dtype=np.float32)
            self._float_buffer = np.empty((self.in_h, self.in_w, input_shape[3]), dtype=np.float32)
        self._rgb_resize_buffer = np.empty((self.in_h, self.in_w, input_shape[3]), dtype=np.uint8)

        self.det_idx = None
        self.proto_idx = None

        # Cached helpers
        self._dilate_kernel = np.ones((max(1, int(DILATION_PIXELS * self.mask_downscale)),
                                       max(1, int(DILATION_PIXELS * self.mask_downscale))),
                                      dtype=np.uint8)
        self._inner_feather = max(1, int(FEATHER_IN * self.mask_downscale))
        self._outer_feather = max(1, int(FEATHER_OUT * self.mask_downscale))
        self._min_area_scaled = max(30, int(MIN_CONTOUR * (self.mask_downscale ** 2)))
        self._mask_logits = None
        self._mask_workspace = np.zeros((self.in_h, self.in_w), dtype=np.uint8)
        self._full_mask_buffer = None

    def submit(self, frame_bgr):
        if frame_bgr is None:
            return
        if not self.in_q.empty():
            try:
                _ = self.in_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.in_q.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def run(self):
        last_inf = 0.0
        while self.running:
            try:
                now = time.time()
                if self.infer_fps > 0:
                    wait = (1.0 / self.infer_fps) - (now - last_inf)
                    if wait > 0:
                        time.sleep(min(wait, 0.002))
                        continue
                try:
                    frame = self.in_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                ts0 = time.time()

                resized_bgr = cv2.resize(frame, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                if self.input_dtype == np.uint8:
                    cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB, dst=self.input_tensor[0])
                else:
                    cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb_resize_buffer)
                    self._float_buffer[...] = self._rgb_resize_buffer
                    self._float_buffer *= 1.0 / 255.0
                    np.copyto(self.input_tensor[0], self._float_buffer)

                inf_start = time.time()
                self.interpreter.set_tensor(self.input_detail['index'], self.input_tensor)
                self.interpreter.invoke()
                outputs = [self.interpreter.get_tensor(od['index']) for od in self.output_details]
                inf_dt = time.time() - inf_start

                self.det_idx, self.proto_idx = _resolve_mask_outputs(
                    outputs, self.det_idx, self.proto_idx
                )

                H, W = frame.shape[:2]
                if self._full_mask_buffer is None or self._full_mask_buffer.shape != (H, W):
                    self._full_mask_buffer = np.zeros((H, W), dtype=np.uint8)
                else:
                    self._full_mask_buffer.fill(0)
                combined_mask = self._full_mask_buffer

                if self.proto_idx is not None and self.det_idx is not None:
                    proto = outputs[self.proto_idx]
                    if proto.ndim == 4 and proto.shape[0] == 1:
                        proto = proto[0]
                    elif proto.ndim == 3 and proto.shape[0] == 1:
                        proto = proto[0]
                    if proto.ndim != 3:
                        proto = None

                    det = outputs[self.det_idx]
                    if det.ndim == 3 and det.shape[0] == 1:
                        det = det[0]
                    if det.ndim == 2 and det.shape[0] < det.shape[1] and det.shape[0] <= 50:
                        det = det.transpose(1, 0)

                    if proto is not None and det.ndim == 2:
                        proto = np.asarray(proto, dtype=np.float32)
                        det = np.asarray(det, dtype=np.float32)
                        P = proto.shape[-1]
                        cols = det.shape[1]
                        if cols >= 5 + P:
                            mask_coeffs = det[:, -P:]
                            scores = det[:, 4]
                            boxes = det[:, :4]
                        else:
                            mask_coeffs = det[:, -P:]
                            scores = det[:, 4] if det.shape[1] > 4 else np.zeros(det.shape[0])
                            boxes = det[:, :4] if det.shape[1] >= 4 else np.zeros((det.shape[0], 4))

                        valid_idx = np.where(scores > 0.25)[0]
                        if valid_idx.size > 0:
                            mask_coeffs = mask_coeffs[valid_idx]
                            boxes = boxes[valid_idx]

                            ph, pw, _ = proto.shape
                            proto_flat = proto.reshape(-1, P)
                            rows = proto_flat.shape[0]
                            cols_valid = mask_coeffs.shape[0]
                            mask_coeffs = np.ascontiguousarray(mask_coeffs, dtype=np.float32)

                            if self._mask_logits is None or self._mask_logits.shape != (rows, cols_valid):
                                self._mask_logits = np.empty((rows, cols_valid), dtype=np.float32)

                            np.dot(proto_flat, mask_coeffs.T, out=self._mask_logits)
                            np.negative(self._mask_logits, out=self._mask_logits)
                            np.exp(self._mask_logits, out=self._mask_logits)
                            self._mask_logits += 1.0
                            np.reciprocal(self._mask_logits, out=self._mask_logits)
                            mask_stack = self._mask_logits.reshape(ph, pw, cols_valid)

                            combined_mask_in = self._mask_workspace
                            combined_mask_in.fill(0)
                            for idx in range(mask_stack.shape[-1]):
                                mask = mask_stack[:, :, idx]

                                cx, cy, bw, bh = boxes[idx]
                                if cx > 1.5 or cy > 1.5 or bw > 1.5 or bh > 1.5:
                                    x1 = int(cx - bw / 2)
                                    y1 = int(cy - bh / 2)
                                    x2 = int(cx + bw / 2)
                                    y2 = int(cy + bh / 2)
                                else:
                                    x1 = int((cx - bw / 2) * self.in_w)
                                    y1 = int((cy - bh / 2) * self.in_h)
                                    x2 = int((cx + bw / 2) * self.in_w)
                                    y2 = int((cy + bh / 2) * self.in_h)

                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(self.in_w, x2), min(self.in_h, y2)

                                w = x2 - x1
                                h = y2 - y1

                                if w > 0 and h > 0:
                                    mask_roi = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                                    mask_roi_u8 = (mask_roi * 255.0).astype(np.uint8, copy=False)
                                    np.maximum(
                                        combined_mask_in[y1:y2, x1:x2],
                                        mask_roi_u8,
                                        out=combined_mask_in[y1:y2, x1:x2]
                                    )
                                else:
                                    mask_in = cv2.resize((mask * 255.0).astype(np.uint8),
                                                         (self.in_w, self.in_h),
                                                         interpolation=cv2.INTER_LINEAR)
                                    np.maximum(combined_mask_in, mask_in, out=combined_mask_in)

                            if combined_mask_in.any():
                                mask_full = cv2.resize(combined_mask_in, (W, H), interpolation=cv2.INTER_LINEAR)
                                _, mask_bin = cv2.threshold(mask_full, 127, 255, cv2.THRESH_BINARY)
                                np.copyto(combined_mask, mask_bin)

            if not combined_mask.any():
                self._build_fallback_mask(outputs, H, W, combined_mask)

            if not combined_mask.any():
                final_bgr = frame
                proc_dt = time.time() - ts0
            else:
                small_w = max(8, int(W * self.mask_downscale))
                small_h = max(8, int(H * self.mask_downscale))
                small_mask = cv2.resize(combined_mask, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                _, small_mask = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)

                refined_small = refine_nail_mask(
                    small_mask,
                    min_area=self._min_area_scaled,
                    kernel=_SMALL_KERNEL
                )
                refined_small = cv2.dilate(refined_small, self._dilate_kernel, iterations=1)

                alpha_small = create_feathered_alpha(
                    refined_small,
                    inner_feather=self._inner_feather,
                    outer_feather=self._outer_feather,
                    max_out_alpha=MAX_OUT_ALPHA
                )

                alpha_map = cv2.resize(alpha_small, (W, H), interpolation=cv2.INTER_LINEAR)

                colored = colorize_nail_texture(
                    frame,
                    alpha_map,
                    InferenceWorker.shared_color_rgb,
                    texture_strength=InferenceWorker.shared_texture,
                    brightness_adjust=1.05
                )

                if InferenceWorker.shared_sheen:
                    refined_up = cv2.resize(refined_small, (W, H), interpolation=cv2.INTER_NEAREST)
                    final_bgr = add_natural_sheen(
                        colored,
                        refined_up,
                        intensity=InferenceWorker.shared_sheen_intensity
                    )
                else:
                    final_bgr = colored

                proc_dt = time.time() - ts0

            with self.out_lock:
                self.latest_result = {
                    'bgr': final_bgr,
                    'ts': time.time(),
                    'inf_dt': inf_dt,
                    'proc_dt': proc_dt
                }
            last_inf = time.time()
            except Exception as exc:
                print("Worker error:", exc)
                traceback.print_exc()
                time.sleep(0.05)

    def get_latest(self):
        with self.out_lock:
            return None if self.latest_result is None else dict(self.latest_result)

    def stop(self):
        self.running = False

    def _build_fallback_mask(self, outputs, H, W, out_mask):
        out_mask.fill(0)
        for idx, o in enumerate(outputs):
            if idx in (self.proto_idx, self.det_idx):
                continue
            candidate = None
            if o.ndim == 4 and o.shape[0] == 1:
                candidate = o[0]
            elif o.ndim == 3:
                candidate = o
            if candidate is None or candidate.ndim != 3:
                continue
            if candidate.shape[0] > 200:
                continue
            for m in candidate:
                mask = m
                if mask.ndim != 2:
                    continue
                mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_LINEAR)
                _, m_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
                np.maximum(out_mask, m_bin, out=out_mask)
                if out_mask.any():
                    return out_mask
        return out_mask


def parse_tflite_outputs(outputs):
    det_idx = None
    proto_idx = None
    P = None
    for i, o in enumerate(outputs):
        if o.ndim == 4 and o.shape[-1] >= 1 and o.shape[-1] <= 1024:
            proto_idx = i
            P = o.shape[-1]
            break
        if o.ndim == 3 and o.shape[-1] >= 1 and o.shape[-1] <= 1024 and o.shape[0] <= 1024:
            proto_idx = i
            P = o.shape[-1]
            break
    if P is not None:
        for i, o in enumerate(outputs):
            if i == proto_idx:
                continue
            if o.ndim >= 2 and o.shape[-1] == 5 + P:
                det_idx = i
                break
    if proto_idx is None:
        for i, o in enumerate(outputs):
            if o.ndim in (3, 4) and o.size > 1000:
                proto_idx = i
                P = o.shape[-1] if o.ndim in (3, 4) else None
                break
    if det_idx is None:
        for i, o in enumerate(outputs):
            if i == proto_idx:
                continue
            if o.ndim >= 2 and o.shape[-1] >= 6:
                det_idx = i
                break
    return det_idx, proto_idx, outputs


# ---------------- Main processing ----------------
def run_live(model_path, cam_index=0, desired_fps=DESIRED_FPS, color_hex="#c71585",
             add_sheen=False, sheen_intensity=0.06, texture_amt=TEXTURE_DEFAULT,
             infer_fps=INFER_FPS, mask_downscale=MASK_DOWNSCALE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at: {model_path}")

    hexv = color_hex.lstrip('#')
    if len(hexv) == 6:
        try:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            color_rgb = (r, g, b)
        except ValueError:
            color_rgb = NAIL_COLOR
    else:
        color_rgb = NAIL_COLOR

    InferenceWorker.shared_color_rgb = color_rgb
    InferenceWorker.shared_texture = texture_amt
    InferenceWorker.shared_sheen = bool(add_sheen)
    InferenceWorker.shared_sheen_intensity = float(sheen_intensity)

    reader = CameraReader(src=cam_index, width=640, height=480)
    reader.start()

    worker = InferenceWorker(
        model_path,
        num_threads=TFLITE_THREADS,
        infer_fps=infer_fps,
        mask_downscale=mask_downscale
    )
    worker.start()

    print("Model loaded. Starting camera. Press Q to quit.")
    last_time = time.time()
    fps_avg = 0.0
    display_surface = None
    try:
        while True:
            start = time.time()
            ok, frame = reader.read(timeout=0.02)
            latest = worker.get_latest()

            if ok and frame is not None:
                worker.submit(frame)
                base_bgr = latest['bgr'] if latest is not None else frame
            else:
                if latest is None:
                    time.sleep(0.005)
                    continue
                base_bgr = latest['bgr']

            now = time.time()
            dt = now - last_time
            last_time = now
            fps_curr = 1.0 / dt if dt > 0 else 0.0
            fps_avg = fps_avg * 0.85 + fps_curr * 0.15
            text = f"FPS {fps_avg:.1f}"

            status = "INF - PRC -"
            if latest is not None:
                proc_ms = latest.get('proc_dt', 0.0) * 1000.0
                inf_ms = latest.get('inf_dt', 0.0) * 1000.0
                status = f"INF {inf_ms:.0f}ms PRC {proc_ms:.0f}ms"

            if display_surface is None or display_surface.shape != base_bgr.shape:
                display_surface = np.empty_like(base_bgr)
            np.copyto(display_surface, base_bgr)
            disp_bgr = display_surface
            cv2.putText(disp_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp_bgr, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow(DISPLAY_WINDOW, disp_bgr)

            target_frame_time = 1.0 / float(desired_fps)
            loop_time = time.time() - start
            if loop_time < target_frame_time:
                time.sleep(max(0, target_frame_time - loop_time))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        worker.stop()
        worker.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("Stopped camera.")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to .tflite model.")
    ap.add_argument("--cam", type=int, default=TARGET_CAM, help="Camera index.")
    ap.add_argument("--fps", type=int, default=DESIRED_FPS, help="Target display FPS.")
    ap.add_argument("--color", type=str, default="#c71585", help="Polish hex color.")
    ap.add_argument("--sheen", action="store_true", help="Enable sheen.")
    ap.add_argument("--sheen_int", type=float, default=0.06, help="Sheen intensity.")
    ap.add_argument("--texture", type=float, default=TEXTURE_DEFAULT, help="Texture preservation.")
    ap.add_argument("--infer_fps", type=float, default=INFER_FPS, help="Max inference FPS (lower => faster UI).")
    ap.add_argument("--mask_downscale", type=float, default=MASK_DOWNSCALE,
                    help="Run heavy mask ops on downscaled mask (0.3-1.0).")
    args = ap.parse_args()

    run_live(
        args.model,
        cam_index=args.cam,
        desired_fps=args.fps,
        color_hex=args.color,
        add_sheen=args.sheen,
        sheen_intensity=args.sheen_int,
        texture_amt=args.texture,
        infer_fps=args.infer_fps,
        mask_downscale=args.mask_downscale
    )
