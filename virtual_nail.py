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
DESIRED_FPS = 30
INFER_FPS = 15
TFLITE_THREADS = max(1, multiprocessing.cpu_count() - 1)
NAIL_COLOR = (199, 21, 133)
TEXTURE_DEFAULT = 0.35
MIN_CONTOUR = 60
DILATION_PIXELS = 2
FEATHER_IN = 1
FEATHER_OUT = 3
MAX_OUT_ALPHA = 0.08
MASK_DOWNSCALE = 0.3
MAX_ACTIVE_MASKS = 6
SIGMOID_CLIP_VALUE = 8.0
SCORE_THRESHOLD = 0.25

cv2.setUseOptimized(True)
cv2.setNumThreads(2)


# ---------------- helper functions ----------------
_SMALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def adaptive_smooth_contour(pts, iterations=1):
    """Minimal contour smoothing"""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)

    p_prev = np.roll(pts, 1, axis=0)
    p_next = np.roll(pts, -1, axis=0)
    pts = 0.5 * pts + 0.25 * p_prev + 0.25 * p_next

    return pts.astype(np.int32)


def refine_nail_mask(bin_mask, min_area=60, kernel=_SMALL_KERNEL):
    """Fast mask refinement - skip heavy processing"""
    bin_mask = (bin_mask > 127).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return bin_mask

    m = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(m)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        if len(c) >= 3:
            smooth = adaptive_smooth_contour(c[:, 0, :], iterations=1)
            if len(smooth) >= 3:
                cv2.fillPoly(refined, [smooth], 255)

    return refined


def create_feathered_alpha(mask, inner_feather=1, outer_feather=3, max_out_alpha=0.10):
    """Minimal alpha channel creation"""
    m = (mask > 127).astype(np.uint8) * 255
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)

    alpha = (m.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def colorize_nail_texture(orig_rgb, alpha, color_rgb, texture_strength=0.35, brightness_adjust=1.0):
    """Simplified colorization"""
    img_f = orig_rgb.astype(np.float32) * brightness_adjust

    color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    colored_layer = img_f * texture_strength + color * (1.0 - texture_strength)

    alpha_3 = alpha[:, :, None]
    out = img_f * (1.0 - alpha_3) + colored_layer * alpha_3
    return np.clip(out, 0, 255).astype(np.uint8)


def add_natural_sheen(img_rgb, hard_mask, intensity=0.06):
    """Fast sheen effect"""
    gloss = np.zeros(hard_mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 5:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

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

    g = (gloss.astype(np.float32) * (intensity / 255.0))
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.q = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
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

    def read(self, timeout=0.001):
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

        self.det_idx = None
        self.proto_idx = None

        self._dilate_kernel = np.ones((max(1, int(DILATION_PIXELS * self.mask_downscale)),
                                       max(1, int(DILATION_PIXELS * self.mask_downscale))),
                                      dtype=np.uint8)
        self._inner_feather = max(1, int(FEATHER_IN * self.mask_downscale))
        self._outer_feather = max(1, int(FEATHER_OUT * self.mask_downscale))
        self._min_area_scaled = max(20, int(MIN_CONTOUR * (self.mask_downscale ** 2)))

        self._rgb_buffer = None
        self._resized_buffer = np.empty((self.in_h, self.in_w, input_shape[3]), dtype=np.uint8)
        self._combined_mask_in = np.zeros((self.in_h, self.in_w), dtype=np.uint8)
        self._frame_shape = None
        self._mask_full_buffer = None
        self._alpha_full_buffer = None
        self._small_dims = None
        self._small_mask_buffer = None

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

    @staticmethod
    def _fast_sigmoid(x):
        np.clip(x, -SIGMOID_CLIP_VALUE, SIGMOID_CLIP_VALUE, out=x)
        return 0.5 * (np.tanh(0.5 * x) + 1.0)

    def _ensure_frame_buffers(self, height, width):
        if self._frame_shape == (height, width):
            return
        self._frame_shape = (height, width)
        self._mask_full_buffer = np.zeros((height, width), dtype=np.uint8)
        self._alpha_full_buffer = np.zeros((height, width), dtype=np.float32)
        small_w = max(8, int(width * self.mask_downscale))
        small_h = max(8, int(height * self.mask_downscale))
        self._small_dims = (small_h, small_w)
        self._small_mask_buffer = np.zeros((small_h, small_w), dtype=np.uint8)

    def run(self):
        last_inf = 0.0
        while self.running:
            try:
                now = time.time()
                if self.infer_fps > 0 and (now - last_inf) < (1.0 / self.infer_fps):
                    time.sleep(0.001)
                    continue
                try:
                    frame = self.in_q.get(timeout=0.01)
                except queue.Empty:
                    continue

                ts0 = time.time()

                if self._rgb_buffer is None or self._rgb_buffer.shape != frame.shape:
                    self._rgb_buffer = np.empty_like(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
                resized = cv2.resize(frame_rgb, (self.in_w, self.in_h),
                                     interpolation=cv2.INTER_NEAREST,
                                     dst=self._resized_buffer)

                if self.input_dtype == np.uint8:
                    np.copyto(self.input_tensor[0], resized, casting='unsafe')
                else:
                    np.multiply(resized, 1.0 / 255.0, out=self._float_buffer, casting='unsafe')
                    np.copyto(self.input_tensor[0], self._float_buffer, casting='unsafe')

                inf_start = time.time()
                self.interpreter.set_tensor(self.input_detail['index'], self.input_tensor)
                self.interpreter.invoke()
                inf_dt = time.time() - inf_start

                outputs_snapshot = None
                proto = None
                det = None

                if self.det_idx is None or self.proto_idx is None:
                    outputs_snapshot = [
                        self.interpreter.get_tensor(od['index'])
                        for od in self.output_details
                    ]
                    self.det_idx, self.proto_idx = _resolve_mask_outputs(
                        outputs_snapshot, self.det_idx, self.proto_idx
                    )
                    if self.proto_idx is not None and self.proto_idx < len(outputs_snapshot):
                        proto = outputs_snapshot[self.proto_idx]
                    if self.det_idx is not None and self.det_idx < len(outputs_snapshot):
                        det = outputs_snapshot[self.det_idx]
                else:
                    if self.proto_idx is not None:
                        proto = self.interpreter.get_tensor(
                            self.output_details[self.proto_idx]['index']
                        )
                    if self.det_idx is not None:
                        det = self.interpreter.get_tensor(
                            self.output_details[self.det_idx]['index']
                        )

                H, W = frame.shape[:2]
                self._ensure_frame_buffers(H, W)
                combined_mask_in = self._combined_mask_in
                combined_mask_in.fill(0)
                combined_mask = None

                if proto is not None and det is not None:
                    if proto.ndim == 4 and proto.shape[0] == 1:
                        proto = proto[0]
                    elif proto.ndim == 3 and proto.shape[0] == 1:
                        proto = proto[0]
                    if proto.ndim != 3:
                        proto = None

                    if det.ndim == 3 and det.shape[0] == 1:
                        det = det[0]
                    if det.ndim == 2 and det.shape[0] < det.shape[1] and det.shape[0] <= 50:
                        det = det.transpose(1, 0)

                    if proto is not None and det.ndim == 2:
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

                        valid_idx = np.where(scores > SCORE_THRESHOLD)[0]
                        if valid_idx.size > 0:
                            if valid_idx.size > MAX_ACTIVE_MASKS:
                                top_idx = np.argpartition(scores[valid_idx], -MAX_ACTIVE_MASKS)[-MAX_ACTIVE_MASKS:]
                                valid_idx = valid_idx[top_idx]
                                order = np.argsort(scores[valid_idx])[::-1]
                                valid_idx = valid_idx[order]

                            mask_coeffs = mask_coeffs[valid_idx]
                            boxes = boxes[valid_idx]

                            proto_flat = proto.reshape(-1, P)
                            mask_logits = proto_flat @ mask_coeffs.T
                            mask_logits = self._fast_sigmoid(mask_logits)
                            mask_stack = mask_logits.reshape(proto.shape[0], proto.shape[1], -1).astype(np.float32, copy=False)
                            mask_stack = cv2.resize(mask_stack, (self.in_w, self.in_h),
                                                    interpolation=cv2.INTER_NEAREST)
                            mask_stack = np.clip(mask_stack * 255.0, 0, 255).astype(np.uint8, copy=False)

                            num_masks = mask_stack.shape[2]
                            for idx_mask in range(num_masks):
                                mask_in = mask_stack[:, :, idx_mask]
                                if not mask_in.any():
                                    continue
                                cx, cy, bw, bh = boxes[idx_mask]
                                if cx > 1.5 or cy > 1.5 or bw > 1.5 or bh > 1.5:
                                    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                                    x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                                else:
                                    x1 = int((cx - bw / 2) * self.in_w)
                                    y1 = int((cy - bh / 2) * self.in_h)
                                    x2 = int((cx + bw / 2) * self.in_w)
                                    y2 = int((cy + bh / 2) * self.in_h)

                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(self.in_w, x2), min(self.in_h, y2)

                                if x2 > x1 and y2 > y1:
                                    roi_src = mask_in[y1:y2, x1:x2]
                                    if roi_src.size:
                                        roi_dst = combined_mask_in[y1:y2, x1:x2]
                                        np.maximum(roi_dst, roi_src, out=roi_dst)
                                else:
                                    np.maximum(combined_mask_in, mask_in, out=combined_mask_in)

                            if combined_mask_in.any():
                                mask_full = cv2.resize(combined_mask_in, (W, H),
                                                       interpolation=cv2.INTER_NEAREST,
                                                       dst=self._mask_full_buffer)
                                cv2.threshold(mask_full, 127, 255, cv2.THRESH_BINARY, dst=mask_full)
                                combined_mask = mask_full

                if combined_mask is None or not combined_mask.any():
                    combined_mask = self._build_fallback_mask(H, W, outputs_snapshot)
                outputs_snapshot = None

                if combined_mask is None or not combined_mask.any():
                    final_rgb = frame_rgb
                    proc_dt = time.time() - ts0
                else:
                    small_h, small_w = self._small_dims
                    small_mask = cv2.resize(combined_mask, (small_w, small_h),
                                            interpolation=cv2.INTER_NEAREST,
                                            dst=self._small_mask_buffer)
                    cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY, dst=small_mask)

                    refined_small = refine_nail_mask(
                        small_mask,
                        min_area=self._min_area_scaled,
                        kernel=_SMALL_KERNEL
                    )
                    refined_small = cv2.dilate(refined_small, self._dilate_kernel,
                                               iterations=1, dst=refined_small)

                    alpha_small = create_feathered_alpha(
                        refined_small,
                        inner_feather=self._inner_feather,
                        outer_feather=self._outer_feather,
                        max_out_alpha=MAX_OUT_ALPHA
                    )

                    alpha_map = cv2.resize(alpha_small, (W, H),
                                           interpolation=cv2.INTER_LINEAR,
                                           dst=self._alpha_full_buffer)

                    colored = colorize_nail_texture(
                        frame_rgb,
                        alpha_map,
                        InferenceWorker.shared_color_rgb,
                        texture_strength=InferenceWorker.shared_texture,
                        brightness_adjust=1.05
                    )

                    if InferenceWorker.shared_sheen:
                        refined_up = cv2.resize(refined_small, (W, H),
                                                interpolation=cv2.INTER_NEAREST,
                                                dst=self._mask_full_buffer)
                        final_rgb = add_natural_sheen(
                            colored,
                            refined_up,
                            intensity=InferenceWorker.shared_sheen_intensity
                        )
                    else:
                        final_rgb = colored

                    proc_dt = time.time() - ts0

                with self.out_lock:
                    self.latest_result = {
                        'rgb': final_rgb,
                        'ts': time.time(),
                        'inf_dt': inf_dt,
                        'proc_dt': proc_dt
                    }
                last_inf = time.time()
            except Exception as exc:
                print("Worker error:", exc)
                traceback.print_exc()
                time.sleep(0.01)

    def get_latest(self):
        with self.out_lock:
            return None if self.latest_result is None else dict(self.latest_result)

    def stop(self):
        self.running = False

    def _build_fallback_mask(self, H, W, outputs_snapshot=None):
        combined = self._mask_full_buffer
        combined.fill(0)
        if outputs_snapshot is None:
            source_outputs = [
                self.interpreter.get_tensor(od['index'])
                for od in self.output_details
            ]
        else:
            source_outputs = outputs_snapshot

        for idx, o in enumerate(source_outputs):
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
            processed = 0
            for m in candidate:
                if processed >= MAX_ACTIVE_MASKS:
                    break
                mask = m
                if mask.ndim != 2:
                    continue
                mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY, dst=mask_resized)
                np.maximum(combined, mask_resized, out=combined)
                processed += 1
        return combined


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

    try:
        while True:
            start = time.time()
            ok, frame = reader.read(timeout=0.001)
            latest = worker.get_latest()

            if ok and frame is not None:
                worker.submit(frame)
                if latest is not None:
                    display_rgb = latest['rgb']
                else:
                    display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                if latest is None:
                    time.sleep(0.001)
                    continue
                display_rgb = latest['rgb']

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
                total_latency = proc_ms
                status = f"INF {inf_ms:.0f}ms TOTAL {total_latency:.0f}ms"

            disp_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
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
    ap.add_argument("--infer_fps", type=float, default=INFER_FPS, help="Max inference FPS.")
    ap.add_argument("--mask_downscale", type=float, default=MASK_DOWNSCALE,
                    help="Mask processing downscale (0.3-1.0).")
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
