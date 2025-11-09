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
INFER_FPS = 8                # how often we actually run inference (lower = faster/less CPU)
TFLITE_THREADS = max(1, multiprocessing.cpu_count() - 1)
NAIL_COLOR = (199, 21, 133)  # fallback RGB
TEXTURE_DEFAULT = 0.35
MIN_CONTOUR = 60
DILATION_PIXELS = 4
FEATHER_IN = 3
FEATHER_OUT = 6
MAX_OUT_ALPHA = 0.08
MASK_DOWNSCALE = 0.35
CAPTURE_WIDTH = 480
CAPTURE_HEIGHT = 360
ROI_PADDING = 8
MASK_REUSE_THRESHOLD = 0.10


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


def create_feathered_alpha(mask, inner_feather=3, outer_feather=6, max_out_alpha=0.10, out_buffer=None):
    mask_bin = (mask > 127).astype(np.uint8)
    if mask_bin.sum() == 0:
        if out_buffer is not None and out_buffer.shape == mask_bin.shape:
            out_buffer.fill(0.0)
            return out_buffer
        return np.zeros_like(mask_bin, dtype=np.float32)

    if out_buffer is None or out_buffer.shape != mask_bin.shape:
        alpha = np.zeros_like(mask_bin, dtype=np.float32)
    else:
        alpha = out_buffer
        alpha.fill(0.0)

    ys, xs = np.where(mask_bin > 0)
    y1 = max(0, ys.min() - 2)
    y2 = min(mask_bin.shape[0], ys.max() + 3)
    x1 = max(0, xs.min() - 2)
    x2 = min(mask_bin.shape[1], xs.max() + 3)

    submask = mask_bin[y1:y2, x1:x2]
    submask = (submask > 0).astype(np.uint8)

    dist_in = cv2.distanceTransform(submask, cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - submask, cv2.DIST_L2, 5).astype(np.float32)

    sub_alpha = np.zeros_like(dist_in, dtype=np.float32)
    sub_alpha[submask > 0] = 1.0

    if outer_feather > 0:
        falloff = np.clip(1.0 - (dist_out / float(outer_feather)), 0.0, 1.0)
        sub_alpha = np.where(submask > 0, sub_alpha, falloff * float(max_out_alpha))

    if inner_feather > 0:
        inside = np.clip(dist_in / float(inner_feather), 0.0, 1.0)
        alpha_inside = 0.90 + 0.10 * inside
        sub_alpha = np.where(submask > 0, alpha_inside, sub_alpha)

    k = max(3, int(outer_feather) * 2 + 1)
    if k % 2 == 0:
        k += 1
    sub_alpha = cv2.GaussianBlur(sub_alpha, (k, k), 0)
    sub_alpha = np.clip(sub_alpha, 0.0, 1.0)

    alpha[y1:y2, x1:x2] = sub_alpha
    return alpha.astype(np.float32)


def colorize_nail_texture(orig_rgb, alpha, color_rgb, texture_strength=0.35, brightness_adjust=1.0,
                          out=None, work_f32=None, blend_f32=None, gray_u8=None, bbox=None):
    h, w = orig_rgb.shape[:2]
    if bbox is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w, x1))
        x2 = max(x1, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(y1, min(h, y2))

    orig_roi = orig_rgb[y1:y2, x1:x2]
    alpha_roi = alpha[y1:y2, x1:x2]

    if out is None or out.shape != orig_rgb.shape:
        out = np.empty_like(orig_rgb)
    out_roi = out[y1:y2, x1:x2]

    if alpha_roi.size == 0 or not np.any(alpha_roi > 1e-3):
        np.copyto(out_roi, orig_roi, casting='unsafe')
        return out

    if work_f32 is None or work_f32.shape != orig_rgb.shape:
        work_roi = orig_roi.astype(np.float32)
        if brightness_adjust != 1.0:
            work_roi *= brightness_adjust
    else:
        work_roi = work_f32[y1:y2, x1:x2]
        np.multiply(orig_roi, brightness_adjust, out=work_roi, casting='unsafe')

    if blend_f32 is None or blend_f32.shape != orig_rgb.shape:
        blend_roi = np.empty_like(orig_roi, dtype=np.float32)
    else:
        blend_roi = blend_f32[y1:y2, x1:x2]

    if gray_u8 is None or gray_u8.shape != orig_rgb.shape[:2]:
        gray = cv2.cvtColor(orig_roi, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray_roi = gray_u8[y1:y2, x1:x2]
        cv2.cvtColor(orig_roi, cv2.COLOR_RGB2GRAY, dst=gray_roi)
        gray = gray_roi.astype(np.float32) / 255.0

    highlights = np.sqrt(gray, dtype=np.float32)
    color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    np.multiply(work_roi, texture_strength, out=blend_roi)
    blend_roi += color * (1.0 - texture_strength)
    highlight_boost = highlights[:, :, None] * 0.2
    blend_roi *= (1 + highlight_boost)

    alpha_3 = alpha_roi[:, :, None]
    np.multiply(work_roi, (1.0 - alpha_3), out=work_roi)
    np.multiply(blend_roi, alpha_3, out=blend_roi)
    np.add(work_roi, blend_roi, out=blend_roi)
    np.clip(blend_roi, 0, 255, out=blend_roi)
    np.rint(blend_roi, out=blend_roi)
    np.copyto(out_roi, blend_roi.astype(np.uint8), casting='unsafe')
    return out


def add_natural_sheen(img_rgb, hard_mask, intensity=0.06,
                      gloss_buffer=None, sheen_map=None, work_f32=None, out=None, bbox=None):
    h, w = hard_mask.shape[:2]
    if bbox is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w, x1))
        x2 = max(x1, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(y1, min(h, y2))

    mask_roi = hard_mask[y1:y2, x1:x2]
    img_roi = img_rgb[y1:y2, x1:x2]

    if mask_roi.size == 0 or cv2.countNonZero(mask_roi) == 0:
        if out is None or out.shape != img_rgb.shape:
            out_frame = np.empty_like(img_rgb)
        else:
            out_frame = out
        out_roi = out_frame[y1:y2, x1:x2]
        np.copyto(out_roi, img_roi, casting='unsafe')
        return out_frame

    if gloss_buffer is None or gloss_buffer.shape != hard_mask.shape:
        gloss = np.zeros(mask_roi.shape, dtype=np.uint8)
    else:
        gloss = gloss_buffer[y1:y2, x1:x2]
        gloss.fill(0)
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, w_box, h_box = cv2.boundingRect(c)
        cx = x + int(w_box * 0.4)
        cy = y + int(h_box * 0.25)
        rx = max(1, int(w_box * 0.35))
        ry = max(1, int(h_box * 0.15))
        cv2.ellipse(gloss, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    if sheen_map is None or sheen_map.shape != hard_mask.shape:
        g = gloss.astype(np.float32) * (intensity / 255.0)
    else:
        g_roi = sheen_map[y1:y2, x1:x2]
        np.multiply(gloss, intensity / 255.0, out=g_roi, casting='unsafe')
        g = g_roi

    if work_f32 is None or work_f32.shape != img_rgb.shape:
        work = img_roi.astype(np.float32)
    else:
        work = work_f32[y1:y2, x1:x2]
        np.copyto(work, img_roi, casting='unsafe')

    g_expanded = g[:, :, None]
    work *= (1.0 - g_expanded)
    work += 255.0 * g_expanded

    if out is None or out.shape != img_rgb.shape:
        out_frame = np.empty_like(img_rgb)
    else:
        out_frame = out
    out_roi = out_frame[y1:y2, x1:x2]
    np.clip(work, 0, 255, out=work)
    np.rint(work, out=work)
    np.copyto(out_roi, work.astype(np.uint8), casting='unsafe')
    return out_frame


# ---------------- TFLite helpers ----------------
def _try_load_delegate(lib_name):
    try:
        return tf.lite.experimental.load_delegate(lib_name)
    except Exception:
        return None


def load_tflite_interpreter(path, num_threads=TFLITE_THREADS):
    if not os.path.exists(path):
        raise FileNotFoundError(f"TFLite model not found at: {path}")

    delegates = []
    for lib in ("libtensorflowlite_gpu_delegate.so", "libtensorflowlite_delegate_xnnpack.so",
                "libtensorflowlite_xnnpack_delegate.so"):
        delegate = _try_load_delegate(lib)
        if delegate is not None:
            delegates.append(delegate)
            print(f"Loaded delegate: {lib}")
            break

    try:
        interpreter = tf.lite.Interpreter(
            model_path=path,
            num_threads=num_threads,
            experimental_delegates=delegates if delegates else None
        )
    except ValueError:
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
    def __init__(self, src=0, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT):
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

        self.det_idx = None
        self.proto_idx = None

        self._dilate_kernel = np.ones((max(1, int(DILATION_PIXELS * self.mask_downscale)),
                                       max(1, int(DILATION_PIXELS * self.mask_downscale))),
                                      dtype=np.uint8)
        self._inner_feather = max(1, int(FEATHER_IN * self.mask_downscale))
        self._outer_feather = max(1, int(FEATHER_OUT * self.mask_downscale))
        self._min_area_scaled = max(30, int(MIN_CONTOUR * (self.mask_downscale ** 2)))

        self._combined_mask_in = np.zeros((self.in_h, self.in_w), dtype=np.uint8)
        self._frame_shape = None
        self._combined_mask = None
        self._small_mask = None
        self._refined_small = None
        self._alpha_small = None
        self._alpha_map = None
        self._refined_up = None
        self._work_f32 = None
        self._blend_f32 = None
        self._gray_u8 = None
        self._mask_resize_buf = np.zeros((self.in_h, self.in_w), dtype=np.uint8)
        self._mask_resize_buf_f32 = np.zeros((self.in_h, self.in_w), dtype=np.float32)
        self._output_buffers = [None, None]
        self._output_toggle = 0
        self._fallback_mask = None
        self._fallback_tmp = None
        self._sheen_map = None
        self._gloss_buffer = None
        self._sheen_work = None
        self._prev_small_mask = None
        self._prev_alpha_map = None
        self._prev_refined_up = None
        self._prev_output_rgb = None
        self._prev_has_valid = False
        self._last_render_params = None
        self._prev_bbox = None

    def _ensure_frame_buffers(self, H, W):
        if self._frame_shape == (H, W):
            return
        self._frame_shape = (H, W)
        small_w = max(8, int(W * self.mask_downscale))
        small_h = max(8, int(H * self.mask_downscale))
        self._combined_mask = np.zeros((H, W), dtype=np.uint8)
        self._small_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        self._refined_small = np.zeros((small_h, small_w), dtype=np.uint8)
        self._alpha_small = np.zeros((small_h, small_w), dtype=np.float32)
        self._alpha_map = np.zeros((H, W), dtype=np.float32)
        self._refined_up = np.zeros((H, W), dtype=np.uint8)
        self._work_f32 = np.zeros((H, W, 3), dtype=np.float32)
        self._blend_f32 = np.zeros((H, W, 3), dtype=np.float32)
        self._gray_u8 = np.zeros((H, W), dtype=np.uint8)
        self._output_buffers = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(2)]
        self._fallback_mask = np.zeros((H, W), dtype=np.uint8)
        self._fallback_tmp = np.zeros((H, W), dtype=np.uint8)
        self._sheen_map = np.zeros((H, W), dtype=np.float32)
        self._gloss_buffer = np.zeros((H, W), dtype=np.uint8)
        self._sheen_work = np.zeros((H, W, 3), dtype=np.float32)
        self._prev_small_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        self._prev_alpha_map = np.zeros((H, W), dtype=np.float32)
        self._prev_refined_up = np.zeros((H, W), dtype=np.uint8)
        self._prev_output_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        self._prev_has_valid = False
        self._last_render_params = None
        self._prev_bbox = None

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
                if self.infer_fps > 0 and (now - last_inf) < (1.0 / self.infer_fps):
                    time.sleep(0.005)
                    continue
                try:
                    frame = self.in_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                ts0 = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                H, W = frame.shape[:2]
                self._ensure_frame_buffers(H, W)
                output_buf = self._output_buffers[self._output_toggle]
                self._output_toggle = 1 - self._output_toggle

                resized = cv2.resize(frame_rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                if self.input_dtype == np.uint8:
                    np.copyto(self.input_tensor[0], resized, casting='unsafe')
                else:
                    self._float_buffer[...] = resized
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

                combined_mask_in = self._combined_mask_in
                combined_mask_in.fill(0)

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
                            mask_logits = proto_flat @ mask_coeffs.T
                            mask_stack = 1.0 / (1.0 + np.exp(-mask_logits))
                            mask_stack = mask_stack.reshape(ph, pw, -1)

                            for idx in range(mask_stack.shape[-1]):
                                mask_plane = mask_stack[:, :, idx]
                                cv2.resize(mask_plane, (self.in_w, self.in_h),
                                           dst=self._mask_resize_buf_f32,
                                           interpolation=cv2.INTER_LINEAR)
                                cv2.convertScaleAbs(self._mask_resize_buf_f32,
                                                    dst=self._mask_resize_buf,
                                                    alpha=255.0)
                                mask_in = self._mask_resize_buf
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

                                if x2 > x1 and y2 > y1:
                                    roi = mask_in[y1:y2, x1:x2]
                                    if roi.size:
                                        np.maximum(combined_mask_in[y1:y2, x1:x2], roi, out=combined_mask_in[y1:y2, x1:x2])
                                else:
                                    np.maximum(combined_mask_in, mask_in, out=combined_mask_in)

                combined_mask = self._combined_mask
                combined_mask.fill(0)

                if combined_mask_in.any():
                    cv2.resize(combined_mask_in, (W, H), dst=combined_mask, interpolation=cv2.INTER_LINEAR)
                    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
                else:
                    fallback = self._build_fallback_mask(outputs, H, W)
                    if fallback is not None:
                        combined_mask[...] = fallback

                if not combined_mask.any():
                    np.copyto(output_buf, frame_rgb, casting='unsafe')
                    final_rgb = output_buf
                    proc_dt = time.time() - ts0
                    self._prev_has_valid = False
                    self._last_render_params = None
                    self._prev_bbox = None
                else:
                    small_mask = self._small_mask
                    cv2.resize(combined_mask, (small_mask.shape[1], small_mask.shape[0]),
                               dst=small_mask, interpolation=cv2.INTER_LINEAR)
                    cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY, dst=small_mask)

                    render_params = (
                        InferenceWorker.shared_color_rgb,
                        round(float(InferenceWorker.shared_texture), 4),
                        bool(InferenceWorker.shared_sheen),
                        round(float(InferenceWorker.shared_sheen_intensity), 4)
                    )
                    reuse_output_only = False
                    reuse_mask_only = False
                    if self._prev_has_valid:
                        diff = cv2.absdiff(small_mask, self._prev_small_mask)
                        changed_pixels = cv2.countNonZero(diff)
                        prev_pixels = cv2.countNonZero(self._prev_small_mask)
                        current_pixels = cv2.countNonZero(small_mask)
                        denom = float(max(prev_pixels, current_pixels, 1))
                        change_ratio = changed_pixels / denom
                        if change_ratio <= MASK_REUSE_THRESHOLD:
                            if render_params == self._last_render_params:
                                reuse_output_only = True
                            else:
                                reuse_mask_only = True

                    if reuse_output_only:
                        np.copyto(output_buf, self._prev_output_rgb, casting='unsafe')
                        final_rgb = output_buf
                        proc_dt = time.time() - ts0
                    else:
                        if not reuse_mask_only:
                            refined_small_arr = refine_nail_mask(
                                small_mask,
                                min_area=self._min_area_scaled,
                                kernel=_SMALL_KERNEL
                            )
                            np.copyto(self._refined_small, refined_small_arr, casting='unsafe')
                            refined_small = self._refined_small

                            cv2.dilate(refined_small, self._dilate_kernel, dst=refined_small, iterations=1)

                            alpha_small = create_feathered_alpha(
                                refined_small,
                                inner_feather=self._inner_feather,
                                outer_feather=self._outer_feather,
                                max_out_alpha=MAX_OUT_ALPHA,
                                out_buffer=self._alpha_small
                            )

                            alpha_map = self._alpha_map
                            cv2.resize(alpha_small, (W, H), dst=alpha_map, interpolation=cv2.INTER_LINEAR)

                            refined_up = self._refined_up
                            cv2.resize(refined_small, (W, H), dst=refined_up, interpolation=cv2.INTER_NEAREST)

                            np.copyto(self._prev_small_mask, small_mask, casting='unsafe')
                            np.copyto(self._prev_alpha_map, alpha_map, casting='unsafe')
                            np.copyto(self._prev_refined_up, refined_up, casting='unsafe')
                        else:
                            alpha_map = self._alpha_map
                            np.copyto(alpha_map, self._prev_alpha_map, casting='unsafe')
                            refined_up = self._refined_up
                            np.copyto(refined_up, self._prev_refined_up, casting='unsafe')

                        alpha_map = self._alpha_map
                        refined_up = self._refined_up
                        if not reuse_mask_only or self._prev_bbox is None:
                            bbox = self._compute_bbox(refined_up, ROI_PADDING, W, H)
                        else:
                            bbox = self._prev_bbox

                        if bbox is None:
                            np.copyto(output_buf, frame_rgb, casting='unsafe')
                            final_rgb = output_buf
                            proc_dt = time.time() - ts0
                            self._prev_has_valid = False
                            self._last_render_params = None
                            self._prev_bbox = None
                        else:
                            np.copyto(output_buf, frame_rgb, casting='unsafe')
                            colorize_nail_texture(
                                frame_rgb,
                                alpha_map,
                                InferenceWorker.shared_color_rgb,
                                texture_strength=InferenceWorker.shared_texture,
                                brightness_adjust=1.05,
                                out=output_buf,
                                work_f32=self._work_f32,
                                blend_f32=self._blend_f32,
                                gray_u8=self._gray_u8,
                                bbox=bbox
                            )

                            if InferenceWorker.shared_sheen:
                                refined_up = self._refined_up
                                add_natural_sheen(
                                    output_buf,
                                    refined_up,
                                    intensity=InferenceWorker.shared_sheen_intensity,
                                    gloss_buffer=self._gloss_buffer,
                                    sheen_map=self._sheen_map,
                                    work_f32=self._sheen_work,
                                    out=output_buf,
                                    bbox=bbox
                                )

                            final_rgb = output_buf
                            proc_dt = time.time() - ts0
                            np.copyto(self._prev_output_rgb, output_buf, casting='unsafe')
                            self._prev_has_valid = True
                            self._last_render_params = render_params
                            self._prev_bbox = bbox

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
                time.sleep(0.05)

    def get_latest(self):
        with self.out_lock:
            return None if self.latest_result is None else dict(self.latest_result)

    def stop(self):
        self.running = False

    def _build_fallback_mask(self, outputs, H, W):
        fallback = self._fallback_mask
        fallback.fill(0)
        has_candidate = False
        tmp_mask = self._fallback_tmp
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
                cv2.resize(mask_uint8, (W, H), dst=tmp_mask, interpolation=cv2.INTER_LINEAR)
                _, tmp_thresh = cv2.threshold(tmp_mask, 127, 255, cv2.THRESH_BINARY)
                np.maximum(fallback, tmp_thresh, out=fallback)
                has_candidate = True
        if has_candidate:
            return fallback
        return None

    def _compute_bbox(self, mask, pad, W, H):
        points = cv2.findNonZero(mask)
        if points is None:
            return None
        x, y, w_box, h_box = cv2.boundingRect(points)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w_box + pad)
        y2 = min(H, y + h_box + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)


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
             infer_fps=INFER_FPS, mask_downscale=MASK_DOWNSCALE,
             capture_width=CAPTURE_WIDTH, capture_height=CAPTURE_HEIGHT):
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

    reader = CameraReader(src=cam_index, width=capture_width, height=capture_height)
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
            ok, frame = reader.read(timeout=0.02)
            latest = worker.get_latest()

            if ok and frame is not None:
                worker.submit(frame)
                if latest is not None:
                    display_rgb = latest['rgb']
                else:
                    display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                if latest is None:
                    time.sleep(0.005)
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
                status = f"INF {inf_ms:.0f}ms PRC {proc_ms:.0f}ms"

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
    ap.add_argument("--infer_fps", type=float, default=INFER_FPS, help="Max inference FPS (lower => faster UI).")
    ap.add_argument("--mask_downscale", type=float, default=MASK_DOWNSCALE,
                    help="Run heavy mask ops on downscaled mask (0.3-1.0).")
    ap.add_argument("--capture_w", type=int, default=CAPTURE_WIDTH, help="Camera capture width.")
    ap.add_argument("--capture_h", type=int, default=CAPTURE_HEIGHT, help="Camera capture height.")
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
        mask_downscale=args.mask_downscale,
        capture_width=args.capture_w,
        capture_height=args.capture_h
    )
