import argparse
import multiprocessing
import os
import queue
import threading
import time
import traceback

import cv2
import numpy as np
import tensorflow as tf


# ---------------- CONFIG ----------------
DEFAULT_MODEL = r"C:\Users\M.I TECH\Desktop\final\nails_seg_s_yolov8_v1_float16.tflite"
TARGET_CAM = 0
DISPLAY_WINDOW = "Virtual Nail Polish. Press Q to quit."
DESIRED_FPS = 25
INFER_FPS = 0  # 0 = run inference as fast as possible
TFLITE_THREADS = max(1, multiprocessing.cpu_count() - 1)
NAIL_COLOR = (199, 21, 133)
NAIL_COLOR_BGR = NAIL_COLOR[::-1]
TEXTURE_DEFAULT = 0.35
DILATION_PIXELS = 4
MASK_DOWNSCALE = 0.5
DELEGATE_MODE = "auto"
MAX_ACTIVE_MASKS = 10
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60
CAMERA_BUFFER_SIZE = 1
ALPHA_CACHE_THRESHOLD = 0.002

cv2.setUseOptimized(True)
try:
    DEFAULT_CAM_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
except AttributeError:
    DEFAULT_CAM_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


# ---------------- Helper functions ----------------
def colorize_nail_texture(orig_bgr, alpha, color_bgr, texture_strength=0.35, brightness_adjust=1.05):
    img_f = orig_bgr.astype(np.float32) * brightness_adjust
    color = np.asarray(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    color_mix = img_f * texture_strength + color * (1.0 - texture_strength)
    alpha_3 = alpha[:, :, None]
    out = img_f * (1.0 - alpha_3) + color_mix * alpha_3
    return np.clip(out, 0, 255).astype(np.uint8)


def add_natural_sheen(img_bgr, hard_mask, intensity=0.04):
    gloss = np.zeros(hard_mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 120:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx = x + int(w * 0.4)
        cy = y + int(h * 0.25)
        rx = max(1, int(w * 0.35))
        ry = max(1, int(h * 0.18))
        cv2.ellipse(gloss, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    g = (gloss.astype(np.float32) / 255.0) * intensity
    g3 = np.stack([g, g, g], axis=2)
    out = img_bgr.astype(np.float32) * (1 - g3) + 255.0 * g3
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------- TFLite helpers ----------------
_DELEGATE_CANDIDATES = {
    "gpu": (
        "tensorflowlite_gpu_delegate.dll",
        "tensorflowlite_gpu.dll",
        "libtensorflowlite_gpu_delegate.so",
        "libtensorflowlite_gpu_delegate.dylib",
    ),
    "nnapi": ("nnapi_delegate.dll", "libnnapi_delegate.so"),
}


def _try_load_delegate(names):
    for name in names:
        try:
            delegate = tf.lite.experimental.load_delegate(name)
            print(f"[TFLite] Using delegate: {name}")
            return delegate
        except (OSError, ValueError, AttributeError):
            continue
    return None


def load_tflite_interpreter(path, num_threads=TFLITE_THREADS, delegate_mode=DELEGATE_MODE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"TFLite model not found at: {path}")

    delegate_mode = (delegate_mode or "cpu").lower()
    delegates = []
    chosen = "cpu"

    if delegate_mode in ("auto", "gpu"):
        gpu_delegate = _try_load_delegate(_DELEGATE_CANDIDATES["gpu"])
        if gpu_delegate:
            delegates.append(gpu_delegate)
            chosen = "gpu"

    if not delegates and delegate_mode in ("auto", "nnapi"):
        nnapi_delegate = _try_load_delegate(_DELEGATE_CANDIDATES["nnapi"])
        if nnapi_delegate:
            delegates.append(nnapi_delegate)
            chosen = "nnapi"

    use_threads = 1 if delegates else num_threads
    interpreter = tf.lite.Interpreter(
        model_path=path,
        num_threads=use_threads,
        experimental_delegates=delegates if delegates else None,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[TFLite] delegate={chosen} threads={use_threads}")
    return interpreter, input_details, output_details


def _resolve_mask_outputs(outputs, cached_det_idx, cached_proto_idx):
    if cached_det_idx is not None and cached_proto_idx is not None:
        if cached_det_idx < len(outputs) and cached_proto_idx < len(outputs):
            return cached_det_idx, cached_proto_idx
    from_idx, proto_idx, _ = parse_tflite_outputs(outputs)
    return from_idx, proto_idx


# ---------------- Video reader thread ----------------
class CameraReader(threading.Thread):
    def __init__(self, src=0, width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
                 backend=None, buffer_size=CAMERA_BUFFER_SIZE):
        super().__init__(daemon=True)
        self.backend = backend if backend is not None else DEFAULT_CAM_BACKEND
        self.cap = self._open_capture(src)
        self._configure_capture(width, height, buffer_size)
        self.q = queue.Queue(maxsize=1)
        self.running = True

    def _open_capture(self, src):
        cap = cv2.VideoCapture(src, self.backend)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(src)
        return cap

    def _configure_capture(self, width, height, buffer_size):
        props = (
            (cv2.CAP_PROP_FRAME_WIDTH, width),
            (cv2.CAP_PROP_FRAME_HEIGHT, height),
            (cv2.CAP_PROP_BUFFERSIZE, max(1, buffer_size)),
            (cv2.CAP_PROP_FPS, CAMERA_FPS),
        )
        for prop, value in props:
            try:
                self.cap.set(prop, value)
            except Exception:
                pass
        self._flush()

    def _flush(self):
        if not self.cap or not self.cap.isOpened():
            return
        for _ in range(3):
            self.cap.grab()

    def run(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self._flush()
                time.sleep(0.01)
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout=0.01):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ---------------- Inference worker ----------------
class InferenceWorker(threading.Thread):
    shared_color_bgr = NAIL_COLOR_BGR
    shared_texture = TEXTURE_DEFAULT
    shared_sheen = False
    shared_sheen_intensity = 0.04

    def __init__(self, model_path, num_threads=TFLITE_THREADS,
                 infer_fps=INFER_FPS, mask_downscale=MASK_DOWNSCALE,
                 delegate_mode=DELEGATE_MODE):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.infer_fps = float(infer_fps)
        self.mask_downscale = float(mask_downscale)
        self.in_q = queue.Queue(maxsize=1)
        self.out_lock = threading.Lock()
        self.latest_result = None
        self.running = True

        self.interpreter, input_details, output_details = load_tflite_interpreter(
            self.model_path, num_threads=num_threads, delegate_mode=delegate_mode
        )
        self.input_detail = input_details[0]
        self.output_details = output_details

        input_shape = self.input_detail["shape"]
        if len(input_shape) != 4:
            raise ValueError("Expected NHWC input shape for TFLite model")
        self.in_h, self.in_w = int(input_shape[1]), int(input_shape[2])
        self.input_dtype = self.input_detail["dtype"]

        if self.input_dtype == np.uint8:
            self.input_tensor = np.empty(input_shape, dtype=np.uint8)
            self._float_buffer = None
        else:
            self.input_tensor = np.empty(input_shape, dtype=np.float32)
            self._float_buffer = np.empty((self.in_h, self.in_w, input_shape[3]), dtype=np.float32)

        self.det_idx = None
        self.proto_idx = None

        self._dilate_kernel = None
        self._blur_kernel = None
        self._blur_sigma = None
        self._alpha_cache_mask = None
        self._alpha_cache = None

        self._prev_gray_small = None
        self._motion_threshold = 6.5

    def _ensure_cached_params(self, small_h, small_w):
        dilate = max(1, int(self.mask_downscale * DILATION_PIXELS * 2))
        if dilate % 2 == 0:
            dilate += 1
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))

        blur = max(3, int(self.mask_downscale * 26))
        if blur % 2 == 0:
            blur += 1
        self._blur_kernel = (blur, blur)
        self._blur_sigma = max(0.5, self.mask_downscale * 9.0)

        if self._alpha_cache_mask is not None and self._alpha_cache_mask.shape != (small_h, small_w):
            self._alpha_cache_mask = None
            self._alpha_cache = None

    def _alpha_for(self, mask_small):
        if self._alpha_cache_mask is not None:
            if self._alpha_cache_mask.shape != mask_small.shape:
                self._alpha_cache_mask = None
                self._alpha_cache = None

        if self._alpha_cache_mask is not None:
            diff = cv2.countNonZero(cv2.bitwise_xor(self._alpha_cache_mask, mask_small))
            if diff <= mask_small.size * ALPHA_CACHE_THRESHOLD:
                return self._alpha_cache

        mask_norm = mask_small.astype(np.float32) / 255.0
        if self._dilate_kernel is not None:
            mask_norm = cv2.dilate(mask_norm, self._dilate_kernel, iterations=1)
        alpha = cv2.GaussianBlur(mask_norm, self._blur_kernel, self._blur_sigma)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        self._alpha_cache_mask = mask_small.copy()
        self._alpha_cache = alpha
        return alpha

    def submit(self, frame_bgr):
        if frame_bgr is None:
            return
        if not self.in_q.empty():
            try:
                self.in_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.in_q.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def _motion_gate(self, frame_small_gray):
        if self._prev_gray_small is None:
            self._prev_gray_small = frame_small_gray
            return True
        diff = cv2.absdiff(self._prev_gray_small, frame_small_gray)
        mean_diff = float(diff.mean())
        self._prev_gray_small = frame_small_gray
        return mean_diff > self._motion_threshold

    def run(self):
        last_inf = 0.0
        while self.running:
            try:
                now = time.time()
                if self.infer_fps > 0 and (now - last_inf) < (1.0 / self.infer_fps):
                    time.sleep(0.002)
                    continue
                try:
                    frame_bgr = self.in_q.get(timeout=0.08)
                except queue.Empty:
                    continue

                ts0 = time.time()
                resized_bgr = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
                resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

                small_w = max(8, int(frame_bgr.shape[1] * self.mask_downscale))
                small_h = max(8, int(frame_bgr.shape[0] * self.mask_downscale))
                self._ensure_cached_params(small_h, small_w)

                small_gray = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                small_gray = cv2.cvtColor(small_gray, cv2.COLOR_BGR2GRAY)

                if not self._motion_gate(small_gray):
                    with self.out_lock:
                        if self.latest_result is not None:
                            self.latest_result['bgr'] = frame_bgr
                            self.latest_result['ts'] = time.time()
                    last_inf = time.time()
                    continue

                if self.input_dtype == np.uint8:
                    np.copyto(self.input_tensor[0], resized_rgb, casting="unsafe")
                else:
                    self._float_buffer[...] = resized_rgb
                    self._float_buffer *= 1.0 / 255.0
                    np.copyto(self.input_tensor[0], self._float_buffer)

                inf_start = time.time()
                self.interpreter.set_tensor(self.input_detail["index"], self.input_tensor)
                self.interpreter.invoke()
                outputs = [self.interpreter.get_tensor(od["index"]) for od in self.output_details]
                inf_dt = time.time() - inf_start

                self.det_idx, self.proto_idx = _resolve_mask_outputs(outputs, self.det_idx, self.proto_idx)

                small_mask = self._build_small_mask(outputs, small_h, small_w)
                if small_mask is None or not small_mask.any():
                    final_bgr = frame_bgr
                    proc_dt = time.time() - ts0
                else:
                    _, mask_binary = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)
                    alpha_small = self._alpha_for(mask_binary)
                    alpha_map = cv2.resize(alpha_small, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)

                    colored_bgr = colorize_nail_texture(
                        frame_bgr,
                        alpha_map,
                        InferenceWorker.shared_color_bgr,
                        texture_strength=InferenceWorker.shared_texture,
                        brightness_adjust=1.03,
                    )

                    if InferenceWorker.shared_sheen:
                        refined_up = cv2.resize(mask_binary, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                        final_bgr = add_natural_sheen(
                            colored_bgr,
                            refined_up,
                            intensity=InferenceWorker.shared_sheen_intensity
                        )
                    else:
                        final_bgr = colored_bgr

                    proc_dt = time.time() - ts0

                with self.out_lock:
                    self.latest_result = {
                        "bgr": final_bgr,
                        "ts": time.time(),
                        "inf_dt": inf_dt,
                        "proc_dt": proc_dt,
                    }
                last_inf = time.time()
            except Exception as exc:
                print("Worker error:", exc)
                traceback.print_exc()
                time.sleep(0.04)

    def _build_small_mask(self, outputs, small_h, small_w):
        combined_small = np.zeros((small_h, small_w), dtype=np.uint8)

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
                    scores = det[:, 4]
                    boxes = det[:, :4]
                    mask_coeffs = det[:, -P:]
                else:
                    scores = det[:, 4] if det.shape[1] > 4 else np.zeros(det.shape[0])
                    boxes = det[:, :4] if det.shape[1] >= 4 else np.zeros((det.shape[0], 4))
                    mask_coeffs = det[:, -P:]

                valid_idx = np.where(scores > 0.27)[0]
                if valid_idx.size > 0:
                    scores_valid = scores[valid_idx]
                    order = np.argsort(scores_valid)[::-1][:MAX_ACTIVE_MASKS]
                    sel_idx = valid_idx[order]

                    mask_coeffs = mask_coeffs[sel_idx]
                    boxes = boxes[sel_idx]

                    ph, pw, _ = proto.shape
                    proto_flat = proto.reshape(-1, P)
                    mask_logits = proto_flat @ mask_coeffs.T
                    mask_stack = 1.0 / (1.0 + np.exp(-mask_logits))
                    mask_stack = mask_stack.reshape(ph, pw, -1)

                    scale_x_small = small_w / float(self.in_w)
                    scale_y_small = small_h / float(self.in_h)

                    for idx in range(mask_stack.shape[-1]):
                        mask = mask_stack[:, :, idx]
                        mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                        mask_small = np.clip(mask_small * 255.0, 0, 255).astype(np.uint8)

                        cx, cy, bw, bh = boxes[idx]
                        if cx > 1.5 or cy > 1.5 or bw > 1.5 or bh > 1.5:
                            x1 = cx - bw / 2.0
                            y1 = cy - bh / 2.0
                            x2 = cx + bw / 2.0
                            y2 = cy + bh / 2.0
                        else:
                            x1 = (cx - bw / 2.0) * self.in_w
                            y1 = (cy - bh / 2.0) * self.in_h
                            x2 = (cx + bw / 2.0) * self.in_w
                            y2 = (cy + bh / 2.0) * self.in_h

                        x1 = float(np.clip(x1, 0, self.in_w))
                        y1 = float(np.clip(y1, 0, self.in_h))
                        x2 = float(np.clip(x2, 0, self.in_w))
                        y2 = float(np.clip(y2, 0, self.in_h))

                        sx1 = int(np.clip(round(x1 * scale_x_small), 0, small_w))
                        sy1 = int(np.clip(round(y1 * scale_y_small), 0, small_h))
                        sx2 = int(np.clip(round(x2 * scale_x_small), 0, small_w))
                        sy2 = int(np.clip(round(y2 * scale_y_small), 0, small_h))

                        if sx2 > sx1 and sy2 > sy1:
                            roi = combined_small[sy1:sy2, sx1:sx2]
                            combined_small[sy1:sy2, sx1:sx2] = np.maximum(
                                roi,
                                mask_small[sy1:sy2, sx1:sx2]
                            )
                        else:
                            combined_small = np.maximum(combined_small, mask_small)

        if combined_small.any():
            return combined_small

        fallback = np.zeros((small_h, small_w), dtype=np.uint8)
        for idx, o in enumerate(outputs):
            if idx in (self.proto_idx, self.det_idx):
                continue
            candidate = None
            if o.ndim == 4 and o.shape[0] == 1:
                candidate = o[0]
            elif o.ndim == 3:
                candidate = o
            if candidate is None or candidate.ndim != 3 or candidate.shape[0] > 200:
                continue
            for m in candidate:
                if m.ndim != 2:
                    continue
                mask_small = cv2.resize(m, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                mask_small = np.clip(mask_small * 255.0, 0, 255).astype(np.uint8)
                fallback = np.maximum(fallback, mask_small)
        return fallback if fallback.any() else None

    def get_latest(self):
        with self.out_lock:
            return None if self.latest_result is None else dict(self.latest_result)

    def stop(self):
        self.running = False


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
             add_sheen=False, sheen_intensity=0.04, texture_amt=TEXTURE_DEFAULT,
             infer_fps=INFER_FPS, mask_downscale=MASK_DOWNSCALE, delegate=DELEGATE_MODE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at: {model_path}")

    hexv = color_hex.lstrip('#')
    if len(hexv) == 6:
        try:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            color_bgr = (b, g, r)
        except ValueError:
            color_bgr = NAIL_COLOR_BGR
    else:
        color_bgr = NAIL_COLOR_BGR

    InferenceWorker.shared_color_bgr = color_bgr
    InferenceWorker.shared_texture = texture_amt
    InferenceWorker.shared_sheen = bool(add_sheen)
    InferenceWorker.shared_sheen_intensity = float(sheen_intensity)

    reader = CameraReader(src=cam_index, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    reader.start()

    worker = InferenceWorker(
        model_path,
        num_threads=TFLITE_THREADS,
        infer_fps=infer_fps,
        mask_downscale=mask_downscale,
        delegate_mode=delegate,
    )
    worker.start()

    print("Model loaded. Starting camera. Press Q to quit.")
    last_time = time.time()
    fps_avg = 0.0
    try:
        while True:
            start = time.time()
            ok, frame = reader.read(timeout=0.01)
            latest = worker.get_latest()

            if ok and frame is not None:
                worker.submit(frame)
                display_bgr = latest["bgr"] if latest is not None else frame
            else:
                if latest is None:
                    time.sleep(0.003)
                    continue
                display_bgr = latest["bgr"]

            now = time.time()
            dt = now - last_time
            last_time = now
            fps_curr = 1.0 / dt if dt > 0 else 0.0
            fps_avg = fps_avg * 0.82 + fps_curr * 0.18
            text = f"FPS {fps_avg:.1f}"

            status = "INF - PRC -"
            if latest is not None:
                proc_ms = latest.get("proc_dt", 0.0) * 1000.0
                inf_ms = latest.get("inf_dt", 0.0) * 1000.0
                status = f"INF {inf_ms:.0f}ms PRC {proc_ms:.0f}ms"

            disp = display_bgr.copy()
            cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow(DISPLAY_WINDOW, disp)

            target_frame_time = 1.0 / float(desired_fps)
            loop_time = time.time() - start
            if loop_time < target_frame_time:
                time.sleep(max(0, target_frame_time - loop_time))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
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
    ap.add_argument("--sheen_int", type=float, default=0.04, help="Sheen intensity.")
    ap.add_argument("--texture", type=float, default=TEXTURE_DEFAULT, help="Texture preservation.")
    ap.add_argument("--infer_fps", type=float, default=INFER_FPS, help="Max inference FPS (0 = unlimited).")
    ap.add_argument("--mask_downscale", type=float, default=MASK_DOWNSCALE,
                    help="Run heavy mask ops on downscaled mask (0.3-1.0).")
    ap.add_argument("--delegate", type=str, default=DELEGATE_MODE,
                    choices=("auto", "gpu", "nnapi", "cpu"),
                    help="Preferred TFLite delegate backend.")
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
        delegate=args.delegate,
    )
