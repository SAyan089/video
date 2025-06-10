from flask import Flask, request, send_file, abort
import tempfile, os, subprocess

app = Flask(__name__)

def compress_video(in_path, out_path, quality):
    crf = "23" if quality=="high" else "28"
    preset = "slow" if quality=="high" else "medium"
    subprocess.run([
      "ffmpeg", "-y", "-i", in_path,
      "-c:v","libx264", "-preset", preset,
      "-crf", crf, "-c:a", "copy", out_path
    ], check=True)

@app.route("/compress", methods=["POST"])
def compress():
    if "video" not in request.files: abort(400, "video missing")
    vid = request.files["video"]
    ext = os.path.splitext(vid.filename)[1].lower()
    if ext not in (".mp4",".mov"): abort(400,"only mp4/mov")
    quality = request.form.get("quality","medium")
    tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    vid.save(tmp_in.name)
    compress_video(tmp_in.name, tmp_out.name, quality)
    return send_file(tmp_out.name, as_attachment=True,
                     download_name=f"compressed_{quality}{ext}")
