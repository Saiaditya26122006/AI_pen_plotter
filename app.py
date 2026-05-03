import os

from flask import Flask, jsonify, render_template, request, send_file

from modules.advanced_pipeline import process_image_full_pipeline
from modules.text_pipeline import process_text_art
from modules.wave_pipeline import process_wave_art

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
UPLOADED_PATH = os.path.join(INPUT_DIR, "uploaded.png")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    style = request.form.get("style", "text")
    word  = request.form.get("word", "VARSHEETHvarsheeth").strip() or "VARSHEETHvarsheeth"
    mode  = request.form.get("mode", "auto")
    chars = (request.form.get("chars", "ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip().upper()
             or "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    try:
        cell_h = int(request.form.get("cell_h", 14))
        cell_h = max(6, min(cell_h, 40))
    except ValueError:
        cell_h = 14
    try:
        row_spacing = int(request.form.get("row_spacing", 8))
        row_spacing = max(4, min(row_spacing, 20))
    except ValueError:
        row_spacing = 8

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file.save(UPLOADED_PATH)

    try:
        if style == "stipple":
            gcode    = process_image_full_pipeline(UPLOADED_PATH)
            filename = "drawing_stipple.gcode"
        elif style == "wave":
            gcode    = process_wave_art(UPLOADED_PATH, row_spacing=row_spacing)
            filename = "drawing_wave.gcode"
        else:
            gcode    = process_text_art(UPLOADED_PATH, word=word, cell_h=cell_h,
                                        mode=mode, chars=chars)
            filename = "drawing_text.gcode"
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    draw_moves = sum(1 for ln in gcode.splitlines() if ln.startswith("G1 X"))
    return jsonify({
        "status":        "done",
        "style":         style,
        "draw_moves":    draw_moves,
        "filename":      filename,
        "gcode_preview": gcode[:2000],
    })


@app.route("/download", methods=["GET"])
def download():
    filename = request.args.get("file", "drawing_text.gcode")
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"status": "error", "message": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(debug=True)
