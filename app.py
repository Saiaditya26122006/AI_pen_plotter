import os

from flask import Flask, jsonify, render_template, request, send_file

from modules.advanced_pipeline import process_image_full_pipeline
from modules.text_pipeline import process_text_art
from modules.wave_pipeline import process_wave_art

app = Flask(__name__)

PROJECT_ROOT  = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR     = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "output")
UPLOADED_PATH = os.path.join(INPUT_DIR, "uploaded.png")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/fonts", methods=["GET"])
def fonts():
    """Return the list of font catalog entries available on this machine."""
    from modules.text_writer import detect_available_fonts
    return jsonify(detect_available_fonts())


@app.route("/process", methods=["POST"])
def process():
    style = request.form.get("style", "text")

    # ── Write mode: no image required ─────────────────────────────────────────
    if style == "write":
        text       = (request.form.get("write_text") or "").strip()
        font_key   = (request.form.get("font_key") or "").strip() or None
        try:
            font_h = float(request.form.get("font_h", 10))
            font_h = max(4.0, min(font_h, 60.0))
        except ValueError:
            font_h = 10.0
        position = request.form.get("text_pos", "bottom")
        align    = request.form.get("text_align", "center")

        if not text:
            return jsonify({"status": "error", "message": "No text provided"}), 400

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            from modules.write_pipeline import process_text_write
            gcode = process_text_write(
                text,
                font_h      = font_h,
                position    = position,
                align       = align,
                font_key    = font_key,
                interactive = False,   # web: never block on stdin
            )
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

        draw_moves = sum(1 for ln in gcode.splitlines() if ln.startswith("G1 X"))
        return jsonify({
            "status":        "done",
            "style":         "write",
            "draw_moves":    draw_moves,
            "filename":      "drawing_write.gcode",
            "gcode_preview": gcode[:2000],
        })

    # ── Art modes: image required ──────────────────────────────────────────────
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

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

    os.makedirs(INPUT_DIR,  exist_ok=True)
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
