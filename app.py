import os

import cv2
from flask import Flask, jsonify, render_template, request, send_file

from modules.gcode_generator import generate_gcode
from modules.subject_detection import detect_and_crop_subject


app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
UPLOADED_PATH = os.path.join(INPUT_DIR, "uploaded.png")
CROPPED_PATH = os.path.join(OUTPUT_DIR, "cropped.png")
GCODE_PATH = os.path.join(OUTPUT_DIR, "drawing.gcode")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file.save(UPLOADED_PATH)

    detect_and_crop_subject(UPLOADED_PATH)
    img = cv2.imread(CROPPED_PATH)
    if img is None:
        return jsonify({"status": "error", "message": "Could not load cropped image"}), 500
    gcode_string = generate_gcode(img)

    return jsonify({"status": "done", "gcode_preview": gcode_string[:500]})


@app.route("/download", methods=["GET"])
def download():
    if not os.path.isfile(GCODE_PATH):
        return jsonify({"status": "error", "message": "G-code file not found"}), 404
    return send_file(GCODE_PATH, as_attachment=True, download_name="drawing.gcode")


if __name__ == "__main__":
    app.run(debug=True)
