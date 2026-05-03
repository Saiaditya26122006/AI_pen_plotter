import os
import sys


def main() -> None:
    """
    Usage:
        python main.py <image>                          # text art (default)
        python main.py <image> --style text             # text art
        python main.py <image> --style stipple          # 4-layer stipple pipeline
        python main.py <image> --style text --word HELLO
        python main.py <image> --style text --cell 12
    """
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    image_path = args[0]
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    style = "text"
    word = "VARSHEETHvarsheeth"
    cell_h = 14

    i = 1
    while i < len(args):
        if args[i] == "--style" and i + 1 < len(args):
            style = args[i + 1]; i += 2
        elif args[i] == "--word" and i + 1 < len(args):
            word = args[i + 1]; i += 2
        elif args[i] == "--cell" and i + 1 < len(args):
            cell_h = int(args[i + 1]); i += 2
        else:
            i += 1

    print(f"Image : {image_path}")
    print(f"Style : {style}")

    if style == "stipple":
        from modules.advanced_pipeline import process_image_full_pipeline
        gcode = process_image_full_pipeline(image_path)
        out_name = "drawing.gcode"
    else:
        from modules.text_pipeline import process_text_art
        print(f"Word  : {word}  |  cell_h={cell_h}")
        gcode = process_text_art(image_path, word=word, cell_h=cell_h)
        out_name = "drawing_text.gcode"

    draw_moves = sum(1 for ln in gcode.splitlines() if ln.startswith("G1 X"))
    print(f"\nTotal draw moves: {draw_moves}  →  output/{out_name}")


if __name__ == "__main__":
    main()
