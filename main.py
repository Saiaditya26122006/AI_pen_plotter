import os
import sys


def main() -> None:
    """
    Usage:
        python main.py <image>                         # generate BOTH (default)
        python main.py <image> --style both            # generate both explicitly
        python main.py <image> --style text            # text art only (word mode)
        python main.py <image> --style stipple         # stipple/points only
        python main.py <image> --style text --mode auto              # A-Z brightness
        python main.py <image> --style text --mode auto --chars ABCM
        python main.py <image> --style text --word HELLO --cell 12
    """
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    image_path = args[0]
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    style = "both"          # default: generate both outputs
    word  = "VARSHEETHvarsheeth"
    cell_h = 14
    mode  = "auto"          # default for text: auto A-Z brightness mapping
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    i = 1
    while i < len(args):
        if args[i] == "--style" and i + 1 < len(args):
            style = args[i + 1]; i += 2
        elif args[i] == "--word" and i + 1 < len(args):
            word = args[i + 1]; i += 2
        elif args[i] == "--cell" and i + 1 < len(args):
            cell_h = int(args[i + 1]); i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]; i += 2
        elif args[i] == "--chars" and i + 1 < len(args):
            chars = args[i + 1].upper(); i += 2
        else:
            i += 1

    print(f"Image : {image_path}")

    do_text    = style in ("text", "both")
    do_stipple = style in ("stipple", "both")

    if do_stipple:
        print("\n" + "="*50)
        print("STIPPLE / POINTS PIPELINE")
        print("="*50)
        from modules.advanced_pipeline import process_image_full_pipeline
        gcode_s = process_image_full_pipeline(image_path)
        moves_s = sum(1 for ln in gcode_s.splitlines() if ln.startswith("G1 X"))
        print(f"Stipple draw moves: {moves_s}  ->  output/drawing_stipple.gcode")

    if do_text:
        print("\n" + "="*50)
        print("TEXT ART PIPELINE")
        print("="*50)
        from modules.text_pipeline import process_text_art
        if mode == "auto":
            print(f"Mode  : auto A-Z  |  chars={chars}  |  cell_h={cell_h}")
        else:
            print(f"Mode  : word  |  word={word}  |  cell_h={cell_h}")
        gcode_t = process_text_art(image_path, word=word, cell_h=cell_h, mode=mode, chars=chars)
        moves_t = sum(1 for ln in gcode_t.splitlines() if ln.startswith("G1 X"))
        print(f"Text draw moves   : {moves_t}  ->  output/drawing_text.gcode")

    print("\n" + "="*50)
    print("Output files in output/ folder:")
    if do_stipple:
        print("  drawing_stipple.gcode  <-- points/dots design")
    if do_text:
        print("  drawing_text.gcode     <-- letter/text design")
    print("Open both in NC Viewer and pick the one you prefer.")


if __name__ == "__main__":
    main()
