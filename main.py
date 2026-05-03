import os
import sys


def main() -> None:
    """
    Usage:
        python main.py <image>                         # generate ALL 3 styles (default)
        python main.py <image> --style all             # same as above
        python main.py <image> --style stipple         # stipple/points only
        python main.py <image> --style wave            # sine-wave modulation only
        python main.py <image> --style text            # text art only (auto A-Z)
        python main.py <image> --style text --mode word --word HELLO
        python main.py <image> --style text --mode auto --chars ABCM
        python main.py <image> --style text --cell 12
    """
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    image_path = args[0]
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    style  = "all"
    word   = "VARSHEETHvarsheeth"
    cell_h = 14
    mode   = "auto"
    chars  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    row_spacing = 8

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
        elif args[i] == "--row-spacing" and i + 1 < len(args):
            row_spacing = int(args[i + 1]); i += 2
        else:
            i += 1

    # "both" kept for backward compat — treat same as "all"
    if style == "both":
        style = "all"

    do_stipple = style in ("stipple", "all")
    do_wave    = style in ("wave",    "all")
    do_text    = style in ("text",    "all")

    print(f"Image : {image_path}\n")

    if do_stipple:
        print("=" * 52)
        print("  STIPPLE / POINTS PIPELINE")
        print("=" * 52)
        from modules.advanced_pipeline import process_image_full_pipeline
        gcode_s = process_image_full_pipeline(image_path)
        moves_s = sum(1 for ln in gcode_s.splitlines() if ln.startswith("G1 X"))
        print(f"  -> {moves_s} draw moves  |  output/drawing_stipple.gcode\n")

    if do_wave:
        print("=" * 52)
        print("  WAVE ART PIPELINE")
        print("=" * 52)
        from modules.wave_pipeline import process_wave_art
        gcode_w = process_wave_art(image_path, row_spacing=row_spacing)
        moves_w = sum(1 for ln in gcode_w.splitlines() if ln.startswith("G1 X"))
        print(f"  -> {moves_w} draw moves  |  output/drawing_wave.gcode\n")

    if do_text:
        print("=" * 52)
        print("  TEXT ART PIPELINE")
        print("=" * 52)
        from modules.text_pipeline import process_text_art
        if mode == "auto":
            print(f"  Mode : auto A-Z  |  chars={chars}  |  cell_h={cell_h}")
        else:
            print(f"  Mode : word      |  word={word}  |  cell_h={cell_h}")
        gcode_t = process_text_art(image_path, word=word, cell_h=cell_h, mode=mode, chars=chars)
        moves_t = sum(1 for ln in gcode_t.splitlines() if ln.startswith("G1 X"))
        print(f"  -> {moves_t} draw moves  |  output/drawing_text.gcode\n")

    print("=" * 52)
    print("Output files saved to output/")
    if do_stipple:
        print("  drawing_stipple.gcode  <-- dots / Voronoi stippling")
    if do_wave:
        print("  drawing_wave.gcode     <-- sine-wave modulation (Joy Division)")
    if do_text:
        print("  drawing_text.gcode     <-- A-Z brightness-mapped letters")
    print("\nOpen all in NC Viewer and pick the one you prefer to plot.")


if __name__ == "__main__":
    main()
