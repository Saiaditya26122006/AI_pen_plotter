import os
import sys


def main() -> None:
    """
    Usage:
        python main.py <image>                         # generate ALL 3 art styles (default)
        python main.py <image> --style all             # same as above
        python main.py <image> --style stipple         # stipple / Voronoi dots only
        python main.py <image> --style wave            # sine-wave modulation only
        python main.py <image> --style text            # text art only (auto A-Z)
        python main.py <image> --style text --mode word --word HELLO
        python main.py <image> --style text --mode auto --chars ABCM
        python main.py <image> --style text --cell 12

        # Write readable text on paper (no image needed)
        python main.py --style write --text "Hello World"
        python main.py --style write --text "Line one\\nLine two" --font-size 12
        python main.py --style write --text "My caption" --text-pos top --text-align right
        python main.py --style write --text "Hello" --font impact          # use font key
        python main.py --style write --text "Hello" --font 3               # use menu number

        # Combine art with a text label
        python main.py <image> --style stipple --label "Portrait, 2024"
        python main.py <image> --style all --label "VARSHEETHA" --label-pos top
    """
    args = sys.argv[1:]
    if not args:
        print(main.__doc__)
        raise SystemExit(1)

    # ── defaults ──────────────────────────────────────────────────────────────
    style       = "all"
    word        = "VARSHEETHvarsheeth"
    cell_h      = 14
    mode        = "auto"
    chars       = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    row_spacing = 8

    # write / label options
    write_text  = ""
    font_size   = 10.0
    text_pos    = "bottom"
    text_align  = "center"
    font_key    = None      # key or 1-based number; None = interactive prompt
    label       = ""
    label_pos   = "bottom"
    label_align = "center"
    label_size  = 8.0
    label_font  = None

    # ── parse args ────────────────────────────────────────────────────────────
    image_path = None
    i = 0
    if args and not args[0].startswith("--"):
        image_path = args[0]
        i = 1

    while i < len(args):
        a = args[i]
        def _next():
            nonlocal i
            i += 2
            return args[i - 1]

        if   a == "--style"       and i + 1 < len(args): style       = _next()
        elif a == "--word"        and i + 1 < len(args): word        = _next()
        elif a == "--cell"        and i + 1 < len(args): cell_h      = int(_next())
        elif a == "--mode"        and i + 1 < len(args): mode        = _next()
        elif a == "--chars"       and i + 1 < len(args): chars       = _next().upper()
        elif a == "--row-spacing" and i + 1 < len(args): row_spacing = int(_next())
        elif a == "--text"        and i + 1 < len(args): write_text  = _next()
        elif a == "--font-size"   and i + 1 < len(args): font_size   = float(_next())
        elif a == "--text-pos"    and i + 1 < len(args): text_pos    = _next()
        elif a == "--text-align"  and i + 1 < len(args): text_align  = _next()
        elif a == "--label"       and i + 1 < len(args): label       = _next()
        elif a == "--label-pos"   and i + 1 < len(args): label_pos   = _next()
        elif a == "--label-align" and i + 1 < len(args): label_align = _next()
        elif a == "--label-size"  and i + 1 < len(args): label_size  = float(_next())
        elif a == "--font"        and i + 1 < len(args): font_key    = _next()
        elif a == "--label-font"  and i + 1 < len(args): label_font  = _next()
        else: i += 1

    if style == "both":
        style = "all"   # backward compat

    # ── WRITE mode — no image required ────────────────────────────────────────
    if style == "write":
        if not write_text:
            print("Error: --style write requires --text \"Your text here\"")
            raise SystemExit(1)
        print(f"Text : {write_text!r}\n")
        print("=" * 52)
        print("  TEXT WRITE PIPELINE")
        print("=" * 52)
        from modules.write_pipeline import process_text_write
        gcode_w = process_text_write(
            write_text,
            font_h      = font_size,
            position    = text_pos,
            align       = text_align,
            font_key    = font_key,
            interactive = True,   # CLI: prompt if font not specified
        )
        moves = sum(1 for ln in gcode_w.splitlines() if ln.startswith("G1 X"))
        print(f"  -> {moves} draw moves  |  output/drawing_write.gcode\n")
        print("=" * 52)
        print("Output files saved to output/")
        print("  drawing_write.gcode  <-- readable text writing")
        print("\nOpen in NC Viewer to preview before plotting.")
        return

    # ── All art styles require an image ──────────────────────────────────────
    if image_path is None:
        print("Error: an image path is required for this style")
        raise SystemExit(1)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

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

    # ── Optional label appended to every generated file ───────────────────────
    if label:
        print("=" * 52)
        print("  TEXT LABEL (appended to all outputs)")
        print("=" * 52)
        from modules.text_writer import text_write_gcode, resolve_font
        label_gcode = text_write_gcode(
            label,
            font_h    = label_size,
            position  = label_pos,
            align     = label_align,
            font_path = resolve_font(label_font) if label_font else None,
        )
        label_block = "\n".join(label_gcode) + "\n"

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        targets = []
        if do_stipple: targets.append("drawing_stipple.gcode")
        if do_wave:    targets.append("drawing_wave.gcode")
        if do_text:    targets.append("drawing_text.gcode")

        for fname in targets:
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Insert label G-code before the M2 end command
                content = content.rstrip()
                if content.endswith("M2"):
                    content = content[:-2] + label_block + "M2\n"
                else:
                    content += "\n" + label_block + "M2\n"
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  Label appended → {fname}")

        label_moves = sum(1 for ln in label_block.splitlines() if ln.startswith("G1 X"))
        print(f"  -> {label_moves} label draw moves\n")

    print("=" * 52)
    print("Output files saved to output/")
    if do_stipple: print("  drawing_stipple.gcode  <-- dots / Voronoi stippling")
    if do_wave:    print("  drawing_wave.gcode     <-- sine-wave modulation (Joy Division)")
    if do_text:    print("  drawing_text.gcode     <-- A-Z brightness-mapped letters")
    if label:      print(f'  (label "{label}" appended to each file above)')
    print("\nOpen all in NC Viewer and pick the one you prefer to plot.")


if __name__ == "__main__":
    main()
