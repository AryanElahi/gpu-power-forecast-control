"""nb_outputs_to_md.py: write the outputs of every code cell of a saved notebook to one Markdown file.

Usage (save the notebook first, Cmd+S, because the script reads the .ipynb file on disk):
  in a notebook cell:   %run ../src/tools/nb_outputs_to_md.py 05_eda_full_v2.ipynb
  in Terminal:          python src/tools/nb_outputs_to_md.py notebooks/05_eda_full_v2.ipynb
Optional second argument: output path (default: same folder and name as the notebook, ending in _outputs.md).
Text outputs go into code blocks; figures are saved as PNG files next to the Markdown file and linked.
"""
import base64, json, re, sys
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def export(nb_path, md_path=None):
    nb_path = Path(nb_path)
    md_path = Path(md_path) if md_path else nb_path.with_name(nb_path.stem + "_outputs.md")
    img_dir = md_path.with_name(md_path.stem + "_figures")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out = [f"# Outputs of `{nb_path.name}`",
           f"Exported {datetime.now():%Y-%m-%d %H:%M} from the saved notebook file.", ""]
    n_code = n_img = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"]).strip()
        title = src.splitlines()[0].lstrip("# ").strip() if src else f"Cell {n_code}"
        out += [f"## {title}", f"_execution count: {cell.get('execution_count')}_", ""]
        if not cell.get("outputs"):
            out += ["(no output)", ""]
            continue
        for o in cell["outputs"]:
            kind = o.get("output_type")
            if kind == "stream":
                text = "".join(o.get("text", ""))
                label = "stderr" if o.get("name") == "stderr" else ""
                out += [f"```{label}", ANSI.sub("", text).rstrip(), "```", ""]
            elif kind in ("execute_result", "display_data"):
                data = o.get("data", {})
                if "image/png" in data:
                    n_img += 1
                    img_dir.mkdir(parents=True, exist_ok=True)
                    f = img_dir / f"cell{n_code:02d}_{n_img:02d}.png"
                    f.write_bytes(base64.b64decode(data["image/png"]))
                    out += [f"![{title}]({img_dir.name}/{f.name})", ""]
                elif "text/plain" in data:
                    out += ["```", "".join(data["text/plain"]).rstrip(), "```", ""]
            elif kind == "error":
                tb = ANSI.sub("", "\n".join(o.get("traceback", [])))
                out += [f"**ERROR: {o.get('ename')}: {o.get('evalue')}**", "```", tb.rstrip(), "```", ""]
    md_path.write_text("\n".join(out), encoding="utf-8")
    print(f"written {md_path} | code cells {n_code} | figures {n_img}" + (f" in {img_dir}" if n_img else ""))
    return md_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
