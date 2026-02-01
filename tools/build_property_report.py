from __future__ import annotations

import base64
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# your single source of truth
MD_PATH = REPO_ROOT / "property_report.md"
JSON_PATH = REPO_ROOT / "output" / "property" / "report_placeholders.json"

# output html (for printing / sharing)
HTML_PATH = REPO_ROOT / "property_report.html"


def replace_placeholders(text: str, mapping: dict[str, str]) -> str:
    """
    Replace {{KEY}} with mapping[KEY]. If KEY not found, keep as-is.
    """
    def repl(m: re.Match) -> str:
        k = m.group(1).strip()
        v = mapping.get(k)
        return str(v) if v is not None else m.group(0)

    return re.sub(r"\{\{([^}]+)\}\}", repl, text)


def normalize_image_paths(md: str) -> str:
    """
    Make image paths repo-root relative and safe for HTML.
    Converts (output/...) -> (./output/...)
    Leaves http(s) images untouched.
    """
    def repl(m: re.Match) -> str:
        alt = m.group(1)
        path = m.group(2).strip()
        if path.startswith("http://") or path.startswith("https://"):
            return m.group(0)
        if path.startswith("./"):
            return f"![{alt}]({path})"
        if path.startswith("output/"):
            return f"![{alt}](./{path})"
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)


def md_to_html_minimal(md: str) -> str:
    """
    Minimal markdown -> HTML (no external packages).
    Supports:
      - #, ##, ### headings
      - bullet lists (- )
      - paragraphs
      - images ![]()
    Everything else is passed through as text.
    """
    lines = md.splitlines()
    out = []
    in_ul = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    img_pat = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    for raw in lines:
        line = raw.rstrip()

        # images (as block)
        m = img_pat.search(line)
        if m and line.strip().startswith("!"):
            close_ul()
            alt = m.group(1)
            src = m.group(2).strip()
            out.append(f'<div class="imgblock"><img alt="{alt}" src="{src}"></div>')
            continue

        # headings
        if line.startswith("# "):
            close_ul()
            out.append(f"<h1>{line[2:].strip()}</h1>")
            continue
        if line.startswith("## "):
            close_ul()
            out.append(f"<h2>{line[3:].strip()}</h2>")
            continue
        if line.startswith("### "):
            close_ul()
            out.append(f"<h3>{line[4:].strip()}</h3>")
            continue

        # bullet list
        if line.strip().startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{line.strip()[2:].strip()}</li>")
            continue
        else:
            close_ul()

        # horizontal rule
        if line.strip() == "---":
            out.append("<hr/>")
            continue

        # empty
        if not line.strip():
            out.append("<div class='sp'></div>")
            continue

        # paragraph
        out.append(f"<p>{line}</p>")

    close_ul()
    body = "\n".join(out)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Property Report</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,'Noto Sans',sans-serif; line-height: 1.35; margin: 28px; }}
  h1,h2,h3 {{ margin: 20px 0 10px; }}
  p {{ margin: 8px 0; }}
  ul {{ margin: 8px 0 12px 20px; }}
  hr {{ margin: 18px 0; border: none; border-top: 1px solid #eee; }}
  .imgblock {{ margin: 10px 0 16px; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #eee; padding: 6px; background: #fff; }}
  .sp {{ height: 8px; }}
  @media print {{
    body {{ margin: 12mm; }}
    img {{ page-break-inside: avoid; }}
    h2,h3 {{ page-break-after: avoid; }}
  }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def embed_images_as_data_uri(html: str) -> str:
    """
    Replace <img src="./output/...png"> with base64 data URIs.
    Guarantees images display in HTML anywhere (best for printing / sending screenshots).
    """
    def repl(m: re.Match) -> str:
        src = m.group(1)
        # only embed local png/jpg
        if src.startswith("http://") or src.startswith("https://"):
            return m.group(0)

        # resolve path relative to repo root
        rel = src
        if rel.startswith("./"):
            rel = rel[2:]
        img_path = REPO_ROOT / rel
        if not img_path.exists():
            # leave it unchanged; you can debug missing file
            return m.group(0)

        suffix = img_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else ("image/jpeg" if suffix in [".jpg", ".jpeg"] else None)
        if mime is None:
            return m.group(0)

        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'<img alt="" src="data:{mime};base64,{b64}">'

    # match <img ... src="...">
    return re.sub(r'<img[^>]*\s+src="([^"]+)"[^>]*>', repl, html)


def main() -> None:
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Missing: {MD_PATH}")
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Missing: {JSON_PATH}")

    mapping = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    md = MD_PATH.read_text(encoding="utf-8")

    md = normalize_image_paths(md)
    filled = replace_placeholders(md, mapping)

    # overwrite the SAME md (as you requested)
    MD_PATH.write_text(filled, encoding="utf-8")
    print(f"[ok] updated in-place: {MD_PATH}")

    # make html
    html = md_to_html_minimal(filled)

    # embed images so printing always works
    html = embed_images_as_data_uri(html)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[ok] wrote printable html: {HTML_PATH}")

    # quick diagnostics: list unresolved placeholders / missing images
    left = sorted(set(re.findall(r"\{\{([^}]+)\}\}", filled)))
    if left:
        print("[warn] unresolved placeholders:", left)

    # find local image paths and verify existence
    imgs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', filled)
    missing = []
    for s in imgs:
        if s.startswith("http"):
            continue
        rel = s
        if rel.startswith("./"):
            rel = rel[2:]
        p = REPO_ROOT / rel
        if not p.exists():
            missing.append(str(s))
    if missing:
        print("[warn] missing image files referenced in md:")
        for x in missing:
            print("  -", x)


if __name__ == "__main__":
    main()
