#!/usr/bin/env python3
"""Render docs/PHASE6_TECHNICAL_REPORT.md into a SINGLE self-contained HTML file.

Every figure is inlined as a base64 data URI, so the teacher can open one file
with no folder next to it and no network access. No external CSS/font/CDN is
referenced at all.

Run:  gpu_env/bin/python scripts/phase6_build_report_html.py
"""
import base64
import os
import re
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "PHASE6_TECHNICAL_REPORT.md")
DST = os.path.join(ROOT, "docs", "PHASE6_TECHNICAL_REPORT.html")

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

CSS = """
:root{
  --ink:#1a1f26; --muted:#5b6570; --line:#e2e6ea; --bg:#ffffff;
  --ours:#c0392b; --accent:#1f4e79; --soft:#f6f8fa;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:#eef1f4; color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
    "Hiragino Sans GB","Microsoft YaHei","Source Han Sans SC","Noto Sans CJK SC",
    "Helvetica Neue",Arial,sans-serif;
  font-size:15.5px; line-height:1.75;
}
.page{max-width:1080px; margin:0 auto; background:var(--bg);
  padding:54px 62px 90px; box-shadow:0 1px 26px rgba(20,35,55,.10)}
h1{font-size:26px; line-height:1.4; margin:0 0 6px; letter-spacing:-.2px;
   padding-bottom:14px; border-bottom:3px solid var(--accent)}
h2{font-size:20px; margin:44px 0 14px; padding:8px 0 8px 13px; letter-spacing:-.1px;
   border-left:5px solid var(--accent); background:linear-gradient(90deg,#f4f7fa,transparent)}
h3{font-size:16.5px; margin:30px 0 10px; color:#22303d}
h4{font-size:15px; margin:22px 0 8px; color:var(--muted)}
p{margin:11px 0}
a{color:var(--accent)}
code{background:var(--soft); border:1px solid var(--line); border-radius:4px;
     padding:1.5px 5px; font-size:13px;
     font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:#1d2530; color:#e6edf3; padding:15px 17px; border-radius:8px;
    overflow-x:auto; font-size:12.9px; line-height:1.62}
pre code{background:none; border:none; color:inherit; padding:0}
table{border-collapse:collapse; width:100%; margin:16px 0; font-size:13.4px}
th,td{border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top}
th{background:#f2f5f8; font-weight:600; white-space:nowrap}
tr:nth-child(even) td{background:#fafbfc}
blockquote{margin:16px 0; padding:12px 18px; background:#fbf7f3;
  border-left:4px solid #d9a441; border-radius:0 6px 6px 0; color:#4a4137; font-size:14.3px}
blockquote p{margin:5px 0}
ul,ol{margin:11px 0; padding-left:26px}
li{margin:5px 0}
hr{border:none; border-top:1px solid var(--line); margin:38px 0}
img{max-width:100%; height:auto; display:block; margin:20px auto 8px;
    border:1px solid var(--line); border-radius:7px; background:#fff}
strong{font-weight:650}
em{color:var(--muted)}
h1 + p, h1 + blockquote{font-size:15px}
"""


def inline_images(md_text):
    """Replace every local image link with a base64 data URI."""
    n = 0
    total = 0

    def repl(m):
        nonlocal n, total
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return m.group(0)
        full = os.path.normpath(os.path.join(ROOT, "docs", path))
        if not os.path.exists(full):
            print(f"  !! MISSING {path}", file=sys.stderr)
            return m.group(0)
        with open(full, "rb") as fh:
            raw = fh.read()
        ext = os.path.splitext(full)[1].lstrip(".").lower()
        mime = "image/png" if ext == "png" else f"image/{ext}"
        b64 = base64.b64encode(raw).decode("ascii")
        n += 1
        total += len(b64)
        return f"![{alt}](data:{mime};base64,{b64})"

    out = IMG_RE.sub(repl, md_text)
    print(f"  inlined {n} images ({total / 1024 / 1024:.2f} MB base64)")
    return out


def main():
    with open(SRC, encoding="utf-8") as fh:
        md_text = fh.read()

    md_text = inline_images(md_text)

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists", "toc"],
        extension_configs={"toc": {"permalink": False}},
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PercepFlex / MARGIN -- Phase 6 完整技术报告</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>
"""
    with open(DST, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  wrote {DST}  ({os.path.getsize(DST) / 1024 / 1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
