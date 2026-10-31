#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把中英文研究陈述编译成 PDF（复用 rp_latex 的 pandoc + xelatex + ctexart 方案）。"""
import os, re, subprocess, shutil, sys

BASE = "/Users/wangfengyi/Documents/work_data_mac/wangzhe_fintech2/AsyncMamba"
TMP = "/tmp/rs_build"
os.makedirs(TMP, exist_ok=True)

PANDOC = shutil.which("pandoc") or "/tmp/pandoc_dist/pandoc-3.12.1-arm64/bin/pandoc"
XELATEX = shutil.which("xelatex") or "/Library/TeX/texbin/xelatex"

PREAMBLE = r"""\documentclass[UTF8,a4paper,fontset=none,zihao=-4]{ctexart}
\usepackage[left=2.5cm,right=2.5cm,top=2.6cm,bottom=2.6cm]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setCJKmainfont{Songti SC}
\setCJKsansfont{Hiragino Sans GB}
\setmonofont{Menlo}[Scale=0.86]
\usepackage{amsmath,amssymb,mathtools}
\usepackage{xcolor}
\definecolor{deepblue}{RGB}{16,54,102}
\usepackage{hyperref}
\hypersetup{colorlinks=true,urlcolor=deepblue,pdftitle={__TITLE__}}
\usepackage{titlesec}
\setcounter{secnumdepth}{-1}
\titleformat{\section}[hang]
  {\sffamily\bfseries\fontsize{13.5}{17}\selectfont\color{deepblue}}
  {}{0pt}{}
  [\vspace{2pt}{\color{deepblue!30}\rule{\linewidth}{0.6pt}}]
\titlespacing*{\section}{0pt}{13pt plus 3pt minus 2pt}{8pt}
\usepackage{enumitem}
\setlist{itemsep=2pt,topsep=3pt,parsep=0pt,leftmargin=1.9em}
\linespread{1.26}\selectfont
\setlength{\parindent}{__PARINDENT__}
\setlength{\parskip}{__PARSKIP__}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\begin{document}
\sloppy

__TITLEBLOCK__
"""

TITLEBLOCK = r"""\begin{center}
{\sffamily\bfseries\fontsize{19}{25}\selectfont\color{deepblue} __TITLE__\par}
\vspace{6pt}
{\sffamily\fontsize{11.5}{15}\selectfont\color{black!58} __SUBTITLE__\par}
\vspace{5pt}
{\fontsize{10.5}{14}\selectfont\color{black!70} __AUTHOR__\par}
\end{center}
\vspace{2pt}
{\color{deepblue!30}\rule{\linewidth}{0.7pt}}\par
\vspace{10pt}
"""


def compile_one(src_md, out_pdf, title, subtitle, author, parindent, parskip, extra_body_pre=""):
    text = open(src_md, encoding="utf-8").read()
    lines = text.split("\n")
    # 去掉文首标题块（第一个 ## 之前的部分），正文交给 LaTeX 自动编号
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("## "))
    body_md = "\n".join(lines[idx:])
    body_md = body_md.replace("\u2212", "-")  # U+2212 减号 -> 普通连字符，避免缺字
    body_path = os.path.join(TMP, "body.md")
    open(body_path, "w", encoding="utf-8").write(body_md)

    tex_path = os.path.join(TMP, "body.tex")
    subprocess.run([PANDOC, body_path,
                    "-f", "markdown+pipe_tables+tex_math_dollars",
                    "-t", "latex", "--syntax-highlighting=none",
                    "--shift-heading-level-by=-1", "-o", tex_path],
                   check=True)
    body_tex = open(tex_path, encoding="utf-8").read()

    tb = (TITLEBLOCK
          .replace("__TITLE__", title)
          .replace("__SUBTITLE__", subtitle)
          .replace("__AUTHOR__", author))
    preamble = (PREAMBLE
                .replace("__TITLEBLOCK__", tb)
                .replace("__TITLE__", title)
                .replace("__PARINDENT__", parindent)
                .replace("__PARSKIP__", parskip))

    main_tex = preamble + extra_body_pre + body_tex + "\n\\end{document}\n"
    main_path = os.path.join(TMP, "main.tex")
    open(main_path, "w", encoding="utf-8").write(main_tex)

    for _ in range(2):
        subprocess.run([XELATEX, "-interaction=nonstopmode",
                        "-output-directory", TMP, main_path],
                       cwd=TMP, check=False)

    pdf = os.path.join(TMP, "main.pdf")
    if not os.path.exists(pdf):
        print(f"[FAIL] 未生成 PDF: {out_pdf}")
        return
    shutil.copy(pdf, out_pdf)
    log = open(os.path.join(TMP, "main.log"), encoding="utf-8",
               errors="ignore").read()
    m = re.search(r"Output written on .*?\((\d+) page", log)
    pages = m.group(1) if m else "?"
    print(f"[ok] {out_pdf}  ({pages} pages)")
    missing = set(re.findall(r"Missing character: There is no (.*?) in font", log))
    for mm in missing:
        print(f"  [warn] 缺字: {mm}")


compile_one(
    os.path.join(BASE, "research_statement_en.md"),
    os.path.join(BASE, "research_statement_en.pdf"),
    "Research Statement — Wang Zhe",
    "Cross-market equity return prediction with asynchronous spatio-temporal modeling",
    r"Wang Zhe · GitHub: \url{https://github.com/Afjes/AsyncMamba}",
    "0pt", "0.18em",
    r"\titlespacing*{\section}{0pt}{7pt plus 2pt minus 1pt}{4pt}\linespread{1.10}\selectfont",
)

compile_one(
    os.path.join(BASE, "research_statement_zh.md"),
    os.path.join(BASE, "research_statement_zh.pdf"),
    "研究陈述 — 王哲",
    "面向跨市场股票收益预测的异步传导时空建模",
    r"王哲 · 代码库：\url{https://github.com/Afjes/AsyncMamba}",
    "2em", "0.18em",
)
