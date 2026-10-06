#!/usr/bin/env python3
"""Convert self-invalidating-alignment-outline-v6.md to acl-v6/main.tex.

RETIRED (v6.1, 8-page restructure): `acl-v6/main.tex` is now hand-maintained as
the canonical source and diverges from the Markdown (compressed body, Experiments
1-2 moved to Appendix A, Experiment 3 as the centerpiece). Do NOT run this script
against the current paper -- it would overwrite the hand-edited main.tex with a
regeneration from the now-stale Markdown. Kept only for reference / earlier drafts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "self-invalidating-alignment-outline-v6.md"
OUT_DIR = ROOT / "acl-v6"
OUT_TEX = OUT_DIR / "main.tex"

# Map bracket citations to bib keys
CITE_MAP = {
    r"Kobak et al\., 2025": "kobak2025",
    r"Kousha and Thelwall, 2025": "kousha2025",
    r"Gray, 2024": "gray2024",
    r"Zhu et al\., 2025": "zhu2025",
    r"Akpinar et al\., 2026": "akpinar2026",
    r"Bradley and Terry, 1952": "bradley1952",
    r"Christiano et al\., 2017": "christiano2017",
    r"Ouyang et al\., 2022": "ouyang2022",
    r"Siththaranjan et al\., 2024": "siththaranjan2024",
    r"Sorensen et al\., 2024": "sorensen2024",
    r"MaxMin-RLHF, Chakraborty et al\., 2024": "chakraborty2024",
    r"PAL, Chen et al\., 2025": "chen2025",
    r"variational preference learning, Poddar et al\., 2024": "poddar2024",
    r"Poddar et al\., 2024": "poddar2024",
    r"Ghasemi and Crowley, 2026": "ghasemi2026",
    r"Lin and Gan, 2026": "lin2026",
    r"Raj, 2026": "raj2026",
    r"Perdomo et al\., 2020": "perdomo2020",
    r"Hardt et al\., 2016": "hardt2016",
    r"Mendler-Dünner et al\., 2020": "mendler2020",
    r"Lasry and Lions, 2007": "lasry2007",
    r"Hirsch, 1976": "hirsch1976",
    r"Frank, 1985": "frank1985",
    r"Heffetz, 2011": "heffetz2011",
    r"Juzek et al\., 2025": "juzek2025",
    r"Bahlous-Boldi et al\., 2024": "bahlous2024",
    r"Gao et al\., 2023": "gao2023",
    r"Kirk et al\., 2024": "kirk2024",
    r"Shumailov et al\., 2024": "shumailov2024",
}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[review]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}

\title{Performative Alignment: Preference Learning under Deployment-Dependent Utility}

\begin{document}
\maketitle
"""

POSTAMBLE = r"""
\bibliography{references}

\end{document}
"""


def convert_citations(text: str) -> str:
    def replace_bracket(m: re.Match) -> str:
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(";")]
        keys = []
        for part in parts:
            found = False
            for pattern, key in CITE_MAP.items():
                if re.search(pattern, part):
                    keys.append(key)
                    found = True
                    break
            if not found:
                keys.append("UNKNOWN")
        if len(keys) == 1:
            return f"\\citep{{{keys[0]}}}"
        return "\\citep{" + ", ".join(keys) + "}"

    return re.sub(r"\[([^\]]+)\]", replace_bracket, text)


def md_inline_to_latex(s: str) -> str:
    s = convert_citations(s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    s = re.sub(
        r"\[anonymous\.4open\.science/r/performative-alignment\]\((https://[^)]+)\)",
        r"\\url{\1}",
        s,
    )
    s = s.replace("---", "---")
    s = s.replace("–", "--")
    s = s.replace("…", "\\ldots{}")
    s = s.replace("§", "Section ")
    s = s.replace("US\\$0", "US\\$0")
    return s


def convert_math_blocks(text: str) -> str:
    text = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)
    text = re.sub(r"\$([^$\n]+)\$", r"\\(\1\\)", text)
    return text


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not all(set(c) <= set("-:") for c in row):
            rows.append(row)
        i += 1
    if not rows:
        return "", start
    ncol = len(rows[0])
    latex = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\begin{{tabular}}{{@{{}}l{'c' * (ncol - 1)}@{{}}}}", "\\toprule"]
    header = " & ".join(md_inline_to_latex(c) for c in rows[0]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    for row in rows[1:]:
        latex.append(" & ".join(convert_math_blocks(md_inline_to_latex(c)) for c in row) + " \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(latex), i


def convert_figure(line: str) -> str:
    m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
    if not m:
        return ""
    cap, path = m.group(1), m.group(2)
    rel = "../" + path.replace("\\", "/")
    label = re.sub(r"[^a-z0-9]+", "-", path.lower())[:40].strip("-")
    return (
        f"\\begin{{figure}}[t]\n"
        f"\\centering\n"
        f"\\includegraphics[width=\\columnwidth]{{{rel}}}\n"
        f"\\caption{{{md_inline_to_latex(cap)}}}\n"
        f"\\label{{fig:{label}}}\n"
        f"\\end{{figure}}"
    )


def convert_md_to_tex(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = [PREAMBLE]
    i = 0
    in_abstract = False
    skip_until_abstract = True

    while i < len(lines):
        line = lines[i]

        if skip_until_abstract:
            if line.strip() == "## Abstract":
                in_abstract = True
                skip_until_abstract = False
                out.append("\\begin{abstract}")
                i += 1
                continue
            i += 1
            continue

        if line.startswith("## Changes from"):
            break

        if line.strip() == "---":
            i += 1
            continue

        if line.startswith("## References"):
            out.append("\\section*{Acknowledgements}")
            out.append("Generative AI tools were used to assist with language revision, code development and debugging, and exploratory literature search. The author independently designed the study, verified the cited literature, reviewed all generated code and text, executed the experiments, and takes responsibility for the paper's content and results.")
            out.append(POSTAMBLE.strip().split("\\bibliography")[0])
            out.append("\\bibliography{references}")
            out.append("\\end{document}")
            break

        if line.startswith("## Limitations"):
            out.append("\\section*{Limitations}")
            i += 1
            continue

        if line.startswith("## Ethics Statement"):
            out.append("\\section*{Ethics Statement}")
            i += 1
            continue

        if line.startswith("## Acknowledgements"):
            i += 1
            continue

        if line.startswith("## "):
            title = line[3:].strip()
            if title == "Abstract":
                i += 1
                continue
            if in_abstract:
                out.append("\\end{abstract}")
                in_abstract = False
            if title.startswith("Limitations") or title.startswith("Ethics"):
                i += 1
                continue
            m = re.match(r"(\d+)\.\s+(.*)", title)
            if m:
                out.append(f"\\section{{{m.group(2)}}}")
            else:
                out.append(f"\\section{{{title}}}")
            i += 1
            continue

        if line.startswith("### "):
            title = line[4:].strip()
            out.append(f"\\subsection{{{title}}}")
            i += 1
            continue

        if line.strip().startswith("|"):
            tbl, i = parse_table(lines, i)
            out.append(tbl)
            continue

        if line.strip().startswith("!["):
            fig = convert_figure(line)
            if fig:
                out.append(fig)
            i += 1
            continue

        if line.strip().startswith(">"):
            quote = md_inline_to_latex(line.strip()[1:].strip())
            out.append(f"\\begin{{quote}}\n{convert_math_blocks(quote)}\n\\end{{quote}}")
            i += 1
            continue

        if line.strip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(convert_math_blocks(md_inline_to_latex(lines[i].strip()[2:])))
                i += 1
            out.append("\\begin{itemize}")
            for item in items:
                out.append(f"\\item {item}")
            out.append("\\end{itemize}")
            continue

        if re.match(r"^\d+\.\s", line.strip()):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(convert_math_blocks(md_inline_to_latex(re.sub(r"^\d+\.\s", "", lines[i].strip()))))
                i += 1
            out.append("\\begin{enumerate}")
            for item in items:
                out.append(f"\\item {item}")
            out.append("\\end{enumerate}")
            continue

        if not line.strip():
            out.append("")
            i += 1
            continue

        body = convert_math_blocks(md_inline_to_latex(line.strip()))
        out.append(body)
        i += 1

    return "\n\n".join(out) + "\n"


def main() -> None:
    md = MD_PATH.read_text(encoding="utf-8")
    tex = convert_md_to_tex(md)
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT_TEX} ({len(tex):,} bytes)")


if __name__ == "__main__":
    main()
