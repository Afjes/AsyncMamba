#!/usr/bin/env bash
# ------------------------------------------------------------
# 把 research_statement_{zh,en}.md 编译成 PDF / Word
#
# 依赖（macOS）:
#   brew install pandoc
#   brew install --cask mactex     # 或 tectonic / tinytex
#
# 用法:
#   ./build_docs.sh pdf          # md -> pdf（中英文各一份）
#   ./build_docs.sh docx         # md -> docx（中英文各一份）
#   ./build_docs.sh all          # 两者都生成
#   ./build_docs.sh pdf zh       # 只生成中文 PDF
#   ./build_docs.sh pdf en       # 只生成英文 PDF
#
# 自定义字体（Linux 上通常需要）:
#   CJK_FONT="Noto Serif CJK SC" MONO_FONT="DejaVu Sans Mono" ./build_docs.sh pdf
# 输出到别的目录:
#   OUT_DIR=/tmp/out ./build_docs.sh all
# ------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

CJK_FONT="${CJK_FONT:-Songti SC}"   # macOS 自带; Linux 用 "Noto Serif CJK SC"
MONO_FONT="${MONO_FONT:-Menlo}"     # macOS 自带; Linux 用 "DejaVu Sans Mono"
OUT_DIR="${OUT_DIR:-.}"
mkdir -p "$OUT_DIR"

# --shift-heading-level-by=-1 把 md 的 H1 提升为 LaTeX 的 \title，
# 避免「title block + 正文 H1」重复出现两次标题。
PDF_OPTS=(
  --pdf-engine=xelatex
  --shift-heading-level-by=-1
  -V CJKmainfont="$CJK_FONT"
  -V CJKmonofont="$CJK_FONT"
  -V monofont="$MONO_FONT"
  -V geometry:a4paper,margin=2.2cm
  -V fontsize=11pt
  -V colorlinks=true
  -V linkcolor=blue
  -V urlcolor=blue
)

pick() {  # $1 = zh | en  ->  输出 "文件名|标题"
  case "$1" in
    zh) echo "research_statement_zh.md|研究陈述 — 王哲" ;;
    en) echo "research_statement_en.md|Research Statement — Wang Zhe" ;;
    *)  echo "未知语言: $1（可选 zh / en）" >&2; exit 1 ;;
  esac
}

ACTION="${1:-all}"
TARGETS="${2:-zh en}"

for lang in $TARGETS; do
  IFS='|' read -r md title <<< "$(pick "$lang")"
  if [ ! -f "$md" ]; then
    echo "跳过：$md 不存在"
    continue
  fi
  base="${md%.md}"
  out_base="$OUT_DIR/$(basename "$base")"
  case "$ACTION" in
    pdf)
      pandoc "$md" -o "$out_base.pdf" "${PDF_OPTS[@]}" -M title="$title"
      echo "✓ $out_base.pdf" ;;
    docx)
      pandoc "$md" -o "$out_base.docx"
      echo "✓ $out_base.docx" ;;
    all)
      pandoc "$md" -o "$out_base.pdf"  "${PDF_OPTS[@]}" -M title="$title"
      pandoc "$md" -o "$out_base.docx"
      echo "✓ $out_base.pdf / $out_base.docx" ;;
    *)
      echo "用法: $0 [pdf|docx|all] [zh|en]" >&2
      exit 1 ;;
  esac
done
