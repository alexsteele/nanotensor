#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="${ROOT_DIR}/data/shakespeare"
RAW_DIR="${CORPUS_DIR}/raw"
CORPUS_PATH="${CORPUS_DIR}/shakespeare_gutenberg.txt"

# Core plays (Gutenberg ebook IDs)
PLAY_IDS=(
  1513  # Romeo and Juliet
  1514  # A Midsummer Night's Dream
  1515  # The Merchant of Venice
  1519  # Hamlet (alt)
  1522  # Julius Caesar
  1524  # Hamlet
  1531  # Othello
  1532  # King Lear
  1533  # Macbeth
  2265  # Hamlet (popular mirror id)
  23042 # The Tempest
)

STEPS="${1:-3000}"
PROMPT="${2:-To be}"
BATCH="${3:-64}"
GEN_LEN="${4:-300}"
LR="${5:-0.2}"
TEMP="${6:-0.9}"
HIDDEN="${7:-96}"

mkdir -p "${RAW_DIR}"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_one() {
  local id="$1"
  local out="$2"
  local urls=(
    "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt.utf-8"
    "https://www.gutenberg.org/files/${id}/${id}-0.txt"
    "https://www.gutenberg.org/files/${id}/${id}.txt"
  )
  local u

  if have_cmd curl; then
    for u in "${urls[@]}"; do
      if curl -fsSL "$u" -o "$out"; then
        return 0
      fi
    done
  elif have_cmd wget; then
    for u in "${urls[@]}"; do
      if wget -qO "$out" "$u"; then
        return 0
      fi
    done
  else
    echo "Need curl or wget installed to download Gutenberg texts." >&2
    return 2
  fi

  return 1
}

strip_gutenberg() {
  local in="$1"
  local out="$2"
  awk '
    BEGIN { keep=0 }
    /\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK/ { keep=1; next }
    /\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK/ { keep=0; exit }
    keep { print }
  ' "$in" > "$out"

  # Fallback: if markers were not found, keep original text.
  if [[ ! -s "$out" ]]; then
    cp "$in" "$out"
  fi
}

echo "Downloading Shakespeare plays from Gutenberg..."
for id in "${PLAY_IDS[@]}"; do
  raw_file="${RAW_DIR}/pg${id}.txt"
  clean_file="${RAW_DIR}/pg${id}.clean.txt"
  if download_one "$id" "$raw_file"; then
    strip_gutenberg "$raw_file" "$clean_file"
    echo "  downloaded and cleaned id ${id}"
  else
    echo "  warning: failed to download id ${id}" >&2
  fi
done

cat /dev/null > "${CORPUS_PATH}"
for clean_file in "${RAW_DIR}"/*.clean.txt; do
  [[ -f "$clean_file" ]] || continue
  cat "$clean_file" >> "${CORPUS_PATH}"
  printf "\n\n" >> "${CORPUS_PATH}"
done

if [[ ! -s "${CORPUS_PATH}" ]]; then
  echo "No corpus data available. Check network access and Gutenberg availability." >&2
  exit 1
fi

echo "Prepared corpus at: ${CORPUS_PATH}"

echo "Building llm demo..."
make -C "${ROOT_DIR}" llm

echo "Training + generating..."
"${ROOT_DIR}/llm" "${CORPUS_PATH}" "$STEPS" "$PROMPT" "$BATCH" "$GEN_LEN" "$LR" "$TEMP" "$HIDDEN"
