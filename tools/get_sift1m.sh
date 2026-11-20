#!/usr/bin/env bash
# Downloads and unpacks SIFT1M (168 MB) from the TEXMEX corpus.
#   tools/get_sift1m.sh [destination, default ~/hvs-data]
set -euo pipefail

dest=${1:-"$HOME/hvs-data"}
url=ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

mkdir -p "$dest"
if [[ -f "$dest/sift/sift_base.fvecs" ]]; then
  echo "already present: $dest/sift"
  exit 0
fi
curl --fail --location --output "$dest/sift.tar.gz" "$url"
tar -xzf "$dest/sift.tar.gz" -C "$dest"
rm "$dest/sift.tar.gz"
ls -l "$dest/sift"
