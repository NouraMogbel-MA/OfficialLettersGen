# RAGAS for every file in ragas_in/ without a result (3 in parallel). Run from the repository root.
ls ragas_in/*.json | while read f; do b=$(basename "$f" .json); o="results/ragas_$b.csv"; [ -f "$o" ] || echo "$f $o"; done \
  | xargs -P 3 -L 1 sh -c 'python scripts/run_ragas.py "$0" "$1"'
