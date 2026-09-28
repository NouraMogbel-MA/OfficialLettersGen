# RAGAS 0.2.15 evaluation of the fine-tuned model outputs (defaults: gpt-4o-mini judge, text-embedding-ada-002)
import os, sys, json, pandas as pd
# the OpenAI key is read from the OPENAI_API_KEY environment variable
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall, answer_correctness

inp, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else None
rows = json.load(open(inp, encoding='utf-8'))
if n: rows = rows[:n]
ds = EvaluationDataset.from_list(rows)
res = evaluate(ds, metrics=[context_precision, faithfulness, answer_relevancy, context_recall, answer_correctness],
               run_config=RunConfig(max_workers=6, timeout=240, max_retries=6), raise_exceptions=False, show_progress=False)
df = res.to_pandas()
df.to_csv(out, index=False, encoding='utf-8-sig')
m = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall', 'answer_correctness']
print(df[m].describe().round(4).to_string())
print('NaN per metric:', df[m].isna().sum().to_dict())
