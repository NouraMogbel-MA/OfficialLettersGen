# Aggregate everything for chapter 4 into results/final.json:
# linguistic + information indicators (evaluate), key-point coverage (kp_judge), RAGAS (run_ragas), copying from few-shot examples.
import json, glob, os, re, numpy as np, pandas as pd
import evaluate as E
H = E.H
LI = json.load(open(f'{H}/results/ling_info.json', encoding='utf-8'))
M = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall', 'answer_correctness']
out = {'ling': {}, 'ragas': {}, 'kp': {}, 'n_runs': {}, 'copied_examples': None}
for c in list(E.COND) + ['real', 'c2_m2m_fields_nongram']:
    if c in LI:
        out['ling'][c] = {'mean': LI[c], 'range': LI.get(c + '_range'), 'n': LI.get(c + '_n')}
for c in list(E.COND) + ['real']:
    kp = []
    for p in sorted(glob.glob(f'{H}/results/kp_{c}_run*.json')):
        r = json.load(open(p)); v = [sum(s) / len(s) for s in r.values() if s]
        kp.append(float(np.mean(v)))
    if kp: out['kp'][c] = {'mean': float(np.mean(kp)), 'min': min(kp), 'max': max(kp), 'n': len(kp)}
    rg = [pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{H}/results/ragas_{c}_run[0-9].csv'))]
    if rg:
        out['ragas'][c] = {m: {'mean': float(np.mean([d[m].mean() for d in rg])), 'min': float(min(d[m].mean() for d in rg)),
                               'max': float(max(d[m].mean() for d in rg)), 'nan': [int(d[m].isna().sum()) for d in rg]} for m in M}
        out['ragas'][c]['ar0'] = float(np.mean([(d.answer_relevancy == 0).mean() for d in rg]))
        out['ragas'][c]['n'] = len(rg)
# few-shot: share of letters that copy a number/email found only in the two examples
P = json.load(open(f'{H}/data/prompts.json', encoding='utf-8'))
tr = {d['row']: d for d in json.load(open(f'{H}/data/inputs.json', encoding='utf-8'))['train']}
cp = []
for p in E.COND['c4_allam_few']:
    tx = [t.translate(E.AD) for t in E.load(p)]; k = 0
    for i, t in enumerate(tx):
        known = E.nums(E.data[i]['fields']) | E.nums(E.real[i]) | E.nums(E.data[i]['question'])
        knownE = E.emails(E.data[i]['fields']) | E.emails(E.real[i])
        fab = (E.nums(t) - known - {'2030'}) | (E.emails(t) - knownE)
        ex = ' '.join(tr[r]['letter'] + ' ' + tr[r]['fields'] for r in P[i]['examples']).translate(E.AD)
        k += any(x in ex for x in fab)
    cp.append(k / len(tx))
if cp: out['copied_examples'] = {'mean': float(np.mean(cp)), 'min': min(cp), 'max': max(cp)}
json.dump(out, open(f'{H}/results/final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({'kp': out['kp'], 'copied': out['copied_examples']}, ensure_ascii=False))
print(pd.DataFrame({c: {m: v[m]['mean'] for m in M} for c, v in out['ragas'].items()}).round(3).to_string())
