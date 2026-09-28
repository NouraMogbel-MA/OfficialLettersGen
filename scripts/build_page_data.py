# -*- coding: utf-8 -*-
# Build docs/data.js for the local viewer (docs/index.html) from the files of this repository.
# usage (from the repository root): python scripts/build_page_data.py
import os, re, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *a: os.path.join(ROOT, *a)
METHODS = [
    ('c1_m2m_question', 'التدريب على السؤال', 'نموذج فيسبوك m2m100'),
    ('c2_m2m_fields', 'التدريب على المعلومات', 'نموذج فيسبوك m2m100'),
    ('c3_allam_zero', 'الأمر دون أمثلة', 'نموذج علّام 7B Instruct Preview'),
    ('c4_allam_few', 'الأمر بمثالين', 'نموذج علّام 7B Instruct Preview'),
    ('c5_allam_ling', 'الأمر اللغوي', 'نموذج علّام 7B Instruct Preview'),
]
PROMPT_C1 = "اكتب رسالة رسمية باللغة العربية بعنوان 'الموضوع: {q}' بدءًا بـ 'السيد/السيدة المحترم/ة' وانتهاء بـ 'مع التحية، مؤسسة مسك'"

test = json.load(open(J('data', 'inputs.json'), encoding='utf-8'))['test']
prompts = json.load(open(J('data', 'prompts.json'), encoding='utf-8'))
final = json.load(open(J('results', 'final.json'), encoding='utf-8'))


def subject(d):
    m = re.search(r'^الموضوع: (.+)$', d['input'], re.M)
    return (m.group(1) if m else d['question']).strip()


letters = []
for k, d in enumerate(test):
    letters.append({'type': d['qtype'], 'subject': subject(d), 'question': d['question'],
                    'info': d['input'], 'original': d['letter'], 'methods': {}})
for key, name, model in METHODS:
    for run in (1, 2, 3):
        out = pd.read_csv(J('outputs', key, f'gen_run{run}.csv'), encoding='utf-8-sig')
        rg = pd.read_csv(J('results', f'ragas_{key}_run{run}.csv'), encoding='utf-8-sig')
        assert len(out) == len(rg) == len(test)
        for k in range(len(test)):
            f = rg.faithfulness.iloc[k]
            letters[k]['methods'].setdefault(key, {'runs': []})['runs'].append(
                {'text': str(out.answer.iloc[k]), 'faithfulness': None if pd.isna(f) else round(float(f), 2)})
for k, d in enumerate(test):
    p = prompts[k]
    letters[k]['methods']['c1_m2m_question']['prompt'] = PROMPT_C1.format(q=d['question'])
    letters[k]['methods']['c2_m2m_fields']['prompt'] = d['input']
    letters[k]['methods']['c3_allam_zero']['prompt'] = p['zero']
    letters[k]['methods']['c4_allam_few']['prompt'] = p['few']
    letters[k]['methods']['c5_allam_ling']['prompt'] = p['ling']

L = final['ling']
summary = []
for key, name, model in METHODS:
    summary.append({'key': key, 'name': name, 'model': model,
                    'faithfulness': round(final['ragas'][key]['faithfulness']['mean'], 2),
                    'correctness': round(final['ragas'][key]['answer_correctness']['mean'], 2),
                    'distinctive': round(100 * L[key]['mean']['مميز (داخل الفئة الصحيحة)']),
                    'keypoints': round(100 * final['kp'][key]['mean']),
                    'honorific': round(100 * L[key]['mean']['لقب التوقير مطابق للخطاب الحقيقي'])})
real = {'distinctive': round(100 * L['real']['mean']['مميز (داخل الفئة الصحيحة)'])}
os.makedirs(J('docs'), exist_ok=True)
data = {'methods': [{'key': k, 'name': n, 'model': m} for k, n, m in METHODS], 'summary': summary, 'real': real, 'letters': letters}
# a script file, not JSON, so that index.html opens straight from the disk (a browser does not fetch files from file://)
with open(J('docs', 'data.js'), 'w', encoding='utf-8') as f:
    f.write('window.DATA = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n')
print('letters', len(letters), 'size KB', os.path.getsize(J('docs', 'data.js')) // 1024)
