# Post-check of fields.json: remove honorific residue left inside content points; turn "null" strings into null.
import json, re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(os.path.join(ROOT, 'data', 'fields_raw.json'), encoding='utf-8'))
REP = [('معالي المحافظ', 'المحافظ'), ('فريق معاليكم', 'فريقكم'), ('تزويد سموكم', 'تزويدكم'), ('من سموكم', 'منكم')]
def fix(v):
    if isinstance(v, dict): return {k: fix(x) for k, x in v.items()}
    if isinstance(v, list): return [fix(x) for x in v]
    if isinstance(v, str):
        if v.strip().lower() in ('null', 'none', ''): return None
        for a, b in REP: v = v.replace(a, b)
    return v
n = 0
for k in F:
    new = fix(F[k]['fields']); n += new != F[k]['fields']; F[k]['fields'] = new
json.dump(F, open(os.path.join(ROOT, 'data', 'fields.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
s = json.dumps([v['fields'] for v in F.values()], ensure_ascii=False)
print('changed', n, 'residue', len(re.findall(r'معالي|سعادة|سموكم|سموه|حفظه الله', s)))
