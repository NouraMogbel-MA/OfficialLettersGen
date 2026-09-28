# Unified evaluation of the five methods on the 75 test letters (348/75/75 split).
# Reference for every method = the real (unmasked) letter from corpus.xlsx.
#   1) linguistic indicators (linguistic_eval.profile + extra_eval.prof), 2) information coverage and fabrication,
#   3) RAGAS inputs (ragas_in/<cond>_run<k>.json) for run_ragas.py.
import re, os, sys, json, glob, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linguistic_eval as L, extra_eval as X
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
data = json.load(open(f'{H}/data/inputs.json', encoding='utf-8'))['test']
F = json.load(open(f'{H}/data/fields.json', encoding='utf-8'))
test = pd.read_csv(L.E + 'test.csv', encoding='utf-16', sep='\t')
assert [d['question'] for d in data] == list(test.question)
types = [d['qtype'] for d in data]; real = [d['letter'] for d in data]
COND = {  # method -> list of generation files (one per run)
    'c1_m2m_question': sorted(glob.glob(f'{H}/outputs/c1_m2m_question/gen_run[123].csv')),
    'c2_m2m_fields': sorted(glob.glob(f'{H}/outputs/c2_m2m_fields/gen_run[123].csv')),
    'c3_allam_zero': sorted(glob.glob(f'{H}/outputs/c3_allam_zero/gen_run[123].csv')),
    'c4_allam_few': sorted(glob.glob(f'{H}/outputs/c4_allam_few/gen_run[123].csv')),
    'c5_allam_ling': sorted(glob.glob(f'{H}/outputs/c5_allam_ling/gen_run[123].csv')),
}
EXTRA = {}
AD = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
FRAME = re.compile(r'^\s*السيد/السيدة المحترم/ة\s*[,،]\s*')


def addressee(t):
    """Addressee line = text before the first blessing/greeting (after removing the student frame and its template clause)."""
    t = FRAME.sub('', str(t)); t = re.sub(r'^[^:\n]{0,160}الجهات:\s*', '', t)
    m = re.search(r'^\s*(.{3,160}?)\s*[-–،,]?\s*(حفظه|حفظها|حفظهم|سلمه|سلّمه|سلمها|سلّمها|السلام عليكم)', t, re.S)
    return re.sub(r'\s+', ' ', L.norm(m.group(1))).strip(' -–،,') if m else None


X.addressee = addressee
TITLES = [('صاحب السمو', 'سمو'), ('سمو', 'سمو'), ('معالي', 'معالي'), ('سعاده', 'سعادة'), ('فضيله', 'فضيلة')]


def title(t):
    """Honorific title of the addressee line (first 150 characters after the student frame)."""
    s = L.norm(re.sub(r'^[^:\n]{0,160}الجهات:\s*', '', FRAME.sub('', str(t))))[:150].replace('ة', 'ه')
    hits = [(s.find(k), v) for k, v in TITLES if k in s]
    return min(hits)[1] if hits else None


def load(p):
    d = pd.read_csv(p, encoding='utf-16', sep='\t')
    assert list(d.question) == list(test.question), p
    return [FRAME.sub('', str(a)) for a in d.answer.fillna('')]


def nums(s):
    return set(re.findall(r'\d{3,}', str(s).translate(AD)))


def emails(s):
    return set(e.lower().strip('.') for e in re.findall(r'[\w.\-]+@[\w\-]+(?:\.[\w\-]+)+', str(s)))


def atoms(i):
    """Exact items the letter must carry: numbers (>=3 digits) and emails in the input fields, plus the addressee's position."""
    f = F[str(data[i]['row'])]['fields']; txt = data[i]['fields']
    a = {('n', x) for x in nums(txt)} | {('e', x) for x in emails(txt)}
    pos = (f.get('المخاطب') or {}).get('المنصب')
    if pos: a.add(('p', L.norm(pos)))
    return a


def has_atom(t, a):
    k, v = a
    if k == 'n': return v in nums(t)
    if k == 'e': return v in emails(t)
    return v in L.norm(t)


def info(texts):
    cov, fab, part = [], [], {'n': [], 'e': [], 'p': []}
    for i, t in enumerate(texts):
        A = atoms(i)
        if A: cov.append(np.mean([has_atom(t, a) for a in A]))
        for a in A: part[a[0]].append(has_atom(t, a))
        known = nums(data[i]['fields']) | nums(real[i]) | nums(data[i]['question'])
        knownE = emails(data[i]['fields']) | emails(real[i])
        fab.append(bool((nums(t) - known - {'2030'}) or (emails(t) - knownE)))   # 2030 = «رؤية المملكة 2030», not letter data
    return {'تغطية المعلومات الدقيقة (أرقام وبريد ومنصب)': float(np.mean(cov)),
            'تغطية: الأرقام': float(np.mean(part['n'])), 'تغطية: البريد': float(np.mean(part['e'])),
            'تغطية: منصب المخاطَب': float(np.mean(part['p'])), 'أرقام أو بريد مختلق': float(np.mean(fab))}


def profile(texts):
    texts = [t.translate(AD) for t in texts]; rl = [t.translate(AD) for t in real]
    r = L.profile(texts, types); r.update(X.prof(texts, rl, types)); r.update(info(texts))
    for k in ('المخاطَب مطابق للخطاب الأصلي', 'عدد المخاطَبين المختلفين'): r.pop(k, None)
    cd = lambda x: set(re.findall(r'05\d{8}', x)) | emails(x)        # extra_eval kept a trailing period on emails
    r['بيانات تواصل غير موجودة في الخطاب الأصلي'] = float(np.mean([bool(cd(t) - cd(g)) for t, g in zip(texts, rl)]))
    r['نص غير عربي أو شرح خارج الخطاب'] = float(np.mean([bool(re.search(r'(ملاحظة|يرجى ملاحظة|\*\*|#|Note)', t)) for t in texts]))
    dummy = lambda x: set(x) <= {'0'} or '00000' in x or x in '1234567890123456789'
    fake = [set() for _ in texts]
    for i, t in enumerate(texts):
        known = nums(data[i]['fields']) | nums(real[i]) | nums(data[i]['question'])
        fake[i] = nums(t) - known - {'2030'}
    r['أرقام نائبة (أصفار أو تسلسل)'] = float(np.mean([any(dummy(x) for x in f) for f in fake]))
    r['أرقام مختلقة حقيقية الشكل أو بريد مختلق'] = float(np.mean([
        any(not dummy(x) for x in f) or bool(emails(t) - emails(data[i]['fields']) - emails(real[i]))
        for i, (f, t) in enumerate(zip(fake, texts))]))
    noref = [i for i in range(len(texts)) if 'المرجع:' not in data[i]['fields']]
    ish = lambda t: bool(re.search('اشاره الي (خطاب|كتاب|برقيه|الامر|قرار|تعميم)', L.norm(t).replace('ة', 'ه')))
    r['إحالة إلى خطاب سابق مع أن المدخل بلا مرجع'] = float(np.mean([ish(texts[i]) for i in noref]))
    r['_n_noref'] = len(noref)
    head = lambda t: L.norm(re.sub(r'^[^:\n]{0,160}الجهات:\s*', '', FRAME.sub('', t))).lstrip(' *#-\n')
    r['يبدأ الخطاب بسطر المخاطَب'] = float(np.mean([not re.match(r'(السلام عليكم|تحيه|تحية|بسم الله|الموضوع|اشاره)', head(t)) for t in texts]))
    tt = [(title(t), title(g)) for t, g in zip(texts, rl) if title(g)]
    r['لقب التوقير مطابق للخطاب الحقيقي'] = float(np.mean([a == b for a, b in tt]))
    r['صيغة «تحية طيبة»'] = float(np.mean([bool(re.search('تحيه طيبه', L.norm(t).replace('ة', 'ه'))) for t in texts]))
    r['ختام «فائق الاحترام والتقدير»'] = float(np.mean([bool(re.search('فائق الاحترام', L.norm(t))) for t in texts]))
    r['عناصر فارغة بين أقواس [..]'] = float(np.mean([bool(re.search(r'\[[^\]]{2,30}\]', t)) for t in texts]))
    return r


def ragas_in(texts, out):
    rows = []
    for d, t in zip(data, texts):
        paras = [p.strip() for p in re.split(r'\n\s*\n', d['letter']) if p.strip()]
        rows.append({'user_input': d['question'], 'response': t, 'reference': d['letter'], 'retrieved_contexts': paras})
    json.dump(rows, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    os.makedirs(f'{H}/ragas_in', exist_ok=True); os.makedirs(f'{H}/results', exist_ok=True)
    res = {'real': profile(real)}
    for c, files in {**COND, **EXTRA}.items():
        runs = []
        for k, p in enumerate(files, 1):
            if not os.path.exists(p): continue
            tx = load(p); runs.append(profile(tx))
            if c in COND: ragas_in(tx, f'{H}/ragas_in/{c}_run{k}.json')
        if runs:
            res[c] = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}
            res[c + '_range'] = {m: [min(r[m] for r in runs), max(r[m] for r in runs)] for m in runs[0]}
            res[c + '_n'] = len(runs)
    json.dump(res, open(f'{H}/results/ling_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    df = pd.DataFrame({k: v for k, v in res.items() if not k.endswith(('_range', '_n'))}).round(3)
    df.to_csv(f'{H}/results/ling_info.csv', encoding='utf-16', sep='\t', index_label='indicator'); print(df.to_string())
