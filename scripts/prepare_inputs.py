# Build inputs for the five methods from data/fields.json + data/split.json (split 348/75/75).
# Output: data/inputs.json (train/dev/test rows with question, type, fields text, real letter) and prompts.json (methods 3-5).
import re, os, json, random, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data')
full = pd.read_excel(os.path.join(D, 'corpus.xlsx'))
F = json.load(open(os.path.join(D, 'fields.json'), encoding='utf-8'))
SPLIT = json.load(open(os.path.join(D, 'split.json'), encoding='utf-8'))   # rows and questions of the 348/75/75 split

def vs(v):
    """Flatten a field value (str / dict / list) to plain text."""
    if isinstance(v, dict): return '، '.join(vs(x) for x in v.values() if x)
    if isinstance(v, list): return '، '.join(vs(x) for x in v if x)
    return str(v) if v else ''

def ftext(f):
    """Render content fields as plain Arabic lines (null fields skipped)."""
    L = []
    if f.get('نوع_الخطاب'): L.append('نوع الخطاب: ' + str(f['نوع_الخطاب']))
    a = f.get('المخاطب') or {}
    ad = ' - '.join(str(a[k]) for k in ('المنصب', 'الجهة') if a.get(k) and str(a[k]) != 'null')
    if ad: L.append('المخاطَب: ' + ad)
    if vs(f.get('الموضوع')): L.append('الموضوع: ' + vs(f['الموضوع']))
    r = f.get('المرجع') or {}
    if r.get('الرقم') or r.get('التاريخ'):
        L.append('المرجع: ' + ' '.join(x for x in [('رقم ' + str(r['الرقم'])) if r.get('الرقم') else '', ('وتاريخ ' + str(r['التاريخ'])) if r.get('التاريخ') else ''] if x))
    pts = [p for p in (f.get('النقاط_الأساسية') or []) if p]
    if pts: L.append('النقاط الأساسية:\n' + '\n'.join('- ' + str(p) for p in pts))
    for k, lab in (('المطلوب', 'المطلوب'), ('بيانات_التواصل', 'بيانات التواصل'), ('المرسل', 'المرسل')):
        if vs(f.get(k)): L.append(lab + ': ' + vs(f[k]))
    return '\n'.join(L)

clean = lambda t: re.sub(r'\n{3,}', '\n\n', str(t).replace('\r', '')).strip()
data = {}
for sp, rows in SPLIT.items():
    data[sp] = [{'i': k, 'row': s['row'], 'doc_id': s['doc_id'], 'question': s['question'], 'qtype': s['type'],
                 'fields': ftext(F[str(s['row'])]['fields']), 'letter': clean(full.Text.iloc[s['row']])} for k, s in enumerate(rows)]
for sp in data:
    for d in data[sp]:
        d['input'] = 'السؤال: ' + d['question'] + '\n' + d['fields']      # common input for methods 2-5
json.dump(data, open(os.path.join(D, 'inputs.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

TASK = 'اكتب خطابًا إداريًا رسميًا باللغة العربية بناءً على المعلومات الآتية. اكتب نص الخطاب وحده، بلا عنوان ولا شرح.'
SHARED = {  # Chapter 2, tables 4-5: shared patterns by position
    'المخاطبة': 'لقب التوقير المناسب للمخاطَب (مثل: معالي) ثم المنصب، ثم الدعاء: حفظه الله',
    'التحية': 'السلام عليكم ورحمة الله وبركاته',
    'الإحالة': 'إشارة إلى خطاب ... رقم (...) وتاريخ ...، أو: بناءً على ...، أو: استنادًا إلى ...',
    'صيغة الطلب': 'نأمل ...، أو: آمل ...',
    'الختام': 'وتفضلوا بقبول خالص تحياتي، أو: والسلام عليكم ورحمة الله وبركاته',
}
DIST = {  # Chapter 2, tables 7-17: distinctive patterns per type
    'طلب': ['نأمل من معاليكم التكرم', 'التكرم بالموافقة', 'والتوجيه بما يلزم'],
    'دعوة': ['يسر الهيئة دعوتكم', 'دعوتكم للمشاركة', 'دعوتكم لحضور'],
    'إفادة': ['إشارة إلى خطاب سعادتكم', 'وفي حال وجود'],
    'ترشيح': ['الهيئة ترشح الأستاذ', 'لتمثيل الهيئة'],
    'شكر': ['تتقدم الهيئة', 'الشكر والتقدير'],
    'إحاطة': ['أود إحاطة', 'أود إحاطة معاليكم بأن الهيئة'],
    'اعتذار': ['نود إفادة', 'دعوتكم الكريمة', 'موعد بديل'],
    'تنسيق': ['للمختصين لديكم'],
    'قرار': ['وبناء على', 'اجتماع مجلس الإدارة رقم'],
    'تهنئة': ['التهاني والتبريكات', 'بمناسبة صدور'],
    'تعميم': ['الجهات الحكومية', 'قرار مجلس الوزراء'],
}
NAME = {'طلب': 'الطلب', 'دعوة': 'الدعوة', 'إفادة': 'الإفادة', 'ترشيح': 'الترشيح', 'شكر': 'الشكر', 'إحاطة': 'الإحاطة',
        'اعتذار': 'الاعتذار', 'تنسيق': 'التنسيق', 'قرار': 'القرار', 'تهنئة': 'التهنئة', 'تعميم': 'التعميم'}

def ling_block(ty):
    s = 'اتبع بنية الخطاب الإداري السعودي كما تظهر في مدونة من 498 خطابًا حقيقيًا:\n'
    s += '\n'.join(f'{k + 1}. {n}: {v}' for k, (n, v) in enumerate(SHARED.items()))
    if ty in DIST:
        s += f'\nومن الصيغ المميزة لخطابات {NAME[ty]}: ' + '، '.join('«' + p + '»' for p in DIST[ty]) + '.'
    s += '\nاستعمل هذه الصيغ في مواضعها بحسب المقام، ولا تضف معلومات أو أسماء أو أرقامًا أو بيانات تواصل غير موجودة في المعلومات.'
    return s

tr = data['train']; test_letters = {re.sub(r'\W', '', d['letter']) for d in data['test']}
pool = {}
for d in tr:
    if re.sub(r'\W', '', d['letter']) in test_letters: continue       # never show a test letter as an example
    pool.setdefault(d['qtype'], []).append(d)
prompts = []
for d in data['test']:
    rng = random.Random(1000 + d['i'])
    ex = rng.sample(pool[d['qtype']], 2)
    zero = TASK + '\n\n' + d['input']
    few = (TASK + ' وهذان مثالان من خطابات حقيقية من النوع نفسه:\n\n' +
           '\n\n'.join(f'المثال {n}:\nالمعلومات:\n{e["input"]}\nالخطاب:\n{e["letter"]}' for n, e in ((1, ex[0]), (2, ex[1]))) +
           '\n\nالمطلوب الآن:\nالمعلومات:\n' + d['input'] + '\nالخطاب:')
    ling = TASK + '\n\n' + ling_block(d['qtype']) + '\n\nالمعلومات:\n' + d['input']
    prompts.append({'i': d['i'], 'question': d['question'], 'qtype': d['qtype'], 'examples': [e['row'] for e in ex],
                    'zero': zero, 'few': few, 'ling': ling})
json.dump(prompts, open(os.path.join(D, 'prompts.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print({k: len(v) for k, v in data.items()}, len(prompts))
print(prompts[3]['ling']); print('-----'); print(prompts[3]['few'][:1500])
