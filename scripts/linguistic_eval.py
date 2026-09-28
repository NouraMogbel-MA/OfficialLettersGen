# Linguistic evaluation of generated letters against the thesis' own pattern inventory (Chapter 2).
import re, json, sys, pandas as pd

import os
E = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'c1_m2m_question') + os.sep


def norm(s):
    s = re.sub(r'[\u064B-\u0652\u0640]', '', str(s))          # diacritics, tatweel
    s = re.sub('[إأآا]', 'ا', s).replace('ى', 'ي')
    return re.sub(r'\s+', ' ', s)


SHARED = ['السلام عليكم ورحمة الله وبركاته', 'اشارة الي', 'حفظه الله', 'معالي', 'نامل', 'امل',
          'وتفضلوا بقبول', 'خالص تحياتي', 'وتقبلوا', 'بناء علي', 'استنادا الي']
DIST = {
    'طلب': ['نامل من معاليكم التكرم', 'التكرم بالموافقة', 'والتوجيه بما يلزم'],
    'دعوة': ['يسر الهيئة دعوتكم', 'دعوتكم للمشاركة', 'دعوتكم لحضور'],
    'إفادة': ['اشارة الي خطاب سعادتكم', 'وفي حال وجود'],
    'ترشيح': ['الهيئة ترشح الاستاذ', 'لتمثيل الهيئة'],
    'شكر': ['تتقدم الهيئة', 'الشكر والتقدير'],
    'إحاطة': ['اود احاطة', 'اود احاطة معاليكم بان الهيئة'],
    'اعتذار': ['نود افادة', 'دعوتكم الكريمة', 'موعد بديل'],
    'تنسيق': ['للمختصين لديكم'],
    'قرار': ['وبناء علي', 'اجتماع مجلس الادارة رقم'],
    'تهنئة': ['التهاني والتبريكات', 'بمناسبة صدور'],
}
Q2T = [('ما هي متطلبات', 'طلب'), ('التعبير عن الشكر', 'شكر'), ('إجراءات تنفيذ التعميم', 'تعميم'), ('تفاصيل التهنئة', 'تهنئة'),
       ('تفاصيل الإفادة', 'إفادة'), ('إجراءات تنسيق', 'تنسيق'), ('ترشيح ممثل', 'ترشيح'), ('أسباب الاعتذار', 'اعتذار'),
       ('تفاصيل الدعوة', 'دعوة'), ('ما هي مستجدات', 'إحاطة')]


def qtype(q):
    for k, t in Q2T:
        if k in q:
            return t
    return 'أخرى'


def has(text, pat):
    t = ' ' + norm(text) + ' '
    return (' ' + pat + ' ') in t or pat in t if len(pat.split()) > 1 else re.search(r'(?<![\u0621-\u064A])' + pat + r'(?![\u0621-\u064A])', t) is not None


def profile(texts, types):
    n = len(texts); r = {}
    for p in SHARED:
        r['مشترك: ' + p] = sum(has(t, p) for t in texts) / n
    hits = tot = 0
    for t, ty in zip(texts, types):
        for p in DIST.get(ty, []):
            tot += 1; hits += has(t, p)
    r['مميز (داخل الفئة الصحيحة)'] = hits / tot if tot else 0
    r['افتتاح السلام عليكم'] = sum(has(t, 'السلام عليكم ورحمة الله وبركاته') for t in texts) / n
    # commas in the letter body only: the student's fixed frame «السيد/السيدة المحترم/ة,» is removed first
    bd = [re.sub(r'^\s*السيد/السيدة المحترم/ة\s*[,،]', '', str(t)) for t in texts]
    r['فاصلة عربية ، (في المتن)'] = sum('،' in t for t in bd) / n
    r['فاصلة لاتينية , (في المتن)'] = sum(',' in t for t in bd) / n
    r['رموز فارغة ????'] = sum('??' in str(t) for t in texts) / n
    r['عبارة القالب «مع الوثائق»/«يتطلب ...»'] = sum(bool(re.search(r'تقديم طلب رسمي مع الوثائق|يتم إرسال دعوة رسمية لحضور|توفر الإفادة معلومات حول|توفر الإحاطة تحديثات حول', str(t))) for t in texts) / n
    r['متوسط الطول بالكلمات'] = sum(len(str(t).split()) for t in texts) / n
    return r


if __name__ == '__main__':
    sets = json.loads(sys.argv[1])            # {"name": "path.csv", ...}
    test = pd.read_csv(E + 'test.csv', encoding='utf-16', sep='\t')
    corp = pd.read_csv(E + 'corpora.csv', encoding='utf-16', sep='\t')
    key = lambda s: re.sub(r'[^\u0621-\u064A0-9]', '', str(s))
    ck = [(key(t), t) for t in corp.Text.astype(str)]
    refs = []
    for gt in test.ground_truth:
        g = key(gt); hits = [t for k, t in ck if k and k in g]; refs.append(max(hits, key=len) if hits else '')
    types = [qtype(q) for q in test.question]
    cols = {'الخطابات الحقيقية (مرجع)': profile(refs, types)}
    for name, path in sets.items():
        d = pd.read_csv(path, encoding='utf-16', sep='\t')
        assert list(d.question) == list(test.question), path
        cols[name] = profile(d.answer.astype(str).tolist(), types)
    out = pd.DataFrame(cols).round(3)
    print(out.to_string())
    out.to_csv(sys.argv[2], encoding='utf-16', sep='\t')
    print(pd.Series(types).value_counts().to_dict())
