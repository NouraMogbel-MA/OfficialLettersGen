# -*- coding: utf-8 -*-
"""Build the inputs and prompts exactly as in the experiment (scripts/prepare_inputs.py)."""
import re, random

TASK = 'اكتب خطابًا إداريًا رسميًا باللغة العربية بناءً على المعلومات الآتية. اكتب نص الخطاب وحده، بلا عنوان ولا شرح.'
SHARED = {  # chapter 2 of the thesis: shared patterns by position in the letter
    'المخاطبة': 'لقب التوقير المناسب للمخاطَب (مثل: معالي) ثم المنصب، ثم الدعاء: حفظه الله',
    'التحية': 'السلام عليكم ورحمة الله وبركاته',
    'الإحالة': 'إشارة إلى خطاب ... رقم (...) وتاريخ ...، أو: بناءً على ...، أو: استنادًا إلى ...',
    'صيغة الطلب': 'نأمل ...، أو: آمل ...',
    'الختام': 'وتفضلوا بقبول خالص تحياتي، أو: والسلام عليكم ورحمة الله وبركاته',
}
DIST = {  # chapter 2 of the thesis: distinctive patterns of each letter type
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
# question templates of the corpus (one per letter type)
QUESTION = {'طلب': 'ما هي متطلبات {}؟', 'شكر': 'كيف يتم التعبير عن الشكر بشأن {}؟', 'تعميم': 'ما هي إجراءات تنفيذ التعميم بشأن {}؟',
            'تهنئة': 'ما هي تفاصيل التهنئة بشأن {}؟', 'إفادة': 'ما هي تفاصيل الإفادة بشأن {}؟', 'تنسيق': 'ما هي إجراءات تنسيق {}؟',
            'ترشيح': 'ما هي تفاصيل ترشيح ممثل بشأن {}؟', 'اعتذار': 'ما هي أسباب الاعتذار عن {}؟',
            'دعوة': 'ما هي تفاصيل الدعوة لحضور {}؟', 'إحاطة': 'ما هي مستجدات {}؟', 'قرار': 'ما هي تفاصيل {}؟'}
TYPES = list(QUESTION)


def info_text(f):
    """Letter information as plain Arabic lines (empty items skipped); f is a dict of the eight items."""
    L = []
    if f.get('type'): L.append('نوع الخطاب: ' + f['type'])
    ad = ' - '.join(x for x in (f.get('position'), f.get('entity')) if x)
    if ad: L.append('المخاطَب: ' + ad)
    if f.get('subject'): L.append('الموضوع: ' + f['subject'])
    ref = ' '.join(x for x in (('رقم ' + f['ref_no']) if f.get('ref_no') else '', ('وتاريخ ' + f['ref_date']) if f.get('ref_date') else '') if x)
    if ref: L.append('المرجع: ' + ref)
    pts = [p.strip() for p in (f.get('points') or []) if p.strip()]
    if pts: L.append('النقاط الأساسية:\n' + '\n'.join('- ' + p for p in pts))
    for k, lab in (('request', 'المطلوب'), ('contact', 'بيانات التواصل'), ('sender', 'المرسل')):
        if f.get(k): L.append(lab + ': ' + f[k])
    return '\n'.join(L)


def question(ty, subject):
    return QUESTION.get(ty, 'ما هي تفاصيل {}؟').format(subject)


def model_input(q, info):
    return 'السؤال: ' + q + '\n' + info


def ling_block(ty):
    s = 'اتبع بنية الخطاب الإداري السعودي كما تظهر في مدونة من 498 خطابًا حقيقيًا:\n'
    s += '\n'.join(f'{k + 1}. {n}: {v}' for k, (n, v) in enumerate(SHARED.items()))
    if ty in DIST:
        s += f'\nومن الصيغ المميزة لخطابات {NAME[ty]}: ' + '، '.join('«' + p + '»' for p in DIST[ty]) + '.'
    s += '\nاستعمل هذه الصيغ في مواضعها بحسب المقام، ولا تضف معلومات أو أسماء أو أرقامًا أو بيانات تواصل غير موجودة في المعلومات.'
    return s


def prompt(method, ty, inp, train_rows=None, seed=0):
    """method: 'zero' | 'few' | 'ling'. train_rows: training letters (dicts with qtype, input, letter) for the examples."""
    if method == 'zero':
        return TASK + '\n\n' + inp
    if method == 'ling':
        return TASK + '\n\n' + ling_block(ty) + '\n\nالمعلومات:\n' + inp
    pool = [d for d in (train_rows or []) if d['qtype'] == ty] or list(train_rows or [])
    ex = random.Random(seed).sample(pool, 2)
    return (TASK + ' وهذان مثالان من خطابات حقيقية من النوع نفسه:\n\n' +
            '\n\n'.join(f'المثال {n}:\nالمعلومات:\n{e["input"]}\nالخطاب:\n{e["letter"]}' for n, e in ((1, ex[0]), (2, ex[1]))) +
            '\n\nالمطلوب الآن:\nالمعلومات:\n' + inp + '\nالخطاب:')
