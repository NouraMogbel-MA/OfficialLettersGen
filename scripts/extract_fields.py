# Step 1: extract content fields (no formulas) from every letter in corpus.xlsx with gpt-4o-mini.
# Resumable: saves fields.json after each letter. Auto-check: digits/emails/names in fields must occur in the letter.
import os, re, json, sys, pandas as pd
from concurrent.futures import ThreadPoolExecutor
# the OpenAI key is read from the OPENAI_API_KEY environment variable
from openai import OpenAI
cli = OpenAI()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full = pd.read_excel(os.path.join(ROOT, 'data', 'corpus.xlsx'))
OUT = os.path.join(ROOT, 'data', 'fields_raw.json')
res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}

SYS = """أنت محلل لغوي للخطابات الإدارية العربية. استخرج من الخطاب معلومات المضمون فقط في JSON بالمفاتيح الآتية:
{"نوع_الخطاب": "كلمة أو كلمتان (طلب، دعوة، شكر، إفادة، ترشيح، إحاطة، اعتذار، تنسيق، تعميم، تهنئة، قرار، ...)",
 "المخاطب": {"الجهة": "...", "المنصب": "..."},
 "الموضوع": "عبارة قصيرة",
 "المرجع": {"الرقم": "...", "التاريخ": "..."},
 "النقاط_الأساسية": ["1 إلى 4 نقاط مختصرة بصياغتك"],
 "المطلوب": "ما يطلبه المرسل من المخاطب، أو null",
 "بيانات_التواصل": "اسم المنسق وهاتفه وبريده كما وردت، أو null",
 "المرسل": "الجهة أو المنصب المرسل إن ذُكر، أو null"}
قواعد صارمة:
- لا تضع أي صيغة لغوية جاهزة: لا ألقاب التوقير (معالي، سعادة، سمو، فضيلة)، ولا الدعاء (حفظه الله، سلمه الله)، ولا التحية ولا الختام، ولا صيغ الطلب (نأمل، التكرم، يسرنا، نود). المنصب يكتب مجردًا مثل: وزير التعليم.
- انقل الأسماء والأرقام والتواريخ والهواتف والبريد حرفيًا كما في الخطاب، ولا تخترع شيئًا غير موجود. ما لا يوجد يكون null.
- المرجع: رقم الخطاب السابق وتاريخه الذي يشير إليه الخطاب (بعد «إشارة إلى»)، إن وجد.
- النقاط الأساسية تحمل المعلومات (الحدث، المكان، الزمان، الأسماء) لا الأسلوب."""

norm = lambda s: re.sub('[إأآا]', 'ا', re.sub(r'[\u064B-\u0652\u0640]', '', str(s))).replace('ى', 'ي').replace('ة', 'ه')

AD = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')

def clean(f):
    if isinstance(f, dict): return {k.strip(): clean(v) for k, v in f.items()}
    if isinstance(f, list): return [clean(v) for v in f]
    return f.translate(AD) if isinstance(f, str) else f

def check(f, text):
    """Return list of problems: digit strings / emails not in text; names in addressee/sender/contact not in text;
    formula words present."""
    text = text.translate(AD); probs = []; T = norm(text); Td = re.sub(r'\D', ' ', text)
    s = json.dumps(f, ensure_ascii=False)
    for num in re.findall(r'\d{2,}', s):
        if num not in re.findall(r'\d+', text) and num not in text: probs.append('num:' + num)
    for em in re.findall(r'[\w.\-]+@[\w.\-]+', s):
        if em.lower() not in text.lower(): probs.append('email:' + em)
    for fld in (f.get('بيانات_التواصل'),):
        if fld:
            for w in re.findall(r'[\u0621-\u064A]{3,}', str(fld)):
                if norm(w) not in T and norm(w).lstrip('ال') not in T and w not in ('الهاتف', 'البريد', 'الإلكتروني', 'جوال', 'هاتف', 'بريد', 'إلكتروني', 'المتنقل', 'الجوال', 'رقم', 'الأستاذ', 'الأستاذة', 'التحويلة', 'تحويلة'):
                    probs.append('name:' + w)
    for bad in ('معالي', 'سعاده', 'سمو', 'فضيله', 'حفظه الله', 'سلمه الله', 'التكرم', 'نامل', 'يسر'):
        if bad in norm(s): probs.append('formula:' + bad)
    return probs

def one(i):
    text = str(full.Text.iloc[i]); msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': text}]
    for attempt in range(3):
        r = cli.chat.completions.create(model='gpt-4o-mini', temperature=0, response_format={'type': 'json_object'}, messages=msgs)
        out = r.choices[0].message.content
        try: f = clean(json.loads(out))
        except Exception: continue
        p = check(f, text)
        if not p: break
        msgs += [{'role': 'assistant', 'content': out},
                 {'role': 'user', 'content': 'صحح هذه المشكلات وأعد JSON كاملًا: ' + '، '.join(p) + '. الأرقام والأسماء حرفيًا من الخطاب فقط، وبلا صيغ توقير أو دعاء أو طلب.'}]
    return i, {'doc_id': full.doc_id.iloc[i], 'fields': f, 'problems': p, 'attempts': attempt + 1}

todo = [i for i in range(len(full)) if str(i) not in res]
with ThreadPoolExecutor(8) as ex:
    for k, (i, r) in enumerate(ex.map(one, todo)):
        res[str(i)] = r
        if k % 10 == 0 or k == len(todo) - 1:
            json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
bad = {k: v['problems'] for k, v in res.items() if v['problems']}
print('done', len(res), 'with problems', len(bad)); print(json.dumps(dict(list(bad.items())[:30]), ensure_ascii=False))
