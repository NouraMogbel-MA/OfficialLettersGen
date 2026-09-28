# Extra linguistic indicators (content and pragmatics) for generated letters vs. the real letters.
import re, sys, json, pandas as pd
sys.path.insert(0, '.')
from linguistic_eval import norm, qtype, E

def addressee(t):
    t = re.sub(r'^\s*السيد/السيدة المحترم/ة\s*[,،]', '', str(t))
    m = re.search(r':\s*(.{3,120}?)\s*(حفظه|حفظها|حفظهم|سلمه|سلّمه|سلمها|سلّمها|سلمهما|سلّمهما|السلام عليكم)', t)
    return norm(m.group(1)).strip() if m else None

def prof(texts, gts, types):
    n = len(texts); r = {}
    r['ختام صحيح «والسلام عليكم ورحمة الله وبركاته»'] = sum('والسلام عليكم ورحمة الله وبركاته' in norm(t) for t in texts) / n
    r['ختام محرّف «والسلام عليهم...»'] = sum('والسلام عليهم' in norm(t) for t in texts) / n
    r['رقم هاتف أو بريد إلكتروني'] = sum(bool(re.search(r'05\d{6,}|@', str(t))) for t in texts) / n
    ph_new = 0
    for t, g in zip(texts, gts):
        ph = set(re.findall(r'05\d{6,}|[\w.]+@[\w.]+', str(t)))
        if ph and not ph <= set(re.findall(r'05\d{6,}|[\w.]+@[\w.]+', str(g))): ph_new += 1
    r['بيانات تواصل غير موجودة في الخطاب الأصلي'] = ph_new / n
    inv = [(t) for t, ty in zip(texts, types) if ty == 'دعوة']
    r['الدعوة تبدأ بـ«إشارة إلى خطاب/خطابكم» (من 19)'] = sum(bool(re.search('اشارة الي خطاب (معاليكم|سعادتكم|سموكم)', norm(t))) for t in inv) / len(inv)
    ok = [addressee(t) == addressee(g) for t, g in zip(texts, gts) if addressee(g)]
    r['المخاطَب مطابق للخطاب الأصلي'] = sum(ok) / len(ok)
    r['عدد المخاطَبين المختلفين'] = len(set(a for a in map(addressee, texts) if a))
    r['عبارة دخيلة «والله أعلم»'] = sum('والله اعلم' in norm(t) for t in texts) / n
    return r

if __name__ == '__main__':
    sets = json.loads(sys.argv[1])
    test = pd.read_csv(E + 'test.csv', encoding='utf-8-sig')
    types = [qtype(q) for q in test.question]; gts = test.ground_truth.astype(str).tolist()
    cols = {'الخطابات الحقيقية (مرجع)': prof(gts, gts, types)}
    for name, path in sets.items():
        d = pd.read_csv(path, encoding='utf-8-sig'); assert list(d.question) == list(test.question)
        cols[name] = prof(d.answer.astype(str).tolist(), gts, types)
    out = pd.DataFrame(cols).round(3); print(out.to_string()); out.to_csv(sys.argv[2], encoding='utf-8-sig')
