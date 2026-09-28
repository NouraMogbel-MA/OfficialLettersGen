# Key-point coverage: gpt-4o-mini judges, for each generated letter, which of the extracted key points it expresses.
# usage: python kp_judge.py            (all generation files known to evaluate.COND; resumable, cached in results/kp_*.json)
import os, json, re
from concurrent.futures import ThreadPoolExecutor
import evaluate as E
# the OpenAI key is read from the OPENAI_API_KEY environment variable
from openai import OpenAI
cli = OpenAI()
SYS = ('أنت مقيّم. أمامك نقاط معلومات وخطاب. لكل نقطة أجب 1 إذا كان الخطاب يذكر مضمونها (ولو بصياغة أخرى) '
       'ذكرًا صحيحًا غير مناقض، و0 إذا لم يذكرها أو ذكرها مخالفة. أعد JSON: {"scores": [..]} بعدد النقاط وترتيبها.')


def points(i):
    f = E.F[str(E.data[i]['row'])]['fields']
    return [E.vs(p) if hasattr(E, 'vs') else str(p) for p in (f.get('النقاط_الأساسية') or []) if p]


def judge(i, text):
    P = points(i)
    if not P: return None
    msg = 'النقاط:\n' + '\n'.join(f'{k + 1}. {p}' for k, p in enumerate(P)) + '\n\nالخطاب:\n' + text
    for _ in range(3):
        r = cli.chat.completions.create(model='gpt-4o-mini', temperature=0, response_format={'type': 'json_object'},
                                        messages=[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': msg}])
        try:
            s = [int(x) for x in json.loads(r.choices[0].message.content)['scores']]
            if len(s) == len(P): return s
        except Exception:
            pass
    return None


def run(cond, k, path):
    out = f'{E.H}/results/kp_{cond}_run{k}.json'
    res = json.load(open(out)) if os.path.exists(out) else {}
    tx = E.load(path)
    todo = [i for i in range(len(tx)) if str(i) not in res]
    with ThreadPoolExecutor(8) as ex:
        for i, s in zip(todo, ex.map(lambda i: judge(i, tx[i]), todo)):
            res[str(i)] = s
    json.dump(res, open(out, 'w'), indent=0)
    v = [sum(s) / len(s) for s in res.values() if s]
    print(cond, k, round(sum(v) / len(v), 3), flush=True)


if __name__ == '__main__':
    tasks = [(c, k, p) for c, files in E.COND.items() for k, p in enumerate(files, 1)]
    tasks.append(('real', 1, None))
    for c, k, p in tasks:
        if p is None:   # sanity check on the real letters
            out = f'{E.H}/results/kp_real_run1.json'
            if not os.path.exists(out):
                json.dump({str(i): judge(i, E.real[i]) for i in range(len(E.real))}, open(out, 'w'), indent=0)
            continue
        run(c, k, p)
