# Methods 3-5: ALLaM-7B-Instruct-preview (no training), bf16, batched sampling.
# Decoding as close to M2M100 as possible: temperature 0.3, top_p 0.95, top_k 50, up to 1024 new tokens.
# (no_repeat_ngram_size is not used: it blocks the repeated formula «السلام عليكم ورحمة الله وبركاته».)
# Resumable: CSV rewritten after each batch; finished rows are skipped.
import os, json, sys, time, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT, MP = 'allam_out', 'humain-ai/ALLaM-7B-Instruct-preview'   # MP: a local copy or the Hugging Face id
os.makedirs(OUT, exist_ok=True)
P = json.load(open(f'{D}/prompts.json', encoding='utf-8'))
data = json.load(open(f'{D}/inputs.json', encoding='utf-8'))['test']
tok = AutoTokenizer.from_pretrained(MP); tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16, device_map='cuda').eval()
seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else '1,2,3').split(',')]
conds = (sys.argv[2] if len(sys.argv) > 2 else 'zero,few,ling').split(',')
BS = {'zero': 8, 'ling': 8, 'few': 4}
enc = lambda text: tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
for seed in seeds:
    for cond in conds:
        path = f'{OUT}/gen_{cond}_run{seed}.csv'
        done = pd.read_csv(path, encoding='utf-16', sep='\t') if os.path.exists(path) else pd.DataFrame(columns=['i', 'answer', 'new_tokens', 'prompt_tokens'])
        done = done[[c for c in ('i', 'answer', 'new_tokens', 'prompt_tokens') if c in done]]
        have = set(done.i.astype(int)); rows = done.to_dict('records')
        todo = sorted([p for p in P if p['i'] not in have], key=lambda p: len(p[cond]))
        t0 = time.time()
        for b in range(0, len(todo), BS[cond]):
            batch = todo[b:b + BS[cond]]
            x = tok([enc(p[cond]) for p in batch], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            set_seed(seed * 1000 + batch[0]['i'])
            with torch.no_grad():
                o = model.generate(**x, do_sample=True, temperature=0.3, top_p=0.95, top_k=50, max_new_tokens=1024,
                                   pad_token_id=tok.pad_token_id)
            for k, (p, seq) in enumerate(zip(batch, o)):
                gen = seq[x['input_ids'].shape[1]:]
                rows.append({'i': p['i'], 'answer': tok.decode(gen, skip_special_tokens=True).strip(),
                             'new_tokens': int((gen != tok.pad_token_id).sum()), 'prompt_tokens': int(x['attention_mask'][k].sum())})
            pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-16', sep='\t')
            print(cond, seed, len(rows), round((time.time() - t0) / 60, 1), 'min', flush=True)
        df = pd.DataFrame(rows).sort_values('i')
        df.insert(0, 'question', [data[int(i)]['question'] for i in df.i]); df.insert(1, 'ground_truth', [data[int(i)]['letter'] for i in df.i])
        df.insert(0, 'doc_id', [data[int(i)]['doc_id'] for i in df.i])
        df.to_csv(path, index=False, encoding='utf-16', sep='\t')
        print('saved', path, len(df), round((time.time() - t0) / 60, 1), 'min', flush=True)
print('ALLAM DONE', flush=True)
