# Method 2: facebook/m2m100_418M fine-tuned on the full input (question + extracted content fields).
# Target = the real letter only (no student frame, no template sentence). Same settings as method 1:
# lr 2e-5, up to 15 epochs, early stopping (patience 3) on dev loss, warmup 10%, label padding -100.
# Generation: 3 runs, same decoding as method 1 (incl. no_repeat_ngram_size=3); plus one ablation run without it.
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sentencepiece', 'datasets', 'accelerate'], check=False)
import os, json, time, math, torch, pandas as pd
from datasets import Dataset
from transformers import (M2M100ForConditionalGeneration, M2M100Tokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback, set_seed)
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = sys.argv[2] if len(sys.argv) > 2 else 'm2m_fields'
os.makedirs(OUT, exist_ok=True)
data = json.load(open(f'{D}/inputs.json', encoding='utf-8'))
MAX_IN, MAX_OUT = 512, 1024
tok = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang='ar', tgt_lang='ar')
flat = lambda s: ' '.join(str(s).split())
lens_in = [len(tok(flat(d['input']))['input_ids']) for d in data['train'] + data['dev'] + data['test']]
lens_out = [len(tok(text_target=flat(d['letter']))['input_ids']) for d in data['train'] + data['dev']]
print('input tokens max', max(lens_in), 'target tokens max', max(lens_out), flush=True)


def ds(rows):
    df = pd.DataFrame({'x': [flat(d['input']) for d in rows], 'y': [flat(d['letter']) for d in rows]})

    def prep(ex):
        mi = tok(ex['x'], max_length=MAX_IN, truncation=True)
        mi['labels'] = tok(text_target=ex['y'], max_length=MAX_OUT, truncation=True)['input_ids']
        return mi
    return Dataset.from_pandas(df).map(prep, batched=True, remove_columns=['x', 'y'])


tr, dv = ds(data['train']), ds(data['dev'])
spe = math.ceil(len(tr) / 2 / 4)
if not os.path.exists(f'{OUT}/best/config.json'):
    set_seed(42)
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    args = TrainingArguments(
        output_dir=f'{OUT}/ckpt', num_train_epochs=15, learning_rate=2e-5,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
        weight_decay=0.01, warmup_steps=int(0.1 * spe * 15), lr_scheduler_type='linear',
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        logging_steps=20, fp16=True, seed=42, report_to='none')
    t = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=dv,
                data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    t0 = time.time(); t.train(); ev = t.evaluate()
    json.dump({'dev_loss': ev['eval_loss'], 'best_ckpt': t.state.best_model_checkpoint, 'epochs_run': t.state.epoch,
               'minutes': round((time.time() - t0) / 60, 1), 'warmup_steps': args.warmup_steps, 'steps_per_epoch': spe,
               'max_in_tokens': max(lens_in), 'max_out_tokens': max(lens_out), 'log_history': t.state.log_history},
              open(f'{OUT}/train_log.json', 'w'), ensure_ascii=False, indent=1, default=str)
    t.save_model(f'{OUT}/best'); tok.save_pretrained(f'{OUT}/best')
    print('trained, dev_loss', ev['eval_loss'], 'epochs', t.state.epoch, flush=True)
    del t, model; torch.cuda.empty_cache()
model = M2M100ForConditionalGeneration.from_pretrained(f'{OUT}/best').cuda().eval()
runs = [(1, 1, 3), (2, 2, 3), (3, 3, 3), ('1_nongram', 1, 0)]
for name, seed, ng in runs:
    p = f'{OUT}/gen_run{name}.csv'
    if os.path.exists(p): continue
    set_seed(seed); rows = []
    for d in data['test']:
        x = tok(flat(d['input']), return_tensors='pt', truncation=True, max_length=MAX_IN).to('cuda')
        kw = dict(no_repeat_ngram_size=ng) if ng else {}
        with torch.no_grad():
            o = model.generate(**x, max_length=1024, min_length=128, temperature=0.3, top_k=50, top_p=0.95,
                               do_sample=True, forced_bos_token_id=tok.get_lang_id('ar'), **kw)
        rows.append({'question': d['question'], 'ground_truth': d['letter'], 'answer': tok.decode(o[0], skip_special_tokens=True)})
    pd.DataFrame(rows).to_csv(p, index=False, encoding='utf-8-sig')
    print('saved', p, flush=True)
print('M2M DONE', flush=True)
