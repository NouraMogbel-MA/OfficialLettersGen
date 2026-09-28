# Method 1: facebook/m2m100_418M fine-tuned on the question alone (run on Google Colab; about 30 minutes of training + 30 of generation).
# Lengths set from the data (targets up to 839 tokens -> 1024; inputs <=123 -> 128).
# Learning rate chosen on Dev from {2e-5, 5e-5}; up to 15 epochs, early stopping (patience 3) on dev loss.
# Warmup 10%, label padding ignored. Generation: 3 runs, lengths from data.
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sentencepiece', 'datasets', 'accelerate'], check=False)
import os, json, time, math, shutil, torch, pandas as pd, numpy as np, transformers
from datasets import Dataset
from transformers import (M2M100ForConditionalGeneration, M2M100Tokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback, set_seed)
# usage: python c1_train_m2m_question.py [DATA_DIR] [OUT_DIR]; DATA_DIR holds train.csv, dev.csv, test.csv (default: data/c1_m2m_question)
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'c1_m2m_question')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'outputs', 'c1_m2m_question', 'run')
os.makedirs(OUT, exist_ok=True)
PROMPT = "اكتب رسالة رسمية باللغة العربية بعنوان 'الموضوع: {q}' بدءًا بـ 'السيد/السيدة المحترم/ة' وانتهاء بـ 'مع التحية، مؤسسة مسك'"
MAX_IN, MAX_OUT = 128, 1024
tok = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang='ar', tgt_lang='ar')
tr_df = pd.read_csv(f'{D}/train.csv', encoding='utf-16', sep='\t'); dv_df = pd.read_csv(f'{D}/dev.csv', encoding='utf-16', sep='\t')
for df in (tr_df, dv_df):
    df['ground_truth'] = df['ground_truth'].str.replace('\n', ' ')


def prep(ex):
    mi = tok([PROMPT.format(q=q) for q in ex['question']], max_length=MAX_IN, truncation=True)
    mi['labels'] = tok(text_target=ex['ground_truth'], max_length=MAX_OUT, truncation=True)['input_ids']
    return mi


tr = Dataset.from_pandas(tr_df[['question', 'ground_truth']]).map(prep, batched=True, remove_columns=['question', 'ground_truth'])
dv = Dataset.from_pandas(dv_df[['question', 'ground_truth']]).map(prep, batched=True, remove_columns=['question', 'ground_truth'])
steps_per_epoch = math.ceil(len(tr) / 2 / 4)
results = {}
for lr in (2e-5, 5e-5):
    set_seed(42)
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    od = f'{OUT}/lr{lr:g}'
    args = TrainingArguments(
        output_dir=od, num_train_epochs=15, learning_rate=lr,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
        weight_decay=0.01, warmup_steps=int(0.1 * steps_per_epoch * 15), lr_scheduler_type='linear',
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        logging_steps=20, fp16=torch.cuda.is_available(), seed=42, report_to='none')
    tr_ = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=dv,
                  data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    t0 = time.time(); tr_.train()
    ev = tr_.evaluate()
    results[lr] = {'dev_loss': ev['eval_loss'], 'best_ckpt': tr_.state.best_model_checkpoint, 'epochs_run': tr_.state.epoch,
                   'minutes': round((time.time() - t0) / 60, 1), 'log_history': tr_.state.log_history,
                   'warmup_steps': args.warmup_steps, 'steps_per_epoch': steps_per_epoch}
    tr_.save_model(f'{od}/best'); tok.save_pretrained(f'{od}/best')
    print('LR', lr, 'dev_loss', ev['eval_loss'], 'epochs', tr_.state.epoch, 'min', results[lr]['minutes'], flush=True)
    del tr_, model; torch.cuda.empty_cache()
best_lr = min(results, key=lambda k: results[k]['dev_loss'])
json.dump({'results': {str(k): v for k, v in results.items()}, 'best_lr': best_lr, 'max_in': MAX_IN, 'max_out': MAX_OUT},
          open(f'{OUT}/selection.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('BEST LR', best_lr, flush=True)
model = M2M100ForConditionalGeneration.from_pretrained(f'{OUT}/lr{best_lr:g}/best').to('cuda' if torch.cuda.is_available() else 'cpu').eval()
TEST_IDS = [e['doc_id'] for e in json.load(open(os.path.join(HERE, '..', 'data', 'inputs.json'), encoding='utf-8'))['test']]
test = pd.read_csv(f'{D}/test.csv', encoding='utf-16', sep='\t')
for i, seed in enumerate((1, 2, 3), 1):
    set_seed(seed); rows = []
    for doc_id, q, gt in zip(TEST_IDS, test['question'], test['ground_truth']):
        x = tok(PROMPT.format(q=q), return_tensors='pt', truncation=True, max_length=MAX_IN).to(model.device)
        with torch.no_grad():
            o = model.generate(**x, max_length=1024, min_length=128, temperature=0.3, top_k=50, top_p=0.95,
                               do_sample=True, no_repeat_ngram_size=3, forced_bos_token_id=tok.get_lang_id('ar'))
        rows.append({'doc_id': doc_id, 'question': q, 'ground_truth': gt, 'answer': tok.decode(o[0], skip_special_tokens=True)})
    pd.DataFrame(rows).to_csv(f'{OUT}/gen_run{i}.csv', index=False, encoding='utf-16', sep='\t')
    print('saved run', i, flush=True)
print('ALL DONE', flush=True)
