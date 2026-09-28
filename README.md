# OfficialLettersGen

Code, data and results for the MA thesis "تقنيات التوجيه اللغوي والتقييم في الكتابة الإدارية باستخدام نماذج اللغة الكبيرة" (Language Prompt Techniques and Evaluation for Administrative Writing Using LLMs), Department of Arabic Language and Literature.

**The data are anonymized.** In every data, output and result file, personal names, mobile and phone numbers, e-mail addresses and national ID numbers are replaced by `[anonymized]`. Names that are part of an institution or a place are kept, such as جامعة الأميرة نورة بنت عبدالرحمن and طريق هشام بن عبدالملك. The results reported here and in the thesis were computed on the full data, so re-running the evaluation on this copy gives slightly different numbers.

The thesis asks how a language model can be guided to write Saudi administrative letters that are correct in both formulas and content. It compares **fine-tuning a small model** with **prompting a large model**, and tests whether a prompt built from the linguistic patterns of the corpus improves prompting.

## Data

All CSV files are UTF-16 LE with BOM and tab-separated, so that Excel opens the long Arabic texts correctly; read them with `pandas.read_csv(path, encoding='utf-16', sep='\t')`.

- `data/corpus.xlsx`: 498 authentic Saudi administrative letters (`OC_type`, `doc_id`, `title`, `Year`, `Text`), sorted by the document number in `doc_id`. The thesis refers to a letter by this number, e.g. "الوثيقة 275" is `letter00275`. The numbers come from the collection and are not consecutive; `letter00043` and `letter330` are each used for two different letters, and `letter00017` and `letter00098` carry the suffixes -A, -B, -C.
- `data/split.json`: the train / dev / test split (348 / 75 / 75), with each letter's row in the corpus, its `doc_id`, the question built from its type and title, and its type.
- `data/fields_raw.json`, `data/fields.json`: content information extracted from every letter by gpt-4o-mini. There are eight items: type, addressee (entity and position), subject, previous-letter reference (number and date), 1–4 key points, the request, contact details and sender. The information contains no formulas: no honorific titles, blessings, greetings, request formulas or closings. `fields.json` is the checked version.
- `data/inputs.json`: for every letter of the split, the question, the information as text, and the real letter.
- `data/prompts.json`: the three prompts given to ALLaM for each test letter.
- `data/c1_m2m_question/`: the question/reference pairs that method 1 was trained on. In this copy the Arabic-Indic digits of 216 letters were lost when it was saved, so they appear as `?`. The rest of the experiment and all evaluation use `data/corpus.xlsx`.

## The five methods

| # | Folder | Model | Input | Adaptation |
|---|---|---|---|---|
| 1 | `outputs/c1_m2m_question` | facebook/m2m100_418M | question only | fine-tuning |
| 2 | `outputs/c2_m2m_fields` | facebook/m2m100_418M | question + letter information | fine-tuning |
| 3 | `outputs/c3_allam_zero` | humain-ai/ALLaM-7B-Instruct-preview | question + letter information | zero-shot prompt |
| 4 | `outputs/c4_allam_few` | ALLaM-7B-Instruct-preview | question + letter information | prompt with two real letters of the same type |
| 5 | `outputs/c5_allam_ling` | ALLaM-7B-Instruct-preview | question + letter information | linguistic prompt: letter structure, shared formulas and the type's distinctive formulas from chapter 2 |

Every method wrote each of the 75 test letters in three runs (`gen_run1.csv` … `gen_run3.csv`: doc_id, question, ground_truth, answer). `doc_id` is the document number of the original letter in `data/corpus.xlsx`, so a generated letter can be found from the number the thesis gives. The examples quoted in chapter 4 of the thesis are from run 1.

- The M2M100 models were trained with learning rate 2e-5, up to 15 epochs with early stopping on dev loss, 10% warm-up, and padding ignored in the loss (`train_log.json`).
- ALLaM was used as published, in bf16, with temperature 0.3, top-p 0.95, top-k 50 and at most 1024 new tokens.

## Evaluation

- `scripts/evaluate.py`: linguistic indicators and information indicators for every run, written to `results/ling_info.*`, plus the RAGAS inputs.
  - The linguistic indicators follow chapter 2 of the thesis: shared and distinctive formulas, the addressee line, whether the honorific matches the real letter, closings, and formulas foreign to Saudi administrative letters.
  - The information indicators are coverage of numbers, e-mail addresses and the addressee's position, plus invented numbers and e-mail addresses.
- `scripts/keypoint_judge.py`: gpt-4o-mini judges whether each key point appears in the letter (`results/kp_*.json`).
- `scripts/run_ragas.py`, `scripts/run_ragas_all.sh`: RAGAS 0.2.15 with five metrics and gpt-4o-mini as judge (`results/ragas_*.csv`). The context and the reference are the real letter.
- `scripts/aggregate.py`: produces `results/final.json`. All numbers in the thesis come from this file.

The key for the OpenAI API is read from the `OPENAI_API_KEY` environment variable.

## Main results (test set, 75 letters)

The columns are the real letters and the five methods. Each value is the mean of the three runs of that method.

| Indicator | Real letters | 1. Fine-tuning on the question | 2. Fine-tuning on the information | 3. Zero-shot prompt | 4. Two-example prompt | 5. Linguistic prompt |
|---|---|---|---|---|---|---|
| Distinctive formulas of the letter's type | 34% | 22% | 30% | 7% | 26% | 45% |
| Letter starts with the addressee line | 93% | 100% | 100% | 45% | 93% | 5% |
| Honorific matches the real letter | 100% | 61% | 79% | 35% | 62% | 54% |
| Foreign greeting "تحية طيبة" | 0% | 0% | 0% | 76% | 35% | 33% |
| Numbers, e-mails and position carried over | 100% | 13% | 90% | 71% | 89% | 82% |
| Key points mentioned | 100% | 40% | 96% | 99% | 98% | 96% |
| Letters with invented numbers or e-mails | 0% | 74% | 42% | 3% | 15% | 13% |
| Faithfulness (RAGAS) | – | 0.21 | 0.67 | 0.68 | 0.64 | 0.72 |
| Answer correctness (RAGAS) | – | 0.42 | 0.61 | 0.64 | 0.65 | 0.63 |

- The input decides the content. The same small model scored 0.21 in faithfulness when trained on the question alone, and 0.67 when trained on the question with the letter information.
- Fine-tuning on the corpus brings the letter closer to Saudi conventions in formulas and honorifics: the honorific matched the real letter in 79% of the letters of method 2.
- The linguistic prompt is the highest of the three prompts in faithfulness (0.72) and in distinctive formulas (45%, against 34% in the real letters), but it seldom starts with the addressee line (5%).

**Note:** generated letters are drafts and must be reviewed before use.

## Browse the letters

Download the repository (the green **Code** button, then **Download ZIP**, and unzip it) or clone it, then open `docs/index.html` in a web browser by double-clicking it. No installation, server or internet connection is needed; the page is not published online. It shows the 75 test letters: the information given to the model, the original letter, and the letters written by the five methods side by side, with each method's prompt and the faithfulness score of each letter. Its data file `docs/data.js` is built by `scripts/build_page_data.py`.

## Demo (runs locally)

`app/app.py` is a small Streamlit interface for trying the methods on your own computer. It is not hosted online. It needs Python 3.10 or later and an NVIDIA GPU with about 16 GB of memory.

```
git clone https://github.com/NouraMogbel-MA/OfficialLettersGen.git
cd OfficialLettersGen
pip install -r requirements.txt
streamlit run app/app.py
```

The interface then opens in the browser at http://localhost:8501.

- Enter the letter type and its information, or fill them in from one of the 75 test letters. Then choose a method and press "اكتب الخطاب".
- The interface shows the exact prompt sent to the model; `app/prompting.py` builds it with the same text as the experiment.
- Methods 3–5 use ALLaM-7B-Instruct-preview, which is downloaded from Hugging Face on first use and needs a GPU with about 16 GB. Another model or a local copy can be set with the `ALLAM_MODEL` environment variable.
- Method 2 uses the trained model in `models/m2m100_fields/` (or the folder named by `M2M_MODEL_DIR`).

## Trained models

`models/` holds the two fine-tuned facebook/m2m100_418M models in fp16 (safetensors, about 0.97 GB each, stored with Git LFS; run `git lfs install` before cloning to download them):

- `models/m2m100_question/`: method 1, trained on the question alone.
- `models/m2m100_fields/`: method 2, trained on the question with the letter information.

They were retrained in 2026 on the full corpus with the same data, split, settings and seed as the thesis (`scripts/c1_train_m2m_question.py` at learning rate 2e-5, and `scripts/c2_train_m2m_fields.py`) on a Colab T4, because the weights of the thesis run were not kept. The thesis numbers come from the saved outputs of the original run and do not change. Training on a different GPU does not give identical weights, so the retrained models were checked against run 1 of the thesis on the 75 test letters:

| | Method 1, thesis | Method 1, retrained | Method 2, thesis | Method 2, retrained |
|---|---|---|---|---|
| Lowest dev loss (epoch) | 1.401 (14) | 1.401 (14) | 1.022 (10) | 1.022 (10) |
| Distinctive formulas of the letter's type | 22% | 24% | 30% | 26% |
| Honorific matches the real letter | 61% | 62% | 79% | 79% |
| Letter starts with the addressee line | 100% | 100% | 100% | 100% |
| Numbers, e-mails and position carried over | 13% | 11% | 90% | 90% |
| Letters with invented numbers or e-mails | 76% | 76% | 47% | 40% |

Each folder also has the training log (`train_log.json`). Generated letters are drafts: a model trained on real letters may repeat details it saw in training, so every generated letter must be reviewed before use.

## Reproducing

```
pip install -r requirements.txt
python scripts/extract_fields.py            # data/fields_raw.json   (OpenAI)
python scripts/postfix_fields.py            # data/fields.json
python scripts/prepare_inputs.py            # data/inputs.json, data/prompts.json
python scripts/c1_train_m2m_question.py     # method 1 (GPU)
python scripts/c2_train_m2m_fields.py       # method 2 (GPU)
python scripts/c3_c5_generate_allam.py      # methods 3-5 (GPU)
python scripts/evaluate.py && python scripts/keypoint_judge.py && sh scripts/run_ragas_all.sh && python scripts/aggregate.py
```

The GPU scripts were run on Google Colab (NVIDIA L4); the models in `models/` were retrained on an NVIDIA T4.

## License

- Code (`scripts/`, `app/`): MIT License, see `LICENSE`.
- Data, generated letters and results (`data/`, `outputs/`, `results/`): Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), see `LICENSE-DATA`. They may be used for research and teaching with attribution, not for commercial purposes.

## Citation

Almogbel, Noura (2026). تقنيات التوجيه اللغوي والتقييم في الكتابة الإدارية باستخدام نماذج اللغة الكبيرة [Language Prompt Techniques and Evaluation for Administrative Writing Using LLMs] (Master's thesis). Department of Arabic Language and Literature, King Saud University, Riyadh(SA). 
