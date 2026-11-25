# Raqib: Arabic Customer-Complaint NLP Prototype

A small prototype built during the @SDAIAAcademy *Applied NLP* course (Labs 1 to 6).
It reads customer complaints written in Arabic (Saudi white dialect, with Lebanese
examples for comparison), classifies them by topic, and finds similar complaints.

**Topics:** `delivery`, `payment`, `quality`, `service`

## What it does

| Step | Code | Lab |
|------|------|-----|
| Clean text, normalise Arabic, mask phone numbers and emails | `src/preprocessing.py` | 1, 4 |
| Rule-based dialect audit and clitic segmentation | `src/dialect.py` | 4 |
| Load data and split by customer (no data leakage) | `src/data.py` | 3 |
| Fine-tune CAMeLBERT for topic classification | `src/classifier.py` | 3 |
| Semantic search (FAISS) and Cross-Encoder re-ranking | `src/search.py` | 5 |
| Sliced evaluation, bootstrap CI, error taxonomy | `src/evaluation.py` | 6 |
| Interactive demo (topic, dialect, similar complaints) | `app.py`, `src/predict.py` | 3, 4 |

## Project structure

```
raqib-complaints-nlp/
├── main.py              # command-line entry point
├── app.py               # Gradio demo
├── data/complaints.csv  # 16 sample complaints (8 customers)
├── src/                 # the modules above
├── tests/test_basic.py  # unit tests for the lightweight parts
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The commands `preprocess`, `dialect`, `split` and `evaluate` only need
`pandas`, `numpy` and `scikit-learn`. `train` and `search` need the full
requirements and download models from Hugging Face (a GPU such as Colab T4 is
recommended for `train`).

## Usage

```bash
python main.py preprocess
python main.py dialect
python main.py split
python main.py evaluate
python main.py train --epochs 4
python main.py search "الطلب ما وصل" --rerank
python -m pytest
```

## Demo

A small web app: type a complaint and see its topic, detected dialect, the text
after PII masking, and the most similar known complaints.

```bash
pip install gradio scikit-learn pandas
python app.py            # http://127.0.0.1:7860
python app.py --share    # public link (useful in Google Colab)
```

In Google Colab:

```python
!git clone https://github.com/7Jumanah7/raqib---complaints---nlp.git
%cd raqib---complaints---nlp
!pip install -q gradio scikit-learn pandas
!python app.py --share
```

The demo uses a lightweight TF-IDF classifier trained on the 16 sample
complaints, so it starts instantly and needs no model downloads. The CAMeLBERT
classifier and the FAISS semantic search live in `src/classifier.py` and
`src/search.py`. Because the training set is tiny, demo predictions are rough.

## Design decisions

- **Split by customer, not by row.** One customer writes in a similar style, so
  putting their complaints in both train and test would leak information.
  `GroupShuffleSplit` prevents this and `group_split` raises an error if any
  customer is shared.
- **PII is masked before tokenisation.** Saudi phone numbers become `[PHONE]`,
  emails become `[EMAIL]`.
- **Arabic normalisation is not applied before CAMeLBERT.** It is used for
  dialect detection only, because the model was pre-trained on text that keeps
  these letter distinctions.
- **Small imports.** Heavy libraries load inside the functions that need them.

## Limitations (please read)

- The dataset is tiny (16 hand-written complaints), so any score is noisy.
  This is a prototype of the pipeline, not a trained production model.
- The dialect audit is a word list, not a model. Words that are not in the
  list give `Unknown`. The clitic splitter is rule-based and wrongly splits
  words such as `بطاقتي`.
- `python main.py evaluate` runs on **illustrative demo values**, not on the
  output of a trained model.
- The `train` and `search` commands depend on downloading models, and were
  adapted from the course labs.

## Possible next steps

- Collect a larger, real, labelled dataset and add more topics.
- Report Macro-F1 per dialect slice from real model predictions.
- Add an out-of-scope class and a threshold for low-confidence predictions.
- Replace the rule-based clitic splitter with a morphological analyser.
