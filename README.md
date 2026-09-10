# Financial News Sentiment Classification

Sentiment classification on financial text using the Financial PhraseBank benchmark. Compares a classical TF-IDF baseline against a fine-tuned DistilBERT transformer and an off-the-shelf domain model (FinBERT), with per-class analysis, error analysis and token-level explainability.

## Why financial sentiment is its own problem

Financial language inverts everyday sentiment. *"The company cut costs"* is positive. *"Growth slowed to 8%"* is negative despite containing growth. General-purpose sentiment models get this wrong routinely, which is what makes a domain-specific model worth training.

## Dataset

[Financial PhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) — 4,846 sentences from financial news, each labelled negative / neutral / positive by finance students.

The classes are heavily imbalanced, with neutral dominating. A model that predicts neutral for everything scores a deceptively high accuracy while being useless, so **macro-F1 is the headline metric here, not accuracy**.

## Results

Per-class F1 on the held-out test set:

| Class | TF-IDF + LogReg | DistilBERT (fine-tuned) | FinBERT (zero-shot) |
|---|---|---|---|
| negative | 0.673 | 0.812 | 0.838 |
| neutral | 0.822 | 0.874 | 0.897 |
| positive | 0.612 | 0.796 | 0.860 |
| **Macro-F1** | **0.702** | **0.827** | **0.865** |

### Reading these numbers correctly

FinBERT scores highest — but that result **cannot be interpreted as a fair win**.

FinBERT was itself fine-tuned on Financial PhraseBank. A random test split of this benchmark therefore overlaps its training data, so its score reflects memorisation as much as capability. It is a contaminated reference point, not a baseline that was beaten.

The clean comparison is **DistilBERT vs. TF-IDF**: fine-tuning bought roughly **+0.12 macro-F1** over bag-of-words, and the gain is concentrated in the minority classes rather than spread evenly —

- positive: 0.612 → 0.796
- negative: 0.673 → 0.812
- neutral: 0.822 → 0.874

which is exactly where a model earns its cost on imbalanced data.

## Approach

**Baseline — TF-IDF + Logistic Regression.** Bigrams to catch negation and multi-word phrases (`cut costs`, `not profitable`), stopwords deliberately kept since *not*, *no* and *down* carry the signal here, class-weighted to counter the neutral-heavy imbalance. A genuinely strong floor on short formulaic sentences — the transformer has to clearly beat it to justify its cost.

**DistilBERT fine-tuning.** Written as an explicit PyTorch training loop rather than the `Trainer` API, so batching, scheduling, gradient clipping and early stopping are all visible and adjustable:

- Class-weighted cross-entropy, matching the baseline's imbalance handling
- AdamW with linear warmup (10% of steps) — warmup stops early batches from wrecking pretrained weights
- Gradient clipping at 1.0
- Early stopping on validation **macro-F1**, not loss, restoring the best checkpoint rather than the last
- `MAX_LEN` chosen against the observed token-length distribution rather than guessed

**Three-way stratified split.** The validation set exists so early stopping and every hyperparameter choice happen without touching test data.

**Error analysis.** Confusion breakdown, plus the *confidently wrong* predictions — the most informative failures, since they show where the model learned a shortcut rather than the meaning. A confidence histogram tests whether the softmax score is trustworthy enough to route uncertain cases to a human reviewer.

**Explainability.** Occlusion-based token attribution: remove one word at a time, re-run the model, measure how far the predicted class probability falls. Model-agnostic and dependency-free.

## Running it

Open `financial_sentiment_Mohit.ipynb` in Colab with a **GPU runtime** (Runtime → Change runtime type → T4). On CPU the fine-tuning step takes well over an hour.

The notebook expects `all-data.csv` (Financial PhraseBank) in the working directory:

```python
df = pd.read_csv("all-data.csv", encoding="latin-1", names=["label_name", "sentence"])
```

`encoding="latin-1"` matters — the file is not UTF-8 and will throw without it.

Then run top to bottom. Section 11 saves the fine-tuned model and verifies it reloads with identical predictions.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy, matplotlib, seaborn

## Known Limitations

- **Label noise.** Includes sentences where annotators did not fully agree. More data than the all-agree subset, but noisier.
- **Contaminated comparison.** The FinBERT result needs re-testing on a corpus outside its training data before it means anything.
- **Sentence-level, not document-level.** Real financial text arrives as articles and filings where sentiment is mixed and context spans sentences.
- **Annotator bias.** Labels come from students judging text in isolation, without market context.
- **No market validation.** This measures agreement with human sentiment labels, *not* whether sentiment predicts returns. That needs timestamped, ticker-linked headlines aligned to prices point-in-time — a separate project, and where look-ahead bias does the most damage.

## Next Steps

- Re-run the FinBERT comparison on a held-out financial corpus outside its training data, to separate memorisation from capability
- Compare against full BERT and RoBERTa to quantify what distillation costs
- Document-level modelling over full articles rather than isolated sentences
- Calibration (temperature scaling) so confidence can be trusted as a routing threshold
