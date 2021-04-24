This repo is home to experiments on the `Power` model's `Texter` component.
These include

- Comparing the `Texter` against a baseline
- Varying the pooling mechanism to obtain sentence embeddings

The repo contains only code for training and evaluating the texter. It does
not contain code for creating a `Texter Dataset`.

# Train `Texter`

Given a `Texter Dataset v5` and a `Power Split v2` a `Texter` is trained
by running `train.py`:

```bash
python src/train.py \
  data/power/samples/cde-irt-5-marked/ \
  100 \
  5 \
  data/power/split/cde-100/ \
  data/power/texter/cde-irt-5-marked.pkl
```

# Evaluate `Texter`

By passing the `--test` flag the Texter is evaluated against predictable
facts after training:

```bash
python src/train.py \
  data/power/samples/cde-irt-5-marked/ \
  100 \
  5 \
  data/power/split/cde-100/ \
  data/power/texter/cde-irt-5-marked.pkl
  --test
```

Evaluation against all facts is performed by running `eval.py`:

```bash
python src/eval.py \
  data/power/texter/cde-irt-5-marked.pkl \
  5 \
  data/power/split/cde-100/ \
  data/irt/text/cde-irt-5-marked/
```
