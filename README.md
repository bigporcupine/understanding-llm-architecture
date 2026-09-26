# Understanding Language Models

A hands-on project for understanding language models from first principles. It favors transparent
formulas, tensor shapes, and observable training behavior over framework abstractions.

## Start with the notebook

Recommended entry point: [From Tokens to a Tiny Language Model](notebooks/01_from_tokens_to_tiny_lm.ipynb)

The notebook follows one continuous path:

```text
token IDs → embedding → causal attention → multi-head attention
          → RMSNorm → RoPE → SwiGLU → Transformer → loss → training
```

Install and launch:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/jupyter lab
```

Open `notebooks/01_from_tokens_to_tiny_lm.ipynb` and run the cells from top to bottom.

The same experiments are also available from the command line:

```bash
.venv/bin/python transformer_tutorial.py --lesson attention
.venv/bin/python transformer_tutorial.py --lesson all
.venv/bin/python transformer_tutorial.py --lesson train --steps 200
```

For more explanation and exercises, see the [learning guide](LEARNING_GUIDE.md).

Run the correctness checks with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Lesson 0: Mini BPE Tokenizer

A small, dependency-free implementation of Byte Pair Encoding (BPE) in Python.
It is intended for learning and experimentation: the entire algorithm fits in
roughly 100 lines and exposes each important step.

## How BPE works

A tokenizer converts text into a sequence of tokens. Word-level tokenizers
produce short sequences but require very large vocabularies and struggle with
unseen words. Character-level tokenizers need tiny vocabularies but produce
long sequences. BPE learns reusable subwords and provides a practical middle
ground.

BPE training begins with one token per character. It then repeatedly:

1. Counts every adjacent token pair in the training corpus.
2. Selects the most frequent pair.
3. Merges that pair into a new token.
4. Adds the new token to the vocabulary.

For example, repeated training words might produce these merges:

```text
l + o   -> lo
lo + w  -> low
e + r   -> er
```

The word `lower` can then be represented as `low` + `er`, using fewer tokens
than a character-level representation. Encoding new text applies the learned
merge rules in the same order in which they were learned.

## Vocabulary size

The initial vocabulary contains every distinct character in the corpus. Each
successful merge adds one token. A larger target vocabulary can represent
frequent text with fewer tokens, but it also makes a language model's embedding
table larger.

If the vocabulary size is `V` and the model hidden size is `D`, the embedding
table has shape `V x D`. After this tokenizer maps text to token IDs, each ID
selects one row from that table:

```text
text -> BPE tokens -> token IDs -> embedding vectors -> model
```

The tokenizer stops at token IDs. The notebook and Transformer tutorial continue
from those IDs through embeddings, attention, and language-model training.

## Usage

Run the included demonstration:

```bash
python3 mini_tokenizer.py
```

Or import the class:

```python
from mini_tokenizer import MiniBPETokenizer

corpus = "low lower lowest low lower lowest"
tokenizer = MiniBPETokenizer()
tokenizer.train(corpus, vocab_size=20)

tokens = tokenizer.tokenize("low lowest")
ids = tokenizer.encode("low lowest")
text = tokenizer.decode(ids)

print(tokens)
print(ids)
print(text)
```

Increase or decrease `vocab_size` and compare `len(tokens)` to see the central
tradeoff directly.

## Design and limitations

This implementation deliberately favors clarity over production performance:

- It starts from Unicode characters, not UTF-8 bytes.
- It trains on one continuous text sequence without pre-tokenization.
- It rejects characters that were absent from the training corpus.
- It stores the model in memory and does not yet serialize it.
- Its pair counting and merging approach is not optimized for large corpora.

Production tokenizers commonly use byte-level input, special tokens,
pre-tokenization rules, model serialization, and substantially faster training
data structures.

## Continue to the language model

After learning BPE, continue with the [language-model architecture learning guide](LEARNING_GUIDE.md).
It connects token IDs to a small decoder-only Transformer and includes runnable
experiments for causal attention, RMSNorm, RoPE, SwiGLU, multi-head attention,
and tiny-corpus training.

## License

MIT
