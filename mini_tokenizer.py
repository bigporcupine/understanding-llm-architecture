"""A tiny, dependency-free Byte Pair Encoding tokenizer."""

from collections import Counter


class MiniBPETokenizer:
    def __init__(self):
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    @staticmethod
    def _merge_pair(tokens, pair):
        """Replace every non-overlapping occurrence of a pair."""
        merged = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, vocab_size, verbose=False):
        """Learn merge rules from text and build the token vocabulary."""
        if not text:
            raise ValueError("Training text must not be empty")

        tokens = list(text)  # This educational version starts with Unicode characters.
        base_vocab = set(tokens)
        if vocab_size < len(base_vocab):
            raise ValueError(f"vocab_size must be at least {len(base_vocab)}")

        self.merges = []
        vocab = set(base_vocab)

        while len(vocab) < vocab_size and len(tokens) > 1:
            pair_counts = Counter(zip(tokens, tokens[1:]))
            if not pair_counts:
                break

            # Counter preserves insertion order, making tie-breaking deterministic.
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = "".join(best_pair)
            if new_token in vocab:
                break

            tokens = self._merge_pair(tokens, best_pair)
            self.merges.append(best_pair)
            vocab.add(new_token)

            if verbose:
                count = pair_counts[best_pair]
                print(f"merge {len(self.merges):2}: {best_pair!r} -> "
                      f"{new_token!r} (frequency: {count})")

        # Sorting the base vocabulary gives stable token IDs across runs.
        ordered_vocab = sorted(base_vocab)
        ordered_vocab += ["".join(pair) for pair in self.merges]
        self.token_to_id = {token: i for i, token in enumerate(ordered_vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def tokenize(self, text):
        """Apply learned merge rules in training order."""
        if not self.token_to_id:
            raise RuntimeError("Train the tokenizer before using it")
        tokens = list(text)
        unknown = set(tokens) - self.token_to_id.keys()
        if unknown:
            raise ValueError(f"Characters outside the vocabulary: {sorted(unknown)!r}")
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        return tokens

    def encode(self, text):
        return [self.token_to_id[token] for token in self.tokenize(text)]

    def decode(self, ids):
        return "".join(self.id_to_token[token_id] for token_id in ids)

    @property
    def vocab_size(self):
        return len(self.token_to_id)


def show(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"\nText: {text!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"IDs: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)!r}")


if __name__ == "__main__":
    corpus = (
        "low lower lowest low lower lowest "
        "machine learning is useful machine learning is powerful"
    )

    tokenizer = MiniBPETokenizer()
    initial_size = len(set(corpus))
    target_size = initial_size + 18

    print(f"Initial character vocabulary: {initial_size}")
    print(f"Target vocabulary size: {target_size}\n")
    tokenizer.train(corpus, vocab_size=target_size, verbose=True)
    print(f"\nFinal vocabulary size: {tokenizer.vocab_size}")

    show(tokenizer, "low lowest")
    show(tokenizer, "machine learning")
    show(tokenizer, "low machine")
