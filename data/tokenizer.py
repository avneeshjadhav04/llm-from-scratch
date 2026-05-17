"""Byte-Pair Encoding (BPE) tokenizer implemented from scratch."""
import json
import regex as re
from typing import List, Tuple, Dict, Optional


# GPT-2 style pre-tokenization regex
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_bytes_to_unicode() -> Dict[int, str]:
    """Returns mapping from bytes to unicode characters for byte-level BPE."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTES_TO_UNICODE = get_bytes_to_unicode()
UNICODE_TO_BYTES = {v: k for k, v in BYTES_TO_UNICODE.items()}


def encode_bytes_to_word(text: str) -> str:
    """Encode text into a string where each byte is mapped to a specific unicode char."""
    return "".join(BYTES_TO_UNICODE[b] for b in text.encode("utf-8"))


def decode_word_to_bytes(word: str) -> str:
    """Decode a string of mapped unicode chars back into the original bytes/text."""
    return bytes(UNICODE_TO_BYTES[c] for c in word).decode("utf-8", errors="replace")


class BPETokenizer:
    """Byte-Pair Encoding tokenizer implemented from scratch."""

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<|endoftext|>": vocab_size,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []

    def _get_word_tokens(self, text: str) -> List[str]:
        """Pre-tokenize text using GPT-2 style regex, then byte-encode each token."""
        tokens = []
        for match in GPT2_PAT.finditer(text):
            word = match.group()
            word = encode_bytes_to_word(word)
            tokens.append(word)
        return tokens

    def _get_pairs(self, word: Tuple[str, ...]) -> Dict[Tuple[str, str], int]:
        """Count all adjacent symbol pairs in a word."""
        pairs = {}
        prev_char = word[0]
        for char in word[1:]:
            pair = (prev_char, char)
            pairs[pair] = pairs.get(pair, 0) + 1
            prev_char = char
        return pairs

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Merge all occurrences of a pair in a word (linear scan)."""
        first, second = pair
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train(self, text: str) -> None:
        """Train BPE on the given text corpus using an efficient incremental algorithm."""
        print("Pre-tokenizing text...")
        word_tokens = self._get_word_tokens(text)

        # Build word frequency map
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for word in word_tokens:
            symbols = tuple(word)
            word_freqs[symbols] = word_freqs.get(symbols, 0) + 1

        # Initialize vocab with all individual characters present in corpus
        all_chars = set()
        for word in word_freqs:
            for char in word:
                all_chars.add(char)

        base_vocab = sorted(all_chars)
        self.vocab = {char: i for i, char in enumerate(base_vocab)}
        self.inverse_vocab = {i: char for char, i in self.vocab.items()}

        num_merges = self.vocab_size - len(base_vocab) - len(self.special_tokens)

        print(f"Base vocab size: {len(base_vocab)}, Target vocab size: {self.vocab_size}")
        print(f"Training {num_merges} BPE merges...")

        # --- Efficient BPE: maintain pair counts incrementally ---
        # Map each word to its current tokenization
        current_words = {word: list(word) for word in word_freqs}

        # Build initial pair counts
        pair_counts: Dict[Tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            tokens = current_words[word]
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        for i in range(num_merges):
            if not pair_counts:
                break

            # Find best pair
            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]

            # Update words that contain best_pair and recompute pair counts
            words_to_update = []
            for word, tokens in current_words.items():
                # Check if best_pair exists in tokens
                has_pair = False
                for j in range(len(tokens) - 1):
                    if tokens[j] == best_pair[0] and tokens[j + 1] == best_pair[1]:
                        has_pair = True
                        break
                if has_pair:
                    words_to_update.append(word)

            for word in words_to_update:
                tokens = current_words[word]
                freq = word_freqs[word]

                # Subtract old pair counts for this word
                for j in range(len(tokens) - 1):
                    pair = (tokens[j], tokens[j + 1])
                    pair_counts[pair] -= freq
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]

                # Perform merge
                new_tokens = []
                j = 0
                while j < len(tokens):
                    if j < len(tokens) - 1 and tokens[j] == best_pair[0] and tokens[j + 1] == best_pair[1]:
                        new_tokens.append(new_token)
                        j += 2
                    else:
                        new_tokens.append(tokens[j])
                        j += 1

                current_words[word] = new_tokens

                # Add new pair counts for this word
                for j in range(len(new_tokens) - 1):
                    pair = (new_tokens[j], new_tokens[j + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            # Add merged token to vocab
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.inverse_vocab[idx] = new_token

            if (i + 1) % 500 == 0:
                print(f"  Merge {i+1}/{num_merges} complete. Vocab size: {len(self.vocab)}")

        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

        print(f"Training complete. Final vocab size: {len(self.vocab)}")

    def _encode_word(self, word: Tuple[str, ...]) -> List[int]:
        """Encode a single pre-tokenized word using greedy BPE merging."""
        # Build merge rank lookup for fast access
        if not hasattr(self, '_merge_rank'):
            self._merge_rank: Dict[Tuple[str, str], int] = {
                pair: i for i, pair in enumerate(self.merges)
            }

        tokens = list(word)
        while len(tokens) >= 2:
            # Find the pair with the best (lowest) merge rank
            best_rank = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break  # No more merges possible

            # Merge the best pair
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        ids = []
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            else:
                # Unknown token - encode as individual bytes
                for byte in t.encode("utf-8"):
                    char = BYTES_TO_UNICODE[byte]
                    ids.append(self.vocab.get(char, self.special_tokens["<|endoftext|>"]))
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        word_tokens = self._get_word_tokens(text)
        token_ids = []
        for word in word_tokens:
            token_ids.extend(self._encode_word(tuple(word)))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        parts = []
        for idx in token_ids:
            if idx in self.inverse_special_tokens:
                continue  # Skip special tokens in output
            token = self.inverse_vocab.get(idx, "")
            parts.append(token)
        text = "".join(parts)
        return decode_word_to_bytes(text)

    def encode_single(self, text: str) -> List[int]:
        """Alias for encode."""
        return self.encode(text)

    def save(self, path_prefix: str) -> None:
        """Save tokenizer vocab and merges to files."""
        vocab_path = f"{path_prefix}_vocab.json"
        merges_path = f"{path_prefix}_merges.json"

        # Save vocab as strings
        serializable_vocab = {k: v for k, v in self.vocab.items()}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

        # Save merges
        serializable_merges = [[a, b] for a, b in self.merges]
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

        print(f"Tokenizer saved to {vocab_path} and {merges_path}")

    def load(self, path_prefix: str) -> None:
        """Load tokenizer vocab and merges from files."""
        vocab_path = f"{path_prefix}_vocab.json"
        merges_path = f"{path_prefix}_merges.json"

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.vocab = {k: int(v) for k, v in self.vocab.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        self.merges = [(a, b) for a, b in raw_merges]

        # Update special tokens
        self.special_tokens = {"<|endoftext|>": self.vocab_size}
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        print(f"Tokenizer loaded from {vocab_path} and {merges_path}")
