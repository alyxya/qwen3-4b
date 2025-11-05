"""
Minimal tokenizer for Qwen3 4B - Built for inference
"""

import json
import regex as re
from huggingface_hub import hf_hub_download


class Tokenizer:
    """Minimal BPE tokenizer for Qwen3 models"""

    def __init__(self) -> None:
        """Initialize the tokenizer and load from HuggingFace"""
        # Download tokenizer config from HuggingFace Hub
        repo_id: str = "Qwen/Qwen3-4B-Instruct-2507"
        tokenizer_path: str = hf_hub_download(repo_id, "tokenizer.json")

        # Load everything from tokenizer.json
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data: dict = json.load(f)

        # Extract vocabulary
        self.vocab: dict[str, int] = tokenizer_data["model"]["vocab"]

        # Build reverse mapping (token_id -> token_string) for decoding
        self.id_to_token: dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Extract and parse merges
        # Format: merge list has ["token1", "token2"] lists
        merge_list: list[list[str]] = tokenizer_data["model"]["merges"]
        self.merges: dict[tuple[str, str], int] = {}
        for i, merge in enumerate(merge_list):
            if len(merge) == 2:
                self.merges[tuple(merge)] = i  # Priority is the index

        # Extract the pre-tokenization regex pattern
        pattern_str: str = tokenizer_data["pre_tokenizer"]["pretokenizers"][0][
            "pattern"
        ]["Regex"]
        self.pattern: re.Pattern = re.compile(pattern_str)

        # Special tokens (e.g., <|endoftext|>, <|im_start|>, <|im_end|>) available in
        # tokenizer_data["added_tokens"] but not needed for basic inference
        self.special_tokens: dict[str, int] = {}

        # Build byte-to-unicode mapping (GPT-2 style byte-level BPE)
        self.byte_encoder: dict[int, str] = self._build_byte_encoder()
        self.byte_decoder: dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

    def _build_byte_encoder(self) -> dict[int, str]:
        """
        Build mapping from bytes to unicode characters for byte-level BPE
        This avoids having to deal with control characters and ensures
        all bytes map to printable characters.
        """
        # Printable ASCII (not including control chars)
        bs: list[int] = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs: list[int] = bs[:]
        n: int = 0
        # Map remaining bytes to unused unicode chars
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def _apply_bpe(self, word: list[str]) -> list[str]:
        """
        Apply BPE merges to a list of string tokens

        Args:
            word: List of strings (e.g., ['H', 'e', 'l', 'l', 'o'])

        Returns:
            List of merged string tokens
        """
        # Keep merging until no more merges are possible
        while len(word) > 1:
            # Find all adjacent pairs
            pairs: list[tuple[str, str]] = [
                (word[i], word[i + 1]) for i in range(len(word) - 1)
            ]

            # Find the pair with highest priority (lowest index in merges)
            best_pair: tuple[str, str] = min(
                pairs, key=lambda pair: self.merges.get(tuple(pair), float("inf"))
            )

            # If no pairs can be merged, we're done
            if tuple(best_pair) not in self.merges:
                break

            # Merge all instances of the best pair
            first, second = best_pair
            new_word: list[str] = []
            i: int = 0
            while i < len(word):
                # If we find the pair, merge it
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word

        return word

    def encode(self, text: str) -> list[int]:
        """
        Convert text to token IDs

        Args:
            text: Input string to tokenize

        Returns:
            List of token IDs
        """
        # Step 1: Pre-tokenize using regex pattern
        chunks: list[str] = self.pattern.findall(text)

        # Step 2: Convert each chunk to list of individual byte characters
        all_tokens: list[list[str]] = []
        for chunk in chunks:
            # Convert to UTF-8 bytes, then use byte encoder mapping
            byte_chars: list[str] = [
                self.byte_encoder[b] for b in chunk.encode("utf-8")
            ]
            all_tokens.append(byte_chars)

        # Step 3: Apply BPE merges to each chunk
        merged_tokens: list[list[str]] = []
        for byte_chars in all_tokens:
            bpe_tokens: list[str] = self._apply_bpe(byte_chars)
            merged_tokens.append(bpe_tokens)

        # Step 4: Convert tokens to IDs
        token_ids: list[int] = []
        for chunk_tokens in merged_tokens:
            for token in chunk_tokens:
                token_ids.append(self.vocab[token])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to text

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded string
        """
        # Step 1: Convert IDs to tokens (strings)
        tokens: list[str] = [self.id_to_token[token_id] for token_id in token_ids]

        # Step 2: Join tokens into a single string
        text: str = "".join(tokens)

        # Step 3: Convert from byte-level characters back to UTF-8 bytes
        byte_array: bytearray = bytearray([self.byte_decoder[c] for c in text])

        # Step 4: Decode UTF-8 bytes to string
        return byte_array.decode("utf-8", errors="replace")
