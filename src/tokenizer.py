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

        # Load special tokens from added_tokens
        self.special_tokens: dict[str, int] = {}
        if "added_tokens" in tokenizer_data:
            for token_data in tokenizer_data["added_tokens"]:
                self.special_tokens[token_data["content"]] = token_data["id"]

        # Build reverse mapping (token_id -> token_string) for decoding
        # Merge both regular vocab and special tokens into one mapping
        self.id_to_token: dict[int, str] = {
            **{v: k for k, v in self.vocab.items()},
            **{v: k for k, v in self.special_tokens.items()},
        }

        # Build byte-to-unicode mapping (GPT-2 style byte-level BPE)
        self.byte_encoder: dict[int, str] = self._build_byte_encoder()
        self.byte_decoder: dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

        # Create regex pattern for splitting text with special tokens
        special_tokens_pattern = "|".join(
            re.escape(token) for token in self.special_tokens.keys()
        )
        self.special_tokens_regex: re.Pattern = re.compile(
            f"({special_tokens_pattern})"
        )

    @property
    def im_start(self) -> str:
        """Get the <|im_start|> token string"""
        return "<|im_start|>"

    @property
    def im_end(self) -> str:
        """Get the <|im_end|> token string"""
        return "<|im_end|>"

    @property
    def endoftext(self) -> str:
        """Get the <|endoftext|> token string"""
        return "<|endoftext|>"

    @property
    def im_start_id(self) -> int:
        """Get the <|im_start|> token ID"""
        return self.special_tokens.get("<|im_start|>", -1)

    @property
    def im_end_id(self) -> int:
        """Get the <|im_end|> token ID"""
        return self.special_tokens.get("<|im_end|>", -1)

    @property
    def endoftext_id(self) -> int:
        """Get the <|endoftext|> token ID"""
        return self.special_tokens.get("<|endoftext|>", -1)

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
        # Split text by special tokens
        parts = self.special_tokens_regex.split(text)

        token_ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                # Add special token ID directly
                token_ids.append(self.special_tokens[part])
            else:
                # Encode regular text with BPE
                # Step 1: Pre-tokenize using regex pattern
                chunks: list[str] = self.pattern.findall(part)

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
        # Convert IDs to token strings
        tokens: list[str] = [self.id_to_token[token_id] for token_id in token_ids]

        # Join tokens into a single string
        text: str = "".join(tokens)

        # Convert from byte-level characters back to UTF-8 bytes
        byte_array: bytearray = bytearray([self.byte_decoder[c] for c in text])

        # Decode UTF-8 bytes to string
        return byte_array.decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> list[int] | str:
        """
        Apply chat template to format conversation for Qwen3 models

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     role can be "system", "user", or "assistant"
            add_generation_prompt: If True, add <|im_start|>assistant to prompt
                                  the model to generate a response
            tokenize: If True, return token IDs. If False, return formatted string

        Returns:
            List of token IDs if tokenize=True, otherwise formatted string

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"},
            ... ]
            >>> token_ids = tokenizer.apply_chat_template(messages)
        """
        formatted_parts: list[str] = []

        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            # Add message with proper formatting
            formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # Add generation prompt if requested
        if add_generation_prompt:
            formatted_parts.append("<|im_start|>assistant\n")

        formatted_text = "".join(formatted_parts)

        if tokenize:
            return self.encode(formatted_text)
        return formatted_text
