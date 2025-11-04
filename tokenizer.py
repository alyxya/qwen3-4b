"""
Minimal tokenizer for Qwen3 4B - Built for inference
"""
import json
import regex as re
from huggingface_hub import hf_hub_download

class Tokenizer:
    """Minimal BPE tokenizer for Qwen3 models"""

    def __init__(self):
        """Initialize the tokenizer and load from HuggingFace"""
        # Download tokenizer config from HuggingFace Hub
        repo_id = "Qwen/Qwen3-4B-Instruct-2507"
        tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")

        # Load everything from tokenizer.json
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        # Extract vocabulary
        self.vocab = tokenizer_data["model"]["vocab"]

        # Build reverse mapping (token_id -> token_string) for decoding
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Extract and parse merges
        # Format: merge list has ["token1", "token2"] lists
        merge_list = tokenizer_data["model"]["merges"]
        self.merges = {}
        for i, merge in enumerate(merge_list):
            if len(merge) == 2:
                self.merges[tuple(merge)] = i  # Priority is the index

        # Extract the pre-tokenization regex pattern
        pattern_str = tokenizer_data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.pattern = re.compile(pattern_str)

        # TODO: Extract special tokens
        self.special_tokens = {}

    def encode(self, text: str):
        """
        Convert text to token IDs

        Args:
            text: Input string to tokenize

        Returns:
            List of token IDs
        """
        # Step 1: Pre-tokenize using regex pattern
        chunks = self.pattern.findall(text)

        # Step 2: Convert each chunk to bytes
        all_tokens = []
        for chunk in chunks:
            # Convert string to UTF-8 bytes, then to list of individual byte tokens
            byte_tokens = [bytes([b]) for b in chunk.encode("utf-8")]
            all_tokens.append(byte_tokens)

        # TODO: Step 3: Apply BPE merges
        # TODO: Step 4: Convert tokens to IDs

        return all_tokens  # For now, return list of byte token lists

    def decode(self, token_ids):
        """
        Convert token IDs back to text

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded string
        """
        # TODO: Implement decoding
        pass


if __name__ == "__main__":
    # Test code
    tokenizer = Tokenizer()
    print("Tokenizer created!")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Merges: {len(tokenizer.merges)}")

    # Test encoding (Step 2: convert to bytes)
    test_text = "Hello, world!"
    print(f"\nTest text: {repr(test_text)}")
    result = tokenizer.encode(test_text)
    print(f"Step 2 - Byte tokens per chunk:")
    for i, chunk_tokens in enumerate(result):
        print(f"  Chunk {i}: {chunk_tokens}")
