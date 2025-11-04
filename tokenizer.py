"""
Minimal tokenizer for Qwen3 4B - Built for inference
"""
import json
from huggingface_hub import hf_hub_download

class Tokenizer:
    """Minimal BPE tokenizer for Qwen3 models"""

    def __init__(self):
        """Initialize the tokenizer and load vocab from HuggingFace"""
        # Download vocab and merges from HuggingFace Hub
        repo_id = "Qwen/Qwen3-4B-Instruct-2507"

        vocab_path = hf_hub_download(repo_id, "vocab.json")
        merges_path = hf_hub_download(repo_id, "merges.txt")

        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # Load merges
        with open(merges_path, 'r', encoding='utf-8') as f:
            merge_lines = f.read().strip().split('\n')

        # Build reverse mapping (token_id -> token_string) for decoding
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # TODO: Parse merges
        self.merges = merge_lines

        # TODO: Extract special tokens from vocab
        self.special_tokens = {}

    def encode(self, text: str):
        """
        Convert text to token IDs

        Args:
            text: Input string to tokenize

        Returns:
            List of token IDs
        """
        # TODO: Implement encoding
        pass

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


if __name__ == '__main__':
    # Test code
    tokenizer = Tokenizer()
    print("Tokenizer created!")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Merges: {len(tokenizer.merges)}")

    # Show some sample tokens
    print("\nFirst 5 vocab entries:")
    for i, (token, token_id) in enumerate(list(tokenizer.vocab.items())[:5]):
        print(f"  {repr(token)} -> {token_id}")

    # Show first few merges
    print("\nFirst 3 merge rules:")
    for merge in tokenizer.merges[:3]:
        print(f"  {merge}")
