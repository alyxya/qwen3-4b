"""
Minimal tokenizer for Qwen3 4B - Built for inference
"""

class Tokenizer:
    """Minimal BPE tokenizer for Qwen3 models"""

    def __init__(self):
        """Initialize the tokenizer"""
        # We'll store the vocabulary here (token_string -> token_id)
        self.vocab = {}

        # Reverse mapping (token_id -> token_string) for decoding
        self.id_to_token = {}

        # Special tokens used by Qwen3
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
