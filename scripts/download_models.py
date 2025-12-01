#!/usr/bin/env python
"""Download and cache sentence-transformer models for Lambda deployment"""

import os
import sys
from sentence_transformers import SentenceTransformer

def main():
    """Download and cache required models"""
    print("Downloading sentence-transformer model...")

    # Set cache directory to /tmp which is writable in Lambda
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/.cache/sentence-transformers'

    # Download the model
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Downloading {model_name}...")

    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Model {model_name} downloaded successfully")

        # Test the model
        test_sentence = "Test sentence"
        embedding = model.encode(test_sentence)
        print(f"✅ Model test successful, embedding shape: {embedding.shape}")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()