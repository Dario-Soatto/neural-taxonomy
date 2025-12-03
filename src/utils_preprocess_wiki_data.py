import pandas as pd
import spacy
import json
import sys
from tqdm import tqdm
import re

# Load spacy for sentence segmentation (only for natural text)
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Configuration for each dataset
DATASET_CONFIG = {
    'train.metadata.jsonl': {
        'text_field': 'tokenized_text',
        'name': 'wikitext',
        'is_tokenized': True,  # No punctuation, use chunking
    },
    'train.metadata (1).jsonl': {
        'text_field': 'text',
        'name': 'wiki_biographies',
        'is_tokenized': False,  # Has punctuation, use spaCy
    },
    'train.metadata (2).jsonl': {
        'text_field': 'summary',
        'name': 'congressional_bills',
        'is_tokenized': False,  # Has punctuation, use spaCy
    }
}

def chunk_by_length(text, words_per_chunk=50):
    """
    Split tokenized text (no punctuation) into chunks by word count.
    
    Args:
        text: String of tokenized text
        words_per_chunk: Target number of words per chunk
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
    
    return chunks

def split_by_sentences(text, nlp):
    """
    Split natural text (with punctuation) into sentences using spaCy.
    
    Args:
        text: String of natural text
        nlp: spaCy language model
    
    Returns:
        List of sentences
    """
    doc_nlp = nlp(text)
    sentences = [sent.text.strip() for sent in doc_nlp.sents]
    return sentences

def detect_if_tokenized(text):
    """
    Auto-detect if text is tokenized (no punctuation) or natural.
    
    Returns:
        True if tokenized, False if natural
    """
    # Count punctuation marks
    punctuation_count = sum(1 for char in text if char in '.!?,;:')
    # If very few punctuation marks relative to length, likely tokenized
    if len(text) > 100:  # Only check if text is long enough
        punctuation_ratio = punctuation_count / len(text)
        return punctuation_ratio < 0.005  # Less than 0.5% punctuation
    return False

def process_document(doc, config, nlp, words_per_chunk=50):
    """
    Process a single document based on whether it's tokenized or natural text.
    
    Args:
        doc: Document dictionary
        config: Dataset configuration
        nlp: spaCy model
        words_per_chunk: Words per chunk for tokenized text
    
    Returns:
        List of processed rows (dict with doc_index, sent_index, sentence_text)
    """
    doc_id = doc['id']
    
    # Get text from appropriate field
    text = doc.get(config['text_field'], '')
    
    if not text:
        return []
    
    # Determine processing method
    is_tokenized = config.get('is_tokenized', detect_if_tokenized(text))
    
    # Process based on text type
    if is_tokenized:
        # Use length-based chunking for tokenized text
        chunks = chunk_by_length(text, words_per_chunk=words_per_chunk)
    else:
        # Use spaCy sentence splitting for natural text
        chunks = split_by_sentences(text, nlp)
    
    # Create rows
    rows = []
    for chunk_idx, chunk_text in enumerate(chunks):
        rows.append({
            'doc_index': doc_id,
            'sent_index': chunk_idx,
            'sentence_text': chunk_text
        })
    
    return rows

def main():
    # Get filename from command line or use default
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename> [words_per_chunk]")
        print(f"\nAvailable datasets:")
        for filename, config in DATASET_CONFIG.items():
            text_type = "tokenized (chunking)" if config['is_tokenized'] else "natural (spaCy)"
            print(f"  - {filename:35s} ({config['name']}) [{text_type}]")
        print(f"\nOptional: Specify words_per_chunk for tokenized text (default: 50)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    words_per_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    # Get configuration
    if input_file not in DATASET_CONFIG:
        print(f"Warning: Unknown dataset. Using auto-detection.")
        config = {
            'text_field': 'text',  # Try common fields
            'name': 'custom',
            'is_tokenized': None  # Will auto-detect
        }
    else:
        config = DATASET_CONFIG[input_file]
    
    # Load data
    print(f"\nLoading {config['name']} data from: {input_file}")
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} documents")
    
    # Sample check for text type
    if data:
        sample_text = data[0].get(config['text_field'], '')
        is_tokenized = config.get('is_tokenized', detect_if_tokenized(sample_text))
        method = "length-based chunking" if is_tokenized else "spaCy sentence splitting"
        print(f"Text type: {'tokenized' if is_tokenized else 'natural'}")
        print(f"Processing method: {method}")
        if is_tokenized:
            print(f"Words per chunk: {words_per_chunk}")
    
    # Process all documents
    processed_rows = []
    for doc in tqdm(data, desc="Processing documents"):
        rows = process_document(doc, config, nlp, words_per_chunk=words_per_chunk)
        processed_rows.extend(rows)
    
    # Create DataFrame
    output_df = pd.DataFrame(processed_rows)
    
    # Generate output filenames
    output_csv = f"processed_{config['name']}_input.csv"
    output_json = f"processed_{config['name']}_input.json"
    
    # Save
    output_df.to_csv(output_csv, index=False)
    output_df.to_json(output_json, orient='records', lines=True)
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total documents: {len(data)}")
    print(f"Total sentences/chunks: {len(output_df)}")
    print(f"Average per document: {len(output_df) / len(data):.2f}")
    print(f"\nOutput files:")
    print(f"  - {output_csv}")
    print(f"  - {output_json}")
    print(f"\nFormat: doc_index, sent_index, sentence_text")
    print(f"Ready for Step 1 of the pipeline!")

if __name__ == "__main__":
    main()