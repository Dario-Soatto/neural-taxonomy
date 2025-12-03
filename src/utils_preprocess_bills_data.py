import pandas as pd
import spacy
import json
from tqdm import tqdm

# Load spacy for sentence segmentation
nlp = spacy.load("en_core_web_sm")

print("Loading congressional bills data...")
data = []
with open('train.metadata (2).jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} bills")

# Process each bill - same format as other experiments
processed_rows = []
for doc in tqdm(data, desc="Processing bills"):
    bill_id = doc['id']
    
    # Use summary (natural language text, not tokenized)
    text = doc.get('summary', '')
    
    if not text:  # Fallback to tokenized_text if no summary
        text = doc.get('tokenized_text', '')
    
    # Sentence segmentation (same as wiki biographies)
    doc_nlp = nlp(text)
    sentences = [sent.text.strip() for sent in doc_nlp.sents]
    
    # Create row for each sentence (same format as other data)
    for sent_idx, sentence in enumerate(sentences):
        processed_rows.append({
            'doc_index': bill_id,
            'sent_index': sent_idx,
            'sentence_text': sentence
        })

# Create DataFrame
output_df = pd.DataFrame(processed_rows)

print(f"\nProcessed {len(output_df)} sentences from {len(data)} bills")
print(f"Average sentences per bill: {len(output_df) / len(data):.2f}")

# Save in the same format as other experiments
output_df.to_csv('processed_congressional_bills_input.csv', index=False)
output_df.to_json('processed_congressional_bills_input.json', orient='records', lines=True)

print(f"\nSaved to: processed_congressional_bills_input.csv")
print(f"Format matches pipeline requirements: doc_index, sent_index, sentence_text")