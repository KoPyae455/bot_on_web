from datasets import load_dataset

dataset = load_dataset('json', data_files='customer_support.jsonl', split='train')

# Optional: Print sample
print(dataset[0])
