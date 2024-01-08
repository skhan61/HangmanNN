from transformers import BertModel, BertTokenizer

from transformers import BertModel, BertTokenizer
from pathlib import Path

# Define your save directory
save_directory = Path("/home/sayem/Desktop/Hangman/pretrained")
save_directory.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist

# Download and save BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model.save_pretrained(str(save_directory))
tokenizer.save_pretrained(str(save_directory))