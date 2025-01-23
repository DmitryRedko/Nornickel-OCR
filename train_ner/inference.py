from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Reload the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("../rubert_ner_model_splitted")
model = BertForTokenClassification.from_pretrained("../rubert_ner_model_splitted")

# Define the label-to-id mapping (optional if already in `model.config.id2label`)
id2label = {0: "B-LINK", 1: "I-LINK",  2: "O"}

def predict_entities(text):
    # Split the text into words (pre-tokenization)
    words = text.split()

    # Tokenize the words list using the tokenizer
    inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1)

    # Align predictions with words
    word_ids = inputs.word_ids()  # Map tokens to words
    result = []

    previous_word_idx = None
    for word_id, label_id in zip(word_ids, predictions[0]):
        if word_id is None:  # Skip special tokens
            continue
        if word_id != previous_word_idx:  # New word
            word = words[word_id]
            label = id2label[label_id.item()]  # Convert label ID to label name
            result.append((word, label))
        previous_word_idx = word_id

    return result


# Example usag  e
test_text = "В соответствии с п.1.6 СНиП 3.01.04-87"
predictions = predict_entities(test_text)
print(predictions)


