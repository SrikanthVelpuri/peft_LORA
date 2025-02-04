from datasets import load_dataset

def load_qa_dataset():
    return load_dataset("natural_questions")

def preprocess_function(examples, tokenizer):
    inputs = [q["question"] for q in examples["questions"]]
    targets = [a["answer"] for a in examples["answers"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
