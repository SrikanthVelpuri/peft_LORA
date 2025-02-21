from datasets import load_dataset

def load_qa_dataset():
    return load_dataset("natural_questions")

def collect_datasets_for_use_cases(use_cases):
    datasets = {}
    for use_case in use_cases:
        if use_case == "customer support":
            datasets[use_case] = load_dataset("customer_support_dataset")
        elif use_case == "content generation":
            datasets[use_case] = load_dataset("content_generation_dataset")
        elif use_case == "coding assistance":
            datasets[use_case] = load_dataset("coding_assistance_dataset")
    return datasets

def preprocess_function(examples, tokenizer, use_case=None):
    inputs = [q["question"] for q in examples["questions"]]
    targets = [a["answer"] for a in examples["answers"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["use_case"] = use_case
    return model_inputs

def tokenize_dataset(dataset, tokenizer, use_case=None):
    return dataset.map(lambda examples: preprocess_function(examples, tokenizer, use_case), batched=True)
