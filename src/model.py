import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_pretrained_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def preprocess_function(examples, tokenizer):
    inputs = [q["question"] for q in examples["questions"]]
    targets = [a["answer"] for a in examples["answers"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
