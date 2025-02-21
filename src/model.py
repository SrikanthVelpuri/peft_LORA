import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_pretrained_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = add_adapters(model)
    return model, tokenizer

def add_adapters(model):
    # Add small extra layers (adapters) to the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            adapter = torch.nn.Linear(module.in_features, module.out_features)
            model.add_module(f"{name}_adapter", adapter)
    return model

def preprocess_function(examples, tokenizer):
    inputs = [q["question"] for q in examples["questions"]]
    targets = [a["answer"] for a in examples["answers"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
