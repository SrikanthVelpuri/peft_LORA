# Model Selection

This project uses a pre-trained LLM such as `LLaMA-2-7B` or `GPT-3.5`.

## Loading the Pre-trained Model

To load and use the pre-trained model, follow these steps:

1. Install the necessary libraries:
   ```bash
   pip install transformers
   ```

2. Load the pre-trained model:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "LLaMA-2-7B"  # or "GPT-3.5"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

# Dataset

This project uses the NaturalQuestions Q&A dataset.

## Loading and Preprocessing the Dataset

To load and preprocess the Q&A dataset, follow these steps:

1. Install the necessary libraries:
   ```bash
   pip install datasets
   ```

2. Load the dataset:
   ```python
   from datasets import load_dataset

   dataset = load_dataset("natural_questions")
   ```

3. Preprocess and tokenize the dataset for generative fine-tuning:
   ```python
   def preprocess_function(examples):
       inputs = [q["question"] for q in examples["questions"]]
       targets = [a["answer"] for a in examples["answers"]]
       model_inputs = tokenizer(inputs, max_length=512, truncation=True)

       # Setup the tokenizer for targets
       with tokenizer.as_target_tokenizer():
           labels = tokenizer(targets, max_length=512, truncation=True)

       model_inputs["labels"] = labels["input_ids"]
       return model_inputs

   tokenized_datasets = dataset.map(preprocess_function, batched=True)
   ```
