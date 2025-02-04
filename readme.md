# Model Selection

This project uses a pre-trained LLM such as `LLaMA-2-7B` or `GPT-3.5`.

## Installation

To install the necessary libraries, run the following command:
```bash
pip install transformers datasets torch psutil
```

## Loading the Pre-trained Model

To load and use the pre-trained model, follow these steps:

1. Load the pre-trained model:
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

1. Load the dataset:
   ```python
   from datasets import load_dataset

   dataset = load_dataset("natural_questions")
   ```

2. Preprocess and tokenize the dataset for generative fine-tuning:
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

# Fine-tuning

This project includes a script to fine-tune the model and log metrics such as training loss, validation loss, and accuracy for each combination of `r`, `alpha`, and `dropout`.

## Running the Fine-tuning Script

To run the fine-tuning script and log the metrics, follow these steps:

1. Run the fine-tuning script:
   ```python
   from src.fine_tune import fine_tune_model
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from datasets import load_dataset

   model_name = "LLaMA-2-7B"  # or "GPT-3.5"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   dataset = load_dataset("natural_questions")

   r_values = [1, 2, 3]
   alpha_values = [0.1, 0.2, 0.3]
   dropout_values = [0.1, 0.2, 0.3]

   for r in r_values:
       for alpha in alpha_values:
           for dropout in dropout_values:
               fine_tune_model(model, tokenizer, dataset, r, alpha, dropout)
   ```

# Evaluation

This project includes an evaluation script to measure various metrics such as perplexity, BLEU/ROUGE scores, factfulness, memory consumption, and inference speed.

## Running the Evaluation Script

To run the evaluation script and measure the metrics, follow these steps:

1. Run the evaluation script:
   ```python
   from src.evaluation import evaluate_perplexity, calculate_bleu_rouge, evaluate_factfulness, measure_memory_usage, measure_inference_speed
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from datasets import load_dataset

   model_name = "LLaMA-2-7B"  # or "GPT-3.5"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   dataset = load_dataset("natural_questions")

   perplexity = evaluate_perplexity(model, dataset, tokenizer)
   bleu_score, rouge_score = calculate_bleu_rouge(model, dataset, tokenizer)
   factfulness_percentage = evaluate_factfulness(model, dataset, tokenizer)
   memory_usage_gb = measure_memory_usage()
   inference_speed = measure_inference_speed(model, dataset, tokenizer)

   print(f"Perplexity: {perplexity}")
   print(f"BLEU Score: {bleu_score}")
   print(f"ROUGE Score: {rouge_score}")
   print(f"Factfulness: {factfulness_percentage}%")
   print(f"Memory Usage: {memory_usage_gb} GB")
   print(f"Inference Speed: {inference_speed} seconds per response")
   ```

## Example Graphs

Below are example graphs for each parameter:

### Perplexity

![Perplexity](example_perplexity.png)

### BLEU/ROUGE Scores

![BLEU/ROUGE Scores](example_bleu_rouge.png)

### Factfulness

![Factfulness](example_factfulness.png)

### Memory Consumption

![Memory Consumption](example_memory_consumption.png)

### Inference Speed

![Inference Speed](example_inference_speed.png)

## Generating Graphs

To generate your own graphs for each parameter, follow these instructions:

### Perplexity

1. Log perplexity during training.
2. Use a plotting library such as `matplotlib` to create a line graph of perplexity over training epochs.

### BLEU/ROUGE Scores

1. Calculate BLEU/ROUGE scores for model responses.
2. Use a plotting library such as `matplotlib` to create bar charts comparing BLEU/ROUGE scores for different models or configurations.

### Factfulness

1. Evaluate the factfulness of model responses.
2. Use a plotting library such as `matplotlib` to create a pie chart or bar graph showing the percentage of factually correct answers.

### Memory Consumption

1. Monitor GPU memory usage during inference.
2. Use a plotting library such as `matplotlib` to create a bar chart comparing GPU memory usage for different models or configurations.

### Inference Speed

1. Measure the average inference time per response.
2. Use a plotting library such as `matplotlib` to create a line graph of average inference time per response for different models or configurations.
