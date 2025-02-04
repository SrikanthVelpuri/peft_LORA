import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
import time
import psutil
import os

def evaluate_perplexity(model, dataset, tokenizer):
    model.eval()
    total_loss = 0
    for batch in dataset:
        inputs = tokenizer(batch['input'], return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        total_loss += outputs.loss.item()
    perplexity = torch.exp(torch.tensor(total_loss / len(dataset)))
    return perplexity.item()

def calculate_bleu_rouge(model, dataset, tokenizer):
    bleu = load_metric('bleu')
    rouge = load_metric('rouge')
    model.eval()
    for batch in dataset:
        inputs = tokenizer(batch['input'], return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = [batch['reference']]
        bleu.add_batch(predictions=predictions, references=references)
        rouge.add_batch(predictions=predictions, references=references)
    bleu_score = bleu.compute()
    rouge_score = rouge.compute()
    return bleu_score, rouge_score

def evaluate_factfulness(model, dataset, tokenizer):
    factfulness = 0
    model.eval()
    for batch in dataset:
        inputs = tokenizer(batch['input'], return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prediction == batch['reference']:
            factfulness += 1
    factfulness_percentage = (factfulness / len(dataset)) * 100
    return factfulness_percentage

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024 ** 3)
    return memory_usage_gb

def measure_inference_speed(model, dataset, tokenizer):
    model.eval()
    total_time = 0
    for batch in dataset:
        inputs = tokenizer(batch['input'], return_tensors='pt')
        start_time = time.time()
        with torch.no_grad():
            model.generate(**inputs)
        end_time = time.time()
        total_time += (end_time - start_time)
    average_inference_time = total_time / len(dataset)
    return average_inference_time

def log_training_loss(trainer):
    return trainer.state.log_history[-1]['loss']

def log_validation_loss(trainer):
    return trainer.evaluate()['eval_loss']

def log_accuracy(trainer):
    return trainer.evaluate()['eval_accuracy']
