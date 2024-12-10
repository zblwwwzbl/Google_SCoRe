import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, MambaForCausalLM
import torch.nn.functional as F
import pandas as pd
import argparse
import yaml

# sc_prompt = "There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution"
GSM_initial_prompt = 'You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "I hope it is correct #### $answer$" '

GSM_self_correcting_prompt = ' There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "I hope it is correct #### $answer$".'

# GSM8K Hugging Face
# from datasets import load_dataset

# ds = load_dataset("openai/gsm8k", "main")

# Load configuration from a YAML file
def load_config(config_file=None):
    if config_file:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "model_name": "state-spaces/mamba-130m-hf",  # Smallest Mamba model
            "learning_rate": 1e-5,
            "epochs_stage_1": 2,
            "epochs_stage_2": 3,
            "beta_kl": 0.1,
            "alpha": 1.5,
            "data_file": "SCoRe_Dataset.csv"
        }
    return config

# Assume inputs are detokenized
def reward_function(y, y_star): 
    '''
    y: nth round including final model output (up to L+1 in the paper)
    y_star: correct answer (oracle response)
    '''
    #
    hash_keyword = "####" # the answer is prepended by #### in GSM8K
    hash_index = y.find(hash_keyword)
    if hash_index == -1:
        response = ""  # Return the original string if "####" is not found
    else:
        response = y[hash_index + len(hash_keyword):].strip()
    if response == y_star:
        return 1.0
    else:
        return 0
    

def first_round_prompt(example):
    return [
        {"role": "user", "content": GSM_initial_prompt+example['question']},
    ]

def second_round_prompt(example, first_round_answer):
    return [
        {"role": "user", "content": GSM_initial_prompt+example['question']},
        {"role": "assistant", "content": f"{first_round_answer}"},
        {"role": "user", "content": GSM_self_correcting_prompt},
    ]

# Stage I: Train initial model to generate first attempt (y1) and prevent mode collapse
def stage_one_initialization(ref_model, model, tokenizer, data, epochs=2, lr=1e-5, beta_kl=0.1):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for example in data:
            # Format input using chat_template
            first_round_conversation = first_round_prompt(example)
            
            # Convert conversation to a single string
            conversation_text = tokenizer.apply_chat_template(first_round_conversation, tokenize=False, add_generation_prompt=True)
            
            inputs1 = tokenizer(conversation_text, return_tensors="pt", padding=True, truncation=True)
            inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
            
            outputs1 = model(**inputs1)

            sample = model.generate(inputs1['input_ids'], max_length=1000, num_return_sequences=1)
            print("START RESPONSE: \n" + tokenizer.decode(sample[0], skip_special_tokens=True) + "\nEND RESPONSE")

            with torch.no_grad():
                ref_outputs = ref_model(**inputs1)  # Reference policy outputs
                ref_probs = F.softmax(ref_outputs.logits, dim=-1)

            second_round_conversation = second_round_prompt(example, tokenizer.decode(outputs1.logits.argmax(dim=-1)[0], skip_special_tokens=True))
            conversation_text2 = tokenizer.apply_chat_template(second_round_conversation, tokenize=False, add_generation_prompt=True)
            inputs2 = tokenizer(conversation_text2, return_tensors="pt", padding=True, truncation=True)
            inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}
            outputs2 = model(**inputs2)

            response2 = tokenizer.decode(outputs2.logits.argmax(dim=-1)[0], skip_special_tokens=True)
            print("START RESPONSE: \n" + response2 + "\nEND RESPONSE")
            
            # Cross-entropy loss (first attempt)
            reward_stage_one = reward_function(response2, example['correct_answer'])
            
            # Log probabilities and apply KL divergence loss
            logits = outputs2.logits
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, ref_probs, reduction='batchmean')
            
            # Total loss combines cross-entropy and scaled KL divergence
            total_loss_value = -reward_stage_one + beta_kl * kl_loss
            
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()
        print(f"Stage I - Epoch {epoch+1}, Loss: {total_loss:.4f}")

def stage_two_training_with_reward_shaping(ref_model, model, tokenizer, data, epochs=3, lr=1e-5, alpha=2.0, beta_kl=0.1):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for example in data:
            # Format input using chat_template
            first_round_conversation = first_round_prompt(example)
            
            # Convert conversation to a single string
            conversation_text = tokenizer.apply_chat_template(first_round_conversation, tokenize=False, add_generation_prompt=True)
            
            inputs1 = tokenizer(conversation_text, return_tensors="pt", padding=True, truncation=True)
            inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
            
            outputs1 = model(**inputs1)

            with torch.no_grad():
                ref_outputs1 = ref_model(**inputs1)  # Reference policy outputs
                ref_probs1 = F.softmax(ref_outputs1.logits, dim=-1)

            second_round_conversation = second_round_prompt(example, tokenizer.decode(outputs1.logits.argmax(dim=-1)[0], skip_special_tokens=True))
            conversation_text2 = tokenizer.apply_chat_template(second_round_conversation, tokenize=False, add_generation_prompt=True)
            inputs2 = tokenizer(conversation_text2, return_tensors="pt", padding=True, truncation=True)
            inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}
            outputs2 = model(**inputs2)

            with torch.no_grad():
                ref_outputs2 = ref_model(**inputs2)
                ref_probs2 = F.softmax(ref_outputs2.logits, dim=-1)
            
            # Cross-entropy loss (first attempt)
            response1 = tokenizer.decode(outputs1.logits.argmax(dim=-1)[0], skip_special_tokens=True)
            response2 = tokenizer.decode(outputs2.logits.argmax(dim=-1)[0], skip_special_tokens=True)
            reward_round_1 = reward_function(response1, example['correct_answer'])
            reward_round_2 = reward_function(response2, example['correct_answer'])
            b = alpha*(reward_round_2 - reward_round_1)
            
            # Log probabilities and apply KL divergence loss
            logits1 = outputs1.logits
            log_probs1 = F.log_softmax(logits1, dim=-1)
            kl_loss1 = F.kl_div(log_probs1, ref_probs1, reduction='batchmean')

            logits2 = outputs2.logits
            log_probs2 = F.log_softmax(logits2, dim=-1)
            kl_loss2 = F.kl_div(log_probs2, ref_probs2, reduction='batchmean')
            
            total_loss_value = -b - reward_round_1 + beta_kl * (kl_loss1 + kl_loss2)
            
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()
        print(f"Stage I - Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Main function to run the training process
def main(config_file=None):
    config = load_config(config_file)

    # Load model and tokenizer
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    model = MambaForCausalLM.from_pretrained(model_name, 
                                      device_map="auto", 
                                      attn_implementation='eager')

    ref_model = MambaForCausalLM.from_pretrained(model_name, 
                                      device_map="auto", 
                                      attn_implementation='eager')

    # Load the dataset
    data_file_path = config["data_file"]
    df = pd.read_csv(data_file_path)

    # Prepare the data for Stage I and Stage II
    data_stage = df[["question", "original_answer", "correct_answer"]].to_dict(orient="records")

    # Stage I training (Initialization)
    stage_one_initialization(
        ref_model, model, tokenizer, data_stage, 
        epochs=config["epochs_stage_1"], 
        lr=config["learning_rate"], 
        beta_kl=config["beta_kl"]
    )

    # Stage II training (Self-correction)
    stage_two_training_with_reward_shaping(
        ref_model, model, tokenizer, data_stage, 
        epochs=config["epochs_stage_2"], 
        lr=config["learning_rate"], 
        alpha=config["alpha"]
    )

    # Save the trained model
    model.save_pretrained("./trained_self_correcting_model")
    tokenizer.save_pretrained("./trained_self_correcting_model")

# Run the main function (can use a config file or default)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    args = parser.parse_args()

    main(args.config)