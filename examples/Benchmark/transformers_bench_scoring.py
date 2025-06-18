import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import hydra


prompt_path = "/home/abouyghf/lamorel/lamorel/examples/Benchmark/prompt.json"



def plot_comparison(lamorel_time, transformers_time):
    x = list(lamorel_time.keys())
    lamorel_y = list(lamorel_time.values())
    transformers_y = list(transformers_time.values())
    plt.figure(figsize=(8, 5))
    plt.plot(x, lamorel_y, label="Lamorel", marker='o')
    plt.plot(x, transformers_y, label="Transformers", marker='x')
    plt.xlabel("Actions number")
    plt.ylabel("Scoring time (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_lamorel_vs_transformers.png")

device2 = torch.device("cuda:0")

def main():

    model_name = "/beegfs/abouyghf/Mistral_7B/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #torch_dtype=torch.float16,
        device_map={"": device2},
        use_safetensors=True
    )

    print("[DEBUG]DOwnloadded !")
    model.eval()
    
    NB_REQUESTS = [1, 2, 4, 8, 16, 32]

    with open(prompt_path, "r") as f:
        config_json = json.load(f)
    prompt = config_json["prompt"]
    possible_actions = config_json["possible_actions"]

    transformers_time = {}

    for i in NB_REQUESTS:
        print("nbr requests")
        print(i)
        actions = possible_actions[:i]
        
        # Transformers scoring
        
        full_texts = [prompt + act for act in actions]

        trnsf_start = time.time()

        
        full_inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=False).to(device2)
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]

        # Tokenize prompt seul (pour connaître la longueur du prompt)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device2)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

            # Décalage pour l’auto-régression
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            # Garder uniquement les positions correspondant à l’action (après le prompt)
            action_labels = shift_labels[:, prompt_len - 1:]  # shape: (batch_size, seq_len_action)
            action_logits = shift_logits[:, prompt_len - 1:, :]  # shape: (batch_size, seq_len_action, vocab_size)

            # Calcul log-probs
            log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)

            # Récupérer la log-proba des tokens corrects
            token_log_probs = log_probs.gather(2, action_labels.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len_action)

            # Somme des log-probas sur la séquence d’action par batch
            total_log_probs = token_log_probs.sum(dim=1)  # (batch_size,)

            # Optionnel : proba en décodant log-prob (si besoin)
            total_probs = total_log_probs.exp()

            trnsf_end = time.time()

            transformers_time[f'{i}'] = trnsf_end - trnsf_start
    	
    print("[DEBUG]:Done! Saving results") 
    # Save timing data
    with open("transformers_timing_summary.json", "w") as f:
        json.dump({
            "NB_REQUESTS": NB_REQUESTS,
            "transformers_times": list(transformers_time.values())}, f, indent=2)
    print("[DEBUG]: saving json")

if __name__ == '__main__':
    main()
