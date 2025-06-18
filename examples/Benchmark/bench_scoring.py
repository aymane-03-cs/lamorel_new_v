from lamorel import Caller, lamorel_init
lamorel_init()
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

device2 = torch.device("cuda:1")


@hydra.main(config_path='config', config_name='config')
def main(config_args):
    lm_server = Caller(config_args.lamorel_args)

    
    model_name = "/beegfs/abouyghf/Mistral_7B/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device2)
    print("[DEBUG]DOwnloadded !")
    model.eval()
    
    NB_REQUESTS = [1, 2, 4, 8, 16, 32, 42]

    with open(prompt_path, "r") as f:
        config_json = json.load(f)
    prompt = config_json["prompt"]
    possible_actions = config_json["possible_actions"]

    lamorel_time = {}
    transformers_time = {}

    for i in NB_REQUESTS:
        print("nbr requests")
        print(i)
        actions = possible_actions[:i]
        
        # Lamorel scoring
        lm_start = time.time()
        raw_llm_scores = lm_server.score([prompt], [actions])[0].tolist()
        lm_end = time.time()
        lamorel_time[f'{i}'] = lm_end - lm_start
        print("before transformers")
        # Transformers scoring
        with torch.no_grad():
            trnsf_start = time.time()
            for act in actions:
                #Concat prompt and action
                
                full = prompt + act
                # Tokenisation du prompt seul
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device2)
                prompt_len = prompt_ids.shape[1]

                # Tokenisation du prompt + action
                full_inputs = tokenizer(full, return_tensors="pt")
                input_ids = full_inputs["input_ids"].to(device2)
                attention_mask = full_inputs["attention_mask"].to(device2)

    
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Décalage pour l'auto-régression
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                # On ne garde que les positions correspondant à l'action
                action_labels = shift_labels[:, prompt_len - 1:]  # inclut le 1er token de l'action
                action_logits = shift_logits[:, prompt_len - 1:, :]

                # Calcul de la log-probabilité
                log_probs = F.log_softmax(action_logits, dim=-1)
                token_log_probs = log_probs.gather(2, action_labels.unsqueeze(-1)).squeeze(-1)
                total_log_prob = token_log_probs.sum().item()

                # Optionnel : probabilité (exp du log)
                total_prob = torch.exp(token_log_probs.sum()).item()
                
            trnsf_end = time.time()
            transformers_time[f'{i}'] = trnsf_end - trnsf_start
    	
    print("[DEBUG]:Done! Saving results") 
    # Save timing data
    with open("lamorel_timing_summary.json", "w") as f:
        json.dump({
            "NB_REQUESTS": NB_REQUESTS,
            "lamorel_times": list(lamorel_time.values()),
            "transformers_times": list(transformers_time.values())}, f, indent=2)
    print("[DEBUG]: saving plot")
    plot_comparison(lamorel_time, transformers_time)	
    lm_server.close()

if __name__ == '__main__':
    main()
