from lamorel import Caller, lamorel_init
lamorel_init()
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import time
import json
import hydra


prompt_path = "/home/abouyghf/lamorel/lamorel/examples/Benchmark/prompt.json"


@hydra.main(config_path='config', config_name='config')
def main(config_args):
    lm_server = Caller(config_args.lamorel_args)
    print("[DEBUG] lm_server Ready (Caller Instace ready) !" )

    NB_REQUESTS = [1, 2, 4, 8, 16, 32,42]

    with open(prompt_path, "r") as f:
        config_json = json.load(f)
    prompt = config_json["prompt"]
    possible_actions = config_json["possible_actions"]

    lamorel_time = {}
    print("[DEBUG]:starting scores")
    for i in NB_REQUESTS:

        print("nbr requests")
        print(i)
        actions = possible_actions[:i]

        # Lamorel scoring
        lm_start = time.time()
        raw_llm_scores = lm_server.score([prompt], [actions])[0].tolist()
        lm_end = time.time()
        lamorel_time[f'{i}'] = lm_end - lm_start
        
        
    	
    print("[DEBUG]:Done! Saving results") 
    # Save timing data
    with open("lamorel_timing_summary.json", "w") as f:
        json.dump({
            "NB_REQUESTS": NB_REQUESTS,
            "lamorel_times": list(lamorel_time.values())}, f, indent=2)
    print("[DEBUG]: saving json")	
    lm_server.close()

if __name__ == '__main__':
    main()
