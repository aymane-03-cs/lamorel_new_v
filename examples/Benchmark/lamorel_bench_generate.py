from lamorel import Caller, lamorel_init
lamorel_init()
import torch.nn.functional as F
import torch
import time
import json
import hydra


prompt_path = "/home/abouyghf/lamorel/lamorel/examples/Benchmark/prompt.json"


def nvidia_smi(i):
    print("--------------------GPU info--------------------------------")
    print(f"=== GPU {i} ({torch.cuda.get_device_name(i)}) ===")
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(i)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(i)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(i)/1024/1024/1024))

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    lm_server = Caller(config_args.lamorel_args)
    print("[DEBUG] lm_server Ready (Caller Instace ready) !" )
    nvidia_smi(0)


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
        if i == 1: 
            print("before first call")
            nvidia_smi(0)

        # Lamorel scoring
        lm_start = time.time()
        responses = lm_server.generate(contexts=[prompt + act for act in actions ], max_length=500)
        lm_end = time.time()
        if i == 1:
            print("[DEBUG] first score done,check your GPU Memory Usage")
            nvidia_smi(0)
        lamorel_time[f'{i}'] = lm_end - lm_start
        print("before transformers")
        # Transformers scoring
        
    	
    print("[DEBUG]:Done! Saving results") 
    # Save timing data
    with open("lamorel_timing_generate_summary.json", "w") as f:
        json.dump({
            "NB_REQUESTS": NB_REQUESTS,
            "lamorel_times": list(lamorel_time.values())}, f, indent=2)
    print("[DEBUG]: saving json")	
    lm_server.close()

if __name__ == '__main__':
    main()
