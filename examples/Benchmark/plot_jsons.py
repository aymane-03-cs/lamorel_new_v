import matplotlib.pyplot as plt
import json

with open("lamorel_timing_summary.json", "r") as f_lamorel:
    lamorel_time = json.load(f_lamorel)

with open("transformers_timing_summary.json", "r") as f_transformers:
    transformers_time = json.load(f_transformers)


def plot_comparison(lamorel_time, transformers_time, fig_name="plot_lamorel_vs_transformers"):
    x = list(lamorel_time["NB_REQUESTS"])
    lamorel_y = list(lamorel_time["lamorel_times"])
    transformers_y = list(transformers_time["transformers_times"])
    plt.figure(figsize=(8, 5))
    plt.plot(x[1:], lamorel_y[1:], label="Lamorel", marker='o')
    plt.plot(x[1:], transformers_y[1:], label="Transformers", marker='x')
    plt.xlabel("Actions number")
    plt.ylabel("Scoring time (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_name}.png")


plot_comparison(lamorel_time, transformers_time,argv[1])
