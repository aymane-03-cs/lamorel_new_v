PROJECT_PATH=/home/abouyghf/lamorel/lamorel

python -m lamorel_launcher.launch \
--config-path $PROJECT_PATH/examples/Benchmark/ \
--config-name bench_config.yaml \
rl_script_args.path=$PROJECT_PATH/examples/Benchmark/lamorel_bench_generate.py \
lamorel_args.accelerate_args.machine_rank=$1 \
lamorel_args.llm_args.model_path=/beegfs/abouyghf/Mistral_7B
