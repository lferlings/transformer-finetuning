trl sft \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --dataset_name stanfordnlp/imdb \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT

python3 trl/examples/scripts/sft.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_4bit \
    --use_peft \
    --gradient_accumulation_steps 2

python -m llama_recipes.finetuning \
    --use_peft --peft_method lora --quantization \
    --model_name meta-llama/Llama-3.2-3B \
    --output_dir outputs \
    --batch_size_training 2 --gradient_accumulation_steps 2