# transformer-finetuning

`python3 main.py --model meta-llama/Llama-3.2-3B --epochs 20 --batch_size 1 --output_dir ./outputs/3B`

`python3 invoke.py --model  ./outputs/3B --prompt "Wir" --max_length 256`