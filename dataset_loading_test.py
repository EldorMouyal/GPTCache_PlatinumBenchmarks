from datasets import load_dataset

BENCH_NAME = "gsm8k"
dataset = load_dataset("madrylab/platinum-bench", name=BENCH_NAME, split="test")

print(f"Loaded {len(dataset)} prompts from {BENCH_NAME}")
print(dataset[0]["question"])
