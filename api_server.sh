# hostname -i
# 4: http://172.22.8.7:8000/v1
# 5: http://172.22.8.16:8000/v1


# finetuned models
# llama3-8b-test-3
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_0/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# llama3-8b-test-25k
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_25000_mini/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# llama3-8b-test-50k
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_50000_mini/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# llama3-8b-test-full
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_100000/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8



# gemma-1.1-7b-it-test-10k
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/gemma-1.1-7b-it_10000/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# gemma-1.1-7b-it-test-50k
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/gemma-1.1-7b-it_50000/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# gemma-1.1-7b-it-test-full
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/gemma-1.1-7b-it/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# mistral-7b-v0.3-test-10k
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/mistral-7b-instruct-v0.3_10000/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-7b-v0.3-test-50k
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/mistral-7b-instruct-v0.3_50000/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-7b-v0.3-test-full
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/mistral-7b-instruct-v0.3_10000/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8







# unsloth base models
# unsloth-llama3-8b
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/unsloth_models/llama-3-8b-Instruct/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# unsloth-gemma-1.1-7b-it
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/unsloth_models/gemma-1.1-7b-it/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# unsloth-mistral-7b-v0.2
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/unsloth_models/mistral-7b-instruct-v0.2/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# unsloth-mistral-7b-v0.3
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/unsloth_models/mistral-7b-instruct-v0.3/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8



# llama3-8B
python -m vllm.entrypoints.openai.api_server --model models/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-70B
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/huggingface_models/Meta-Llama-3-70B-Instruct/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8






# gemma-1.1-2b
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/huggingface_models/gemma-1.1-2b-it/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# gemma-1.1-7b
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/huggingface_models/gemma-1.1-7b-it/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8





# 172.22.8.6
# mistral-7B-v0.3
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/huggingface_models/Mistral-7B-Instruct-v0.3/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-7B-v0.2
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/huggingface_models/Mistral-7B-Instruct-v0.2/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-7B-v0.1
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/zqiu/huggingface_models/Mistral-7B-Instruct-v0.1/ --dtype auto --api-key token-abc123 --tensor-parallel-size 8





# c4ai-command-r-v01
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/c4ai-command-r-v01 --dtype auto --api-key token-abc123 --tensor-parallel-size 8






# codellama-7B
# python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/CodeLlama-7b-Instruct-hf --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# codellama-70B
# python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/CodeLlama-70b-Instruct-hf --dtype auto --api-key token-abc123 --tensor-parallel-size 8





# Phi-3-mini-128k
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Phi-3-mini-128k-instruct --dtype auto --api-key token-abc123 --tensor-parallel-size 8




# qwen-1.5-7B
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-7B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# qwen-1.5-32B
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-32B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# qwen-1.5-72B
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-72B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# qwen-1.5-110B
python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-110B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8






# Yi-1.5-6B
# python -m vllm.entrypoints.openai.api_server --model /lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Yi-1.5-6B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# Yi-1.5-9B
python -m vllm.entrypoints.openai.api_server --model 01-ai/Yi-1.5-9B-Chat-16K --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# Yi-1.5-34B
python -m vllm.entrypoints.openai.api_server --model 01-ai/Yi-1.5-34B-Chat-16K --dtype auto --api-key token-abc123 --tensor-parallel-size 8