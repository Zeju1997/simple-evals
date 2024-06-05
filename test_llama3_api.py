from openai import OpenAI

SYS_PROMPT = "You are a helpful, respectful and honest assistant."

def setup_api_client(base_url, api_key):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    return client

def execute_forward_pass(fx_prompt, llm_endpoint, model_name):
    forward_state=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": fx_prompt}
            ]
    completion = llm_endpoint.chat.completions.create(
            model=model_name,
            messages=forward_state,
        )
    forward_state.append({"role": "assistant", "content": completion.choices[0].message.content})

    print(completion)
    # print(completion.choices[0].message.content)

    return completion.choices[0].message.content, forward_state

hp = {
    "model_name": "/lustre/fast/fast/zqiu/huggingface_models/Meta-Llama-3-70B-Instruct/",
    "tensor_parallel_size": 8,
    "api_key": "token-abc123",
    "base_url": "http://172.22.8.7:8000/v1",
}

llm_endpoint = setup_api_client(hp['base_url'], hp['api_key'])


user_prompt = "What is a occam's razor in machine learning?"

for i in range(10):
    forward_output, forward_state = execute_forward_pass(user_prompt, llm_endpoint, hp['model_name'])
    print(forward_output)
