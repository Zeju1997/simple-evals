import json
import time

import pandas as pd

from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .local_eval import LOCALEval
from .mnist_eval import MNISTEval
from .cad_eval import CADEval
from .symmetry_eval import SymmetryEval
from .sampler.gpt_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OpenaiChatCompletionSampler,
)

from .sampler.claude_sampler import (
    ClaudeCompletionSampler, 
    CLAUDE_SYSTEM_MESSAGE_API,
    CLAUDE_SYSTEM_MESSAGE_LMSYS,
)

from .sampler.gemini_sampler import (
    GeminiCompletionSampler,
    GEMINI_SYSTEM_MESSAGE_API,
)

from .sampler.qwen_sampler import (
    QwenCompletionSampler,
    QWEN_SYSTEM_MESSAGE_API,
)

from .sampler.open_sampler import (
    OpenChatCompletionSampler,
    OPEN_SYSTEM_MESSAGE_API,
)

import os
import argparse



def main(args):
    debug = False

    if args.api == "openai":
        samplers = {
            # chatgpt models: gpt-4o-2024-05-13
            "gpt-4o-2024-05-13": OpenaiChatCompletionSampler(
                model="gpt-4o-2024-05-13",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            )
        }
    elif args.api == "claude":
        samplers = {
            # claude models:
            "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
                model="claude-3-opus-20240229", system_message=CLAUDE_SYSTEM_MESSAGE_API,
            ),
        }
    elif args.api == "gemini":
        samplers = {
            # claude models:
            "gemini-1.5-pro-latest": GeminiCompletionSampler(
                model="gemini-1.5-pro-latest", system_message=GEMINI_SYSTEM_MESSAGE_API,
            ),
        }
    elif args.api == "qwen":
        samplers = {
            # qwen models:
            "qwen-turbo": QwenCompletionSampler(
                model="qwen-turbo",
                system_message=QWEN_SYSTEM_MESSAGE_API,
            ),
        }
    elif args.api == "llama3-8B":
        samplers = {
            "llama3-8B": OpenChatCompletionSampler(
                model="models/Meta-Llama-3-8B-Instruct",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "llama3-70B":
        samplers = {
            "llama3-70B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/huggingface_models/Meta-Llama-3-70B-Instruct/",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "gemma-1.1-2b":
        samplers = {
            "gemma-1.1-2b": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/huggingface_models/gemma-1.1-2b-it/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "gemma-1.1-7b":
        samplers = {
            "gemma-1.1-7b": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/huggingface_models/gemma-1.1-7b-it/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "mistral-7B-v0.1":
        samplers = {
            "mistral-7B-v0.1": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/huggingface_models/Mistral-7B-Instruct-v0.1/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "mistral-7B-v0.2":
        samplers = {
            "mistral-7B-v0.2": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/huggingface_models/Mistral-7B-Instruct-v0.2/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "mistral-7B-v0.3":
        samplers = {
            "mistral-7B-v0.3": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/huggingface_models/Mistral-7B-Instruct-v0.3/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "c4ai-command-r-v01":
        samplers = {
            "c4ai-command-r-v01": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/c4ai-command-r-v01",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url,
                max_tokens=1024
            ),
        }
    elif args.api == "codellama-7B":
        samplers = {
            "codellama-7B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/CodeLlama-7b-Instruct-hf",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "codellama-70B":
        samplers = {
            "codellama-70B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/CodeLlama-70b-Instruct-hf",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "Phi-3-mini-128k":
        samplers = {
            "Phi-3-mini-128k": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Phi-3-mini-128k-instruct",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "qwen-1.5-7B":
        samplers = {
            "qwen-1.5-7B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-7B-Chat",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "qwen-1.5-32B":
        samplers = {
            "qwen-1.5-32B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-32B-Chat",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "qwen-1.5-72B":
        samplers = {
            "qwen-1.5-72B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-72B-Chat",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "qwen-1.5-110B":
        samplers = {
            "qwen-1.5-110B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/models/Qwen1.5-110B-Chat",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "Yi-1.5-9B":
        samplers = {
            "Yi-1.5-9B": OpenChatCompletionSampler(
                model="01-ai/Yi-1.5-9B-Chat-16K",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "Yi-1.5-34B":
        samplers = {
            "Yi-1.5-34B": OpenChatCompletionSampler(
                model="01-ai/Yi-1.5-34B-Chat-16K",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    # unsloth-base
    elif args.api == "unsloth-llama3-8B":
        samplers = {
            "unsloth-llama3-8B": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/unsloth_models/llama-3-8b-Instruct/",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "unsloth-gemma-1.1-7b-it":
        samplers = {
            "gemma-1.1-7b-it": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/unsloth_models/gemma-1.1-7b-it/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "unsloth-mistral-7b-v0.3":
        samplers = {
            "unsloth-mistral-7b-v0.3": OpenChatCompletionSampler(
                model="/lustre/fast/fast/zqiu/unsloth_models/mistral-7b-instruct-v0.3/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    # evaluate finetuned models
    elif args.api == "llama3-8b-test-0":
        samplers = {
            "llama3-8b-test-0": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_0/",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "llama3-8b-test-25k":
        samplers = {
            "llama3-8b-test-25k": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_25000_mini",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "llama3-8b-test-50k":
        samplers = {
            "llama3-8b-test-50k": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_50000_mini/",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "llama3-8b-test-full":
        samplers = {
            "llama3-8b-test-full": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/llama-3-8b-Instruct_full/",
                system_message=OPEN_SYSTEM_MESSAGE_API,
                base_url=args.base_url
            ),
        }
    elif args.api == "gemma-1.1-7b-it-test-25k":
        samplers = {
            "gemma-1.1-7b-it-test-25k": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/gemma-1.1-7b-it_10000/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "gemma-1.1-7b-it-test-50k":
        samplers = {
            "gemma-1.1-7b-it-test-50k": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/gemma-1.1-7b-it_50000/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "gemma-1.1-7b-it-test-full":
        samplers = {
            "gemma-1.1-7b-it-test-full": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/gemma-1.1-7b-it_full/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "mistral-7b-v0.3-test-25k":
        samplers = {
            "mistral-7b-v0.3-test-25k": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/mistral-7b-instruct-v0.3_10000/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "mistral-7b-v0.3-test-50k":
        samplers = {
            "mistral-7b-v0.3-test-50k": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/mistral-7b-v0.3_50000/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    elif args.api == "mistral-7b-v0.3-test-full":
        samplers = {
            "mistral-7b-v0.3-test-full": OpenChatCompletionSampler(
                model="/lustre/fast/fast/groups/ps-invsolid/InvSolid/unsloth/mistral-7b-v0.3_full/",
                system_message=None,
                base_url=args.base_url
            ),
        }
    else:
        raise ValueError(f"Invalid API: {args.api}")

    equality_checker = OpenaiChatCompletionSampler(model="gpt-4o")
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math



    def get_evals(eval_name, api):
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=10 if debug else 2500)
            case "gb_main":
                return LOCALEval(num_examples=10 if debug else None, mode="main")
            case "gb_raw":
                return LOCALEval(num_examples=10 if debug else None, mode="raw")
            case "gb_color":
                return LOCALEval(num_examples=10 if debug else None, mode="color")
            case "gb_count":
                return LOCALEval(num_examples=10 if debug else None, mode="count")
            case "gb_semantics":
                return LOCALEval(num_examples=10 if debug else None, mode="semantics")
            case "gb_reasoning":
                return LOCALEval(num_examples=10 if debug else None, mode="reasoning")
            case "gb_shape":
                return LOCALEval(num_examples=10 if debug else None, mode="shape")
            case "gb_raw":
                return LOCALEval(num_examples=10 if debug else None, mode="raw")
            case "gb_symmetry":
                return SymmetryEval(
                    equality_checker=equality_checker, num_examples=10 if debug else None, mode="symmetry"
                )
            case "mnist":
                return MNISTEval(
                    equality_checker=equality_checker, num_examples=10 if debug else None, mode="mnist"
            )
            case "gb_inv":
                return LOCALEval(num_examples=10 if debug else None, mode="inv")
            case "gb_inv_t0":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_t0")
            case "gb_inv_t1":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_t1")
            case "gb_inv_t2":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_t2")
            case "gb_inv_t3":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_t3")
            case "gb_inv_t4":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_t4")
            case "gb_inv_r0":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_r0")
            case "gb_inv_r1":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_r1")
            case "gb_inv_r2":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_r2")
            case "gb_inv_r3":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_r3")
            case "gb_inv_r4":
                return LOCALEval(num_examples=10 if debug else None, mode="inv_r4")

            case "cad":
                return CADEval(num_examples=10 if debug else None)

            case "math":
                return MathEval(
                    equality_checker=equality_checker, num_examples=5 if debug else 2500
                )
            case "gpqa":
                return GPQAEval(n_repeats=1 if debug else 10, num_examples=5 if debug else None)
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug else 250)
            case "drop":
                return DropEval(num_examples=10 if debug else 2000, train_samples_per_prompt=3)
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.api) for eval_name in ["gb_main"] #, "gb_semantics", "gb_count", "gb_color", "gb_shape", "gb_reasoning", "cad", "mnist", "gb_inv", "gb_inv_t0", "gb_inv_t1", "gb_inv_t2", "gb_inv_t3", "gb_inv_t4", "gb_inv_r0", "gb_inv_r1", "gb_inv_r2", "gb_inv_r3", "gb_inv_r4"]
        # eval_name: get_evals(eval_name, args.api) for eval_name in ["gb_inv", "gb_inv_t0", "gb_inv_t1", "gb_inv_t2", "gb_inv_t3", "gb_inv_t4", "gb_inv_r0", "gb_inv_r1", "gb_inv_r2", "gb_inv_r3", "gb_inv_r4"]
        # eval_name: get_evals(eval_name) for eval_name in ["gb_main", "gb_symmetry", "gb_raw", "cad", "mnist", "math", "gpqa", "mgsm", "drop"]
    }
    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = os.path.join(os.path.dirname(__file__), f"results/{file_stem}{debug_suffix}.html")
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = os.path.join(os.path.dirname(__file__), f"results/{file_stem}{debug_suffix}.json")
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_sampler_name[: eval_sampler_name.find("_")]
        sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://172.22.8.7:8000/v1")
    parser.add_argument(
        "--api", 
        choices=[
            "openai", 
            "claude", 
            "gemini", 
            "qwen", 
            "llama3-8B", 
            "llama3-70B",
            "gemma-1.1-2b",
            "gemma-1.1-7b",
            "mistral-7B-v0.1",
            "mistral-7B-v0.2",
            "mistral-7B-v0.3",
            "c4ai-command-r-v01",
            "codellama-7B",
            "codellama-70B",
            "Phi-3-mini-128k",
            "qwen-1.5-7B",
            "qwen-1.5-32B",
            "qwen-1.5-72B",
            "qwen-1.5-110B",
            # "Yi-1.5-6B", # This model's maximum context length is 4096 tokens. However, you requested 4165 tokens (2117 in the messages, 2048 in the completion).
            "Yi-1.5-9B",
            "Yi-1.5-34B",
            # "unsloth-llama3-8B",
            # "unsloth-gemma-1.1-7b-it",
            # "unsloth-mistral-7b-v0.3",
            "llama3-8b-test-0",
            "llama3-8b-test-25k",
            "llama3-8b-test-50k",
            "llama3-8b-test-full",
            "gemma-1.1-7b-it-test-25k",
            "gemma-1.1-7b-it-test-50k",
            "gemma-1.1-7b-it-test-full",
            "mistral-7b-v0.3-test-25k",
            "mistral-7b-v0.3-test-50k",
            "mistral-7b-v0.3-test-full"
        ], 
        default="llama3-70B"
    )
    args = parser.parse_args()

    main(args)

# "gemma-1.1-7b-it-test-10k",
# "mistral-7b-v0.3-test-10k",

# python -m simple-evals.demo --api openai
# 0: python -m simple-evals.demo --base_url http://172.22.8.1:8000/v1 --api llama3-8B
# 2: python -m simple-evals.demo --base_url http://172.22.8.6:8000/v1 --api llama3-70B
# 3: python -m simple-evals.demo --base_url http://172.22.8.14:8000/v1 --api gemma-1.1-2b
# 4: python -m simple-evals.demo --base_url http://172.22.8.7:8000/v1 --api gemma-1.1-7b
# 5: python -m simple-evals.demo --base_url http://172.22.8.16:8000/v1 --api c4ai-command-r-v01