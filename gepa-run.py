import argparse
import os
import json
import csv
from datetime import datetime
from typing import Dict, Any, List

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration
    )
import wandb

import gepa
from util_data import init_math500, init_aime2025, init_amc23, init_aime2024, init_minerva, init_olympiad

from src.math_grader import grade_answer

# ==============================
# 1. 모델 로딩 / task_lm 생성
# ==============================

def normalize_multimodal_context(context: list) -> list:
    """
    이러한 형식은 multimodal을 지원하는 모델에서 나오는 패턴
    """
    new_context = []
    for message in context:
        content = message["content"]
        if isinstance(content, str):
            new_content = [{"type": "text", "text": content}]
        else:
            new_content = content  # 이미 list[dict] 형태면 그대로 유지
        new_context.append({
            "role": message["role"],
            "content": new_content
        })
    return new_context

def build_model_and_tokenizer(model_name: str, cache_dir: str | None) -> tuple[Any, Any]:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    quantization_config = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
    }

    tokenizer = None
    processor = None

    if 'llama' in model_name:

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(**quantization_config),
            cache_dir=cache_dir,
            attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2"),
            token=os.getenv("HF_TOKEN"),
        )

        step_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)

        eos_id = tokenizer.encode(
            "<|end_of_text|>", add_special_tokens=False
        )[0]
        pad_id = 128248
        pad_token = tokenizer.decode(pad_id)
        model.generation_config.pad_token_id = pad_id
        model.generation_config.eos_token_id = eos_id
    elif 'gemma' in model_name:

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(**quantization_config),
            cache_dir=cache_dir,
            attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2"),
            token=os.getenv("HF_TOKEN"),
        )

        eos_id = tokenizer.encode(
            "<|end_of_turn|>", add_special_tokens=False
        )[0]
        pad_id = 0
        pad_token = tokenizer.decode(pad_id)
        model.generation_config.pad_token_id = pad_id
        model.generation_config.eos_token_id = [eos_id, model.generation_config.eos_token_id]

        processor = AutoProcessor.from_pretrained(model_name)
    else:
        raise KeyError
    # model.config.pad_token_id = model.config.eos_token_id
    # tokenizer.pad_token = tokenizer.eos_token

    # model.to(DEVICE)

    return model, tokenizer, processor


def normalize_messages(messages):
    normalized = []
    for m in messages:
        if isinstance(m, str):
            normalized.append({"role": "user", "content": m})
        elif isinstance(m, dict):
            normalized.append(m)
        else:
            raise TypeError(f"Unsupported message type: {type(m)}")
    return normalized


def make_task_lm(model, tokenizer, processor=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cache: Dict[str, str] = {}

    def get_task_key(messages: List[Dict[str, str]]) -> str:
        import json as _json
        return str(("task_lm", _json.dumps(messages, sort_keys=True)))

    def task_lm(messages: List[Dict[str, str]]) -> str:
        """
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            ...
        ]
        """

        key = get_task_key(messages)
        if key in cache:
            return cache[key]

        # Gemma 계열 (processor 존재) vs 일반 text-only 모델 분기
        if processor is not None:
            # 🔹 multimodal-friendly 포맷으로 정규화
            chat = normalize_multimodal_context(messages)

            # Gemma3 + AutoProcessor는 보통 이렇게 씀:
            # apply_chat_template(..., return_tensors="pt") → 이미 input_ids 텐서 반환
            inputs = processor.apply_chat_template(
                chat,
                add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt",
            ).to(DEVICE)

        else:
            # 🔹 기존 LLaMA 코드 경로
            messages_norm = normalize_messages(messages)
            prompt = tokenizer.apply_chat_template(
                messages_norm,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,                
            )

        # 🔹 디코딩은 tokenizer로 (Gemma도 tokenizer 공유)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Gemma 계열 (processor 존재) vs 일반 text-only 모델 분기
        if processor is not None:
            response = text.split("<start_of_turn>model")[-1].strip()
        else: 
            response = text.split("assistant<|end_header_id|>")[-1].strip()

        cache[key] = response
        return response

    return task_lm



# ==============================
# 2. 평가 함수
# ==============================

def extract_final_answer(s: str) -> str:
    if '\boxed{' in s:
        start = s.find(r"\boxed{")
        if start == -1:
            return None
        
        i = start + len(r"\boxed{")
        depth = 1
        content = []
        
        while i < len(s) and depth > 0:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            content.append(s[i])
            i += 1
        return ''.join(content)           
    else:
        return s.split('###')[-1].strip(' ')
# def extract_final_answer(s: str) -> str:
#     start = s.find(r"\boxed{")
#     if start == -1:
#         return None
    
#     i = start + len(r"\boxed{")
#     depth = 1
#     content = []
    
#     while i < len(s) and depth > 0:
#         if s[i] == '{':
#             depth += 1
#         elif s[i] == '}':
#             depth -= 1
#             if depth == 0:
#                 break
#         content.append(s[i])
#         i += 1
    
#     return ''.join(content)   


def evaluate_on_test(
    best_candidate: Dict[str, str],
    dataset_name:str,
    testset: list[Dict[str, Any]],
    task_lm,
    save_jsonl: str,
    save_csv: str,
    save_summary: str,
    log_to_wandb: bool = True,
) -> Dict[str, Any]:
    system_prompt = best_candidate["system_prompt"]

    csv_rows = []
    num_correct = 0
    total = len(testset)

    os.makedirs(os.path.dirname(save_jsonl), exist_ok=True)

    # JSONL 파일
    with open(save_jsonl, "w", encoding="utf-8") as jsonl_f:
        for idx, inst in enumerate(testset):
            question = inst["input"]
            gold = str(inst["answer"].split('###')[-1]).strip()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            model_output = task_lm(messages)
            pred = extract_final_answer(model_output)

            if dataset_name in ['amc23']:
                correct = grade_answer(pred, gold)
            else:
                correct = pred == gold
            if correct:
                num_correct += 1

            record = {
                "idx": idx,
                "correct": correct,
                "gold": gold,
                "pred": pred,
                "question": question,
                "raw_output": model_output,
            }

            # JSONL: 샘플별 1 라인
            import json as _json

            jsonl_f.write(_json.dumps(record, ensure_ascii=False) + "\n")

            # CSV row 준비
            csv_rows.append(
                [
                    idx,
                    question,
                    gold,
                    pred,
                    correct,
                    model_output.replace("\n", "\\n"),
                ]
            )

    acc = num_correct / total

    # CSV 저장
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "question", "gold", "pred", "correct", "raw_output"])
        writer.writerows(csv_rows)

    # Summary 저장
    summary = {
        "created_at": datetime.now().isoformat(),
        "accuracy": acc,
        "num_correct": num_correct,
        "num_examples": total,
        "jsonl_path": save_jsonl,
        "csv_path": save_csv,
    }

    with open(save_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[TEST] accuracy={acc:.4f} ({num_correct}/{total})")
    print(f"[TEST] Sample results saved → {save_jsonl}")
    print(f"[TEST] CSV saved → {save_csv}")
    print(f"[TEST] Summary saved → {save_summary}")

    if log_to_wandb and wandb.run is not None:
        wandb.log(
            {
                "test/accuracy": acc,
                "test/num_correct": num_correct,
                "test/num_examples": total,
            }
        )
        wandb.save(save_jsonl, policy="now")
        wandb.save(save_csv, policy="now")
        wandb.save(save_summary, policy="now")

    return summary


# ==============================
# 3. Dataset 로딩 헬퍼
# ==============================

def load_dataset_by_name(name: str, train_full_size:int):
    if name == "math500":
        return init_math500(train_full_size)
    elif name == 'aime2025':
        return init_aime2025(train_full_size)
    elif name == 'aime2024':
        return init_aime2024(train_full_size)
    elif name == 'amc23':
        return init_amc23(train_full_size)
    elif name == 'minerva':
        return init_minerva(train_full_size)
    elif name == 'olympiad':
        return init_olympiad(train_full_size)
    else:
        raise KeyError(f"Unsupported dataset: {name}")


# ==============================
# 4. Argparse & main
# ==============================

def sanitize_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=["math500", 'aime2025', 'amc23', 'aime2024', 'minerva', 'olympiad'],
        help="Which dataset loader to use",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="default",
        choices=["default", "stepPrompt"],
        help="Seed prompt version"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.getenv("CACHE_DIR"),
        help="HF cache dir",
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default="./runs",
        help="Root directory for GEPA run dirs",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="./results",
        help="Root directory for test results",
    )
    parser.add_argument(
        "--max_metric_calls",
        type=int,
        default=10,
        help="GEPA max metric calls budget",
    )
    parser.add_argument(
        "--train_full_size",
        type=int,
        default=None,
        help="GEPA max metric calls budget",
    )
    parser.add_argument(
        "--reflection_minibatch_size",
        type=int,
        default=2,
        help="Minibatch size for reflection",
    )

    # wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging inside GEPA and for test evaluation",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="gepa-math500",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="wandb run name (default: auto-generated)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_tag = sanitize_name(args.model_name)
    dataset_name = args.dataset
    prompt_version = args.prompt_version

    # --- 디렉토리들 ---
    run_dir = os.path.join(f'{args.run_root}-prompt{prompt_version}', f"{dataset_name}-{model_tag}")
    results_dir = os.path.join(f'{args.results_root}-prompt{prompt_version}', dataset_name, model_tag)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Using model   : {args.model_name}")
    print(f"Using dataset : {dataset_name}")
    print(f"Run dir       : {run_dir}")
    print(f"Results dir   : {results_dir}")

    # --- 데이터 로드 ---
    trainset, valset, testset = load_dataset_by_name(dataset_name, args.train_full_size)
    print(f"trainset len = {len(trainset)}")
    print(f"valset   len = {len(valset)}")
    print(f"testset  len = {len(testset)}")

    # --- 모델 로드 & task_lm 생성 ---
    model, tokenizer, processor = build_model_and_tokenizer(args.model_name, args.cache_dir)
    task_lm = make_task_lm(model, tokenizer, processor=processor)


    # --- 초기 시스템 프롬프트 ---
    SEED_PROMPTS = {
        "default": {
            "system_prompt": (
                "You are a helpful assistant. "
                "You are given a question and you need to answer it. "
                "The answer should be given at the end of your response "
                "in exactly the format '### <final answer>'"
            )
        },
        "stepPrompt": {
            "system_prompt": """
    Solve the following math problem efficiently and clearly:

    - For simple problems (2 steps or fewer):
    Provide a concise solution with minimal explanation.

    - For complex problems (3 steps or more):
    Use this step-by-step format:

    ## Step 1: [Concise description]
    [Brief explanation and calculations]

    ## Step 2: [Concise description]
    [Brief explanation and calculations]

    ...

    Regardless of the approach, always conclude with:

    Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

    Where [answer] is just the final number or expression that solves the problem.
    """
        },
    }

    # seed_prompt = {
    #     "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    # }    
    # # seed_prompt = {
    # #     "system_prompt": """
    # #     Solve the following math problem efficiently and clearly:

    # #     - For simple problems (2 steps or fewer):
    # #     Provide a concise solution with minimal explanation.

    # #     - For complex problems (3 steps or more):
    # #     Use this step-by-step format:

    # #     ## Step 1: [Concise description]
    # #     [Brief explanation and calculations]

    # #     ## Step 2: [Concise description]
    # #     [Brief explanation and calculations]

    # #     ...

    # #     Regardless of the approach, always conclude with:

    # #     Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

    # #     Where [answer] is just the final number or expression that solves the problem.
    # #     """
    # # }

    # use_merge=True
    use_merge=False
    # --- GEPA 최적화 ---
    gepa_kwargs: Dict[str, Any] = dict(
        seed_candidate=SEED_PROMPTS[args.prompt_version],
        trainset=trainset,
        valset=valset,
        task_lm=task_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=task_lm,
        reflection_minibatch_size=args.reflection_minibatch_size,
        use_merge=use_merge,
        logger=None,
        run_dir=run_dir,
        use_mlflow=False,
        display_progress_bar=True,
        seed=0,
    )

    if args.use_wandb:
        gepa_kwargs.update(
            dict(
                use_wandb=True,
                wandb_api_key=os.getenv("WANDB_API_KEY"),
                wandb_init_kwargs={
                    "project": args.wandb_project,
                    "name": args.wandb_run_name
                    or f"gepa-{dataset_name}-{model_tag}",
                    "config": {
                        "model_name": args.model_name,
                        "dataset": dataset_name,
                        'use_merge': use_merge,
                        "max_metric_calls": args.max_metric_calls,
                        "reflection_minibatch_size": args.reflection_minibatch_size,
                        "train_full_size": args.train_full_size,
                    },
                },
            )
        )
    else:
        gepa_kwargs.update(
            dict(
                use_wandb=False,
                wandb_api_key=None,
                wandb_init_kwargs=None,
            )
        )

    gepa_result = gepa.optimize(**gepa_kwargs)

    print("GEPA Optimized Prompt:")
    print(gepa_result.best_candidate["system_prompt"])

    # --- Test 평가 & 저장 ---
    jsonl_path = os.path.join(results_dir, f"{dataset_name}-{model_tag}.jsonl")
    csv_path = os.path.join(results_dir, f"{dataset_name}-{model_tag}.csv")
    summary_path = os.path.join(
        results_dir, f"{dataset_name}-{model_tag}-summary.json"
    )

    test_summary = evaluate_on_test(
        best_candidate=gepa_result.best_candidate,
        dataset_name=dataset_name,
        testset=testset,
        task_lm=task_lm,
        save_jsonl=jsonl_path,
        save_csv=csv_path,
        save_summary=summary_path,
        log_to_wandb=args.use_wandb,
    )

    # wandb에 추가 요약 로그 (이미 evaluate_on_test에서 log했다면 생략해도 됨)
    if args.use_wandb and wandb.run is not None:
        wandb.log(
            {
                "test/accuracy_final": test_summary["accuracy"],
                "test/num_examples_final": test_summary["num_examples"],
                "test/num_correct_final": test_summary["num_correct"],
            }
        )
        # 결과 파일들도 run에 첨부
        wandb.save(summary_path, policy="now")

    print("Done.")


if __name__ == "__main__":
    main()
