from openai import OpenAI
import argparse
from datasets import Dataset
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
 ...
</reasoning>
<answer>
  ...
</answer>
"""



def extract_answer_from_model_output(text):
    """提取 <answer>...</answer> 中的内容"""
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    last = parts[-1]
    if "</answer>" not in last:
        return None
    return last.split("</answer>")[0].strip()


def parse_options(options_str):
    return [o.strip() for o in options_str.strip().split(",")]


def parse_options_excluding_answer(options_str, answer):
    return [o for o in parse_options(options_str) if o != answer]



def generate_with_openai(messages, client,temperature, model_name="gpt-4o-mini", max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            return resp.choices[0].message.content

        except Exception as e:
            wait = 2 ** attempt
            print(f"[WARN] API 调用失败: {e}, 重试 {attempt+1}/{max_retries}, 等待 {wait}s")
            time.sleep(wait)

    # NEW: 最终失败返回 None
    return None



def prepare_dataset(path, split):
    data = Dataset.load_from_disk(path)
    data = data.shuffle(seed=42)

    formatted = []
    for idx, e in enumerate(data):
        if e["answer"] not in e["Options"]:
            continue

        formatted.append({
            "UID": f"{idx}_{e['Q_id']}",   # NEW: 唯一 ID
            "Q_id": str(e["Q_id"]),
            "prompt": e["prompt"],
            "answer": e["answer"],
            "Options": e["Options"],
            "Country": e.get("Country", ""),
            "Gender": e.get("Gender", ""),
            "Marital Status": e.get("Marital Status", ""),
            "Has Children": e.get("Has Children", ""),
            "Education Level": e.get("Education Level", ""),
            "Occupation": e.get("Occupation", ""),
            "Work Nature": e.get("Work Nature", ""),
            "Religion": e.get("Religion", ""),
            "Life Stage": e.get("Life Stage", ""),
            "Income Bracket": e.get("Income Bracket", "")
        })

    return formatted



def load_processed_uids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    uids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uids.add(row["UID"])
    return uids




def append_results(csv_path, rows, fieldnames):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)




def evaluate_model_openai(
    eval_examples,
    client,
    model_name,
    output_csv_path,
    temperature,
    batch_size=64,
    max_workers=64,
    max_retries=3,
    resume=True
):

    fieldnames = [
        "UID","Q_id","prompt","Country","Gender","Marital Status","Has Children",
        "Education Level","Occupation","Work Nature","Religion","Life Stage",
        "Income Bracket","response","answer","frequency","Options",
        "expected","predicted","is_correct"
    ]

    processed = load_processed_uids(output_csv_path) if resume else set()
    print(f"[INFO] 已处理 {len(processed)} 条")

    correct = 0
    already = 0


    def worker(example):
        uid = example["UID"]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": example["prompt"]}]},
        ]

        resp = generate_with_openai(messages, client,temperature, model_name, max_retries)

        if resp is None:
            # NEW: 返回 None 不写入 CSV
            print(f"[ERROR] UID={uid} 调用失败（超时或异常），跳过写入")
            return None

        predicted = extract_answer_from_model_output(resp)
        expected = example["answer"]
        un_opts = parse_options_excluding_answer(example["Options"], expected)

        is_correct = False
        if predicted == expected:
            is_correct = True
        else:
            if predicted and expected in predicted and all(opt not in predicted for opt in un_opts):
                is_correct = True

        return {
            "uid": uid,
            "row": {
                **example,
                "response": resp,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct
            },
            "correct": is_correct
        }


    rows_buffer = []
    batch = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for example in eval_examples:
            uid = example["UID"]

            if resume and uid in processed:
                already += 1
                continue

            batch.append(example)

            if len(batch) >= batch_size:
                futures = [pool.submit(worker, ex) for ex in batch]

                for fut in as_completed(futures):
                    r = fut.result()
                    if r is None:   # NEW: 跳过失败请求
                        continue
                    rows_buffer.append(r["row"])
                    processed.add(r["uid"])
                    if r["correct"]:
                        correct += 1

                append_results(output_csv_path, rows_buffer, fieldnames)
                rows_buffer = []
                batch = []

        # 剩余 batch
        if batch:
            futures = [pool.submit(worker, ex) for ex in batch]
            for fut in as_completed(futures):
                r = fut.result()
                if r is None:    # NEW: 跳过失败请求
                    continue
                rows_buffer.append(r["row"])
                processed.add(r["uid"])
                if r["correct"]:
                    correct += 1

            append_results(output_csv_path, rows_buffer, fieldnames)

    print(f"[INFO] 完成评估：写入 {len(processed)} 条，跳过已存在 {already} 条")
    print(f"[INFO] 正确数量：{correct}")

    return correct



def main(args):
    eval_data = prepare_dataset(args.eval_data_path, "test")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    evaluate_model_openai(
        eval_data,
        client,
        args.model_name,
        args.output_csv_path,
        args.temperature,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        resume=args.resume
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    main(args)
