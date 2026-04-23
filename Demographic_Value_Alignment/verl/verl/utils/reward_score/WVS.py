# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    """
    Extract the content inside the last <answer>...</answer> tag from the model output.

    Args:
        solution_str (str): The text output from the model.
        method (str): Reserved for compatibility with previous interface ("strict"/"flexible").
                      This argument no longer affects extraction logic.

    Returns:
        str or None: The extracted answer content, or None if no valid <answer> tag is found.

    Explanation:
        1. Looks for <answer>...</answer> tag pairs in the string.
        2. Returns the content from the last pair.
        3. If the tag is missing or the content is empty/"...", returns None.
    """
    # 保留 method 参数，为了兼容原来的接口
    assert method in ["strict", "flexible"]

    # 安全优化：长字符串仅保留末尾部分，提升正则匹配速度
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    # 查找所有 <answer>...</answer> 对
    import re
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL)

    if not matches:  # 没有找到任何 <answer> 标签
        return None

    # 取最后一个 <answer> 块，并去除空白字符
    final_answer = matches[-1].strip()

    # 如果内容为空或仅为 "..."，认为无效
    if final_answer == "" or final_answer == "...":
        return None

    return final_answer




def compute_score(solution_str, ground_truth,data_option, method="strict", format_score=1.0, score=2.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    filtered_options = [opt for opt in data_option if opt != ground_truth]


    answer = extract_solution(solution_str=solution_str, method=method)
    fmt_bonus = 0.0
    if "<reasoning>" in solution_str:
        fmt_bonus += 0.2
    if "</reasoning>" in solution_str:
        fmt_bonus += 0.2
    if "<answer>" in solution_str:
        fmt_bonus += 0.2
    if "</answer>" in solution_str:
        fmt_bonus += 0.2
    if answer is None:
        return fmt_bonus
    else:
        if answer == ground_truth:
            return score + fmt_bonus
        elif ground_truth in answer and all(unop not in answer for unop in filtered_options):
            return format_score+ fmt_bonus
        else:
            return fmt_bonus

        
