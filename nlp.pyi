import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
MODEL_NAME = "google/flan-t5-xl"  # 可替换为 LLaMA2 / Mistral
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
SAMPLE_SIZE = 10

# ===== 自动评估类 =====
class Evaluator:
    def __init__(self):
        self.results = {
            "baseline": {"correct": 0, "total": 0, "details": []},
            "cot": {"correct": 0, "total": 0, "details": []}
        }

    def extract_number(self, text):
        """提取文本中的最后一个数字"""
        numbers = re.findall(r'\d+', text)
        return int(numbers[-1]) if numbers else None

    def add_result(self, method, question, pred_text, correct_num):
        pred_num = self.extract_number(pred_text)
        is_correct = pred_num == correct_num
        self.results[method]["total"] += 1
        if is_correct:
            self.results[method]["correct"] += 1
        self.results[method]["details"].append({
            "question": question,
            "pred_text": pred_text,
            "pred_num": pred_num,
            "correct_num": correct_num,
            "correct": is_correct
        })

    def summarize(self):
        print("\n" + "="*50)
        print("自动评估结果汇总")
        print("="*50)
        for method in ["baseline", "cot"]:
            acc = self.results[method]["correct"] / self.results[method]["total"] * 100
            print(f"{method.upper():<10} 准确率: {self.results[method]['correct']}/{self.results[method]['total']} = {acc:.1f}%")

        # 输出详细对比表
        print("\n详细对比:")
        for i in range(len(self.results["baseline"]["details"])):
            b = self.results["baseline"]["details"][i]
            c = self.results["cot"]["details"][i]
            print(f"\n问题 {i+1}: {b['question']}")
            print(f"Baseline: {b['pred_text']} (预测: {b['pred_num']}, 正确: {b['correct_num']}, {'✔' if b['correct'] else '✘'})")
            print(f"CoT:      {c['pred_text']} (预测: {c['pred_num']}, 正确: {c['correct_num']}, {'✔' if c['correct'] else '✘'})")

# ===== 加载数据集 =====
print(f"加载数据集: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

# ===== 模型加载 =====
print(f"加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ===== Prompt 模板 =====
def baseline_prompt(q):
    return f"Q: {q}\nA:"

def cot_prompt(q):
    return f"""
Q: 停车场有15辆车，开走7辆，又开来4辆，现在有几辆？
A: 停车场原有15辆车，开走7辆后剩下 15 - 7 = 8 辆。然后开来4辆，所以现在有 8 + 4 = 12 辆。答案是 12。

Q: {q}
A:"""

# ===== 生成答案 =====
def generate_answer(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=False
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("A:")[-1].strip()

# ===== 运行对比并评估 =====
evaluator = Evaluator()

print("\n===== Baseline =====")
for sample in dataset:
    q = sample["question"]
    correct_num = evaluator.extract_number(sample["answer"])
    pred_text = generate_answer(baseline_prompt(q))
    evaluator.add_result("baseline", q, pred_text, correct_num)

print("\n===== CoT =====")
for sample in dataset:
    q = sample["question"]
    correct_num = evaluator.extract_number(sample["answer"])
    pred_text = generate_answer(cot_prompt(q))
    evaluator.add_result("cot", q, pred_text, correct_num)

# ===== 自动评估结果 =====
evaluator.summarize()