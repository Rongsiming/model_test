import os
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# 配置API客户端
client = OpenAI(
    api_key="sk-qckbdjlxsfwejnhghzimuwleeuvnvvvsqxvfcbmujknlsopm",
    base_url="https://api.siliconflow.cn/v1",
)

def load_answers(file_path):
    """加载答案文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取问题ID和答案
    import re
    answers_dict = {}
    qa_blocks = re.findall(r'Q(\d+):(.*?)(?=Q\d+:|$)', content, re.DOTALL)

    for q_id, answer in qa_blocks:
        answers_dict[q_id.strip()] = answer.strip()

    return answers_dict


def evaluate_answer(reference_answer, test_answer):
    """使用大模型评估答案"""
    prompt = f"""
    作为一个专业评估员，请比较参考答案和待测答案，并从以下四个方面对待测答案进行评分（满分为10分）：

    1. 自然语言处理能力：语言流畅性、语法准确性、表达清晰度
    2. 推理能力：逻辑性、推理深度、论证质量
    3. 回答准确性：内容正确性、全面性、相关性
    4. 回答效率：回答简洁度、要点把握、无冗余内容

    参考答案（满分10分）：
    {reference_answer}

    待测答案：
    {test_answer}

    请在评分后用1-2段话简要说明评分理由，最后给出总体评分（四项平均分）。

    输出格式：
    自然语言处理能力：分数
    推理能力：分数
    回答准确性：分数
    回答效率：分数
    评分理由：理由说明
    总体评分：平均分
    """

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[
            {"role": "system", "content": "您是一个专业的模型评估系统，负责评估AI模型回答的质量。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


def main():
    # 文件路径
    reference_file = r"./test_model_answer/deepseek-R1_answer.txt"
    test_file =r"./test_model_answer/DeepSeek-R1-Distill-Qwen-14B_answer.txt"
    # 提取测试模型名称
    test_model_name = os.path.basename(test_file).split('_')[0]
    output_file = f"./evaluation/{test_model_name}_evaluation.txt"

    # 加载答案
    print(f"📚 加载参考答案：{reference_file}")
    reference_answers = load_answers(reference_file)

    print(f"📝 加载待测答案：{test_file}")
    test_answers = load_answers(test_file)

    # 检查问题匹配
    common_questions = set(reference_answers.keys()) & set(test_answers.keys())
    print(f"🔍 找到 {len(common_questions)} 个共有问题")

    # 创建评估报告头部
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {test_model_name} 评估报告\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"参考模型: {os.path.basename(reference_file).split('_')[0]}\n")
        f.write(f"待测模型: {test_model_name}\n")
        f.write(f"评估问题数量: {len(common_questions)}\n\n")
        f.write("## 评估结果摘要\n\n")
        f.write("| 评估维度 | 平均分 |\n")
        f.write("|---------|-------|\n")
        f.write("| 自然语言处理能力 | 待计算 |\n")
        f.write("| 推理能力 | 待计算 |\n")
        f.write("| 回答准确性 | 待计算 |\n")
        f.write("| 回答效率 | 待计算 |\n")
        f.write("| **总体评分** | **待计算** |\n\n")
        f.write("## 详细评估结果\n\n")

    # 初始化分数统计
    total_scores = {
        "自然语言处理能力": 0,
        "推理能力": 0,
        "回答准确性": 0,
        "回答效率": 0,
        "总体评分": 0
    }

    # 逐题评估
    for q_id in tqdm(sorted(common_questions, key=int), desc="评估进度"):
        ref_answer = reference_answers[q_id]
        test_answer = test_answers[q_id]

        # 调用API评估
        try:
            evaluation = evaluate_answer(ref_answer, test_answer)

            # 解析评分
            import re
            nlp_score = float(re.search(r'自然语言处理能力：(\d+\.?\d*)', evaluation).group(1))
            reasoning_score = float(re.search(r'推理能力：(\d+\.?\d*)', evaluation).group(1))
            accuracy_score = float(re.search(r'回答准确性：(\d+\.?\d*)', evaluation).group(1))
            efficiency_score = float(re.search(r'回答效率：(\d+\.?\d*)', evaluation).group(1))
            total_score = float(re.search(r'总体评分：(\d+\.?\d*)', evaluation).group(1))

            # 累加分数
            total_scores["自然语言处理能力"] += nlp_score
            total_scores["推理能力"] += reasoning_score
            total_scores["回答准确性"] += accuracy_score
            total_scores["回答效率"] += efficiency_score
            total_scores["总体评分"] += total_score

            # 将评估结果写入文件
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"### Q{q_id}\n\n")
                f.write("#### 评估结果\n\n")
                f.write(f"{evaluation}\n\n")
                f.write("---\n\n")

        except Exception as e:
            print(f"评估问题 Q{q_id} 时出错: {str(e)}")
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"### Q{q_id} - 评估失败\n\n")
                f.write(f"错误信息: {str(e)}\n\n")
                f.write("---\n\n")

    # 计算平均分
    question_count = len(common_questions)
    avg_scores = {k: v / question_count for k, v in total_scores.items()}

    # 更新评估报告摘要
    with open(output_file, 'r', encoding='utf-8') as f:
        report = f.read()

    report = report.replace("自然语言处理能力 | 待计算", f"自然语言处理能力 | {avg_scores['自然语言处理能力']:.2f}")
    report = report.replace("推理能力 | 待计算", f"推理能力 | {avg_scores['推理能力']:.2f}")
    report = report.replace("回答准确性 | 待计算", f"回答准确性 | {avg_scores['回答准确性']:.2f}")
    report = report.replace("回答效率 | 待计算", f"回答效率 | {avg_scores['回答效率']:.2f}")
    report = report.replace("**总体评分** | **待计算**", f"**总体评分** | **{avg_scores['总体评分']:.2f}**")

    # 添加总结
    conclusion = f"""
## 评估总结

{test_model_name} 在与参考模型的比较中表现如下：

1. **自然语言处理能力**: {avg_scores['自然语言处理能力']:.2f}/10
   - 语言流畅性、语法准确性、表达清晰度的整体评价

2. **推理能力**: {avg_scores['推理能力']:.2f}/10
   - 逻辑性、推理深度、论证质量的整体评价

3. **回答准确性**: {avg_scores['回答准确性']:.2f}/10
   - 内容正确性、全面性、相关性的整体评价

4. **回答效率**: {avg_scores['回答效率']:.2f}/10
   - 回答简洁度、要点把握、无冗余内容的整体评价

**综合评分**: {avg_scores['总体评分']:.2f}/10

评估完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    report += conclusion

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 评估完成！报告已保存至 {output_file}")
    print(f"📊 总体评分: {avg_scores['总体评分']:.2f}/10")


if __name__ == "__main__":
    try:
        print("🚀 开始模型评估流程")
        main()
    except Exception as e:
        print(f"❌ 评估过程出错: {str(e)}")
        with open("evaluation_error.log", "a") as log:
            log.write(f"[{datetime.now()}] {str(e)}\n")