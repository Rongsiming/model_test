import re
import os
import time
from datetime import datetime, timedelta
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# 配置API客户端
client = OpenAI(
    api_key="22e0a307-fcf4-4029-b697-d42cde4b1733",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 确保输出目录存在
os.makedirs("test_model_answer", exist_ok=True)


def load_questions(file_path="mil_questions.txt"):
    """从文件加载问题"""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 分离每个问题
    q_pattern = r'Q\d+:(.*?)(?=Q\d+:|$)'
    matches = re.findall(q_pattern, content, re.DOTALL)

    # 清理并添加问题
    for q in matches:
        questions.append(q.strip())

    return questions


def get_model_answer(question, model_name, question_num=0):
    """使用指定模型回答问题"""
    system_prompt = """
    您是军事指挥系统运维领域的专家。请详细回答以下问题。
    答案必须包含具体参数（如温度范围：-40℃~71℃，加密标准：AES-256）。
    要求答案精简简短，准确率高，包括具体的操作步骤、技术规范和军事标准要求。
    需要重复问题内容。
    """

    start_time = time.time()
    try:
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.5,
            "max_tokens": 1500
        }


        response = client.chat.completions.create(**params)
        elapsed_time = time.time() - start_time
        return {
            "num": question_num,
            "answer": response.choices[0].message.content,
            "time": elapsed_time
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "num": question_num,
            "answer": f"模型回答出错: {str(e)}",
            "time": elapsed_time
        }


def process_questions_with_model(model_name, model_display_name=None, max_workers=4):
    """处理所有问题并保存模型回答，使用多线程加速"""
    if model_display_name is None:
        model_display_name = model_name

    questions = load_questions()
    output_file = f"test_model_answer/{model_display_name}_answer.txt"

    print(f"\n🤖 使用模型 {model_display_name} 进行问题回答...")
    print(f"🧵 启用并行处理，最大线程数: {max_workers}")

    # 记录整体开始时间
    total_start_time = time.time()

    # 使用线程池并行处理问题
    answers = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_question = {
            executor.submit(get_model_answer, question, model_name, idx + 1): idx
            for idx, question in enumerate(questions)
        }

        # 创建进度条
        progress_bar = tqdm(total=len(questions), desc="模型回答进度", unit="问题")

        # 处理完成的任务
        for future in as_completed(future_to_question):
            result = future.result()
            answers.append(result)
            progress_bar.update(1)

        progress_bar.close()

    # 记录整体结束时间和总耗时
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    total_elapsed_formatted = str(timedelta(seconds=round(total_elapsed_time)))

    # 按问题顺序排序答案
    answers.sort(key=lambda x: x["num"])

    # 计算平均响应时间
    response_times = [a["time"] for a in answers]
    avg_response_time = sum(response_times) / len(response_times)

    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        # 写入头部信息
        f.write(f"# 模型 {model_display_name} 军事指挥系统运维问题回答\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # 写入答案（不包含问题）
        for answer_data in answers:
            f.write(f"A{answer_data['num']}: {answer_data['answer']}\n")
            f.write(f"[响应时间: {answer_data['time']:.2f}秒]\n\n")
            f.write("-" * 80 + "\n\n")

        # 写入时间统计信息到文件最后
        f.write("=" * 80 + "\n")
        f.write("时间统计\n")
        f.write("-" * 40 + "\n")
        f.write(f"总处理时间: {total_elapsed_formatted} (总计: {total_elapsed_time:.2f}秒)\n")
        f.write(f"平均响应时间: {avg_response_time:.2f}秒\n")
        f.write(f"最快响应: {min(response_times):.2f}秒\n")
        f.write(f"最慢响应: {max(response_times):.2f}秒\n")
        f.write("=" * 80 + "\n")

    print(f"\n✅ 模型回答已保存至: {output_file}")
    print(f"⏱️ 总处理时间: {total_elapsed_formatted}")
    return output_file


def run_multiple_models(model_config):
    """运行多个模型测试"""
    print("🛡️ 启动GPU加速的军事指挥系统运维问题模型评测程序")
    total_evaluation_start = time.time()

    results = []
    for model in model_config:
        max_workers = 8
        output_file = process_questions_with_model(model["id"], model["name"], max_workers)
        results.append({
            "model": model["name"],
            "output_file": output_file
        })

    total_evaluation_time = time.time() - total_evaluation_start
    total_time_formatted = str(timedelta(seconds=round(total_evaluation_time)))

    print("\n📊 评测完成概览:")
    for result in results:
        print(f"- {result['model']}: {result['output_file']}")
    print(f"\n⏱️ 全部评测总耗时: {total_time_formatted}")


if __name__ == "__main__":
    try:
        # 配置要测试的模型
        models_to_test = [
            {
                "id": "ep-20250227212251-6zq2v",  # API调用时使用的ID
                "name": "DeepSeek-R1-Distill-Qwen-32B"  # 显示名称和文件名
            },

            {
                "id": "ep-20250218175517-6jc9n",
                "name": "deepseek-R1",
            }
        ]

        # 运行多模型评测
        run_multiple_models(models_to_test)

    except Exception as e:
        print(f"\n❌ 系统告警：{str(e)}")
        # 记录错误日志
        with open("combat_system_error.log", "a") as log:
            log.write(f"[{datetime.now()}] {str(e)}\n")

