import re
import os
import time
from datetime import datetime, timedelta
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# 配置API客户端
client = OpenAI(
    api_key="sk-qckbdjlxsfwejnhghzimuwleeuvnvvvsqxvfcbmujknlsopm",
    base_url="https://api.siliconflow.cn/v1",
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
    您是军事指挥系统运维领域的专家。请根据以下问题，结合逻辑推理、技术规范和军事标准，提供详细且准确的回答。  
    要求答案精练简洁，准确率高，体现大模型的推理能力，包含以下内容：  
    1. **操作步骤**：清晰的执行流程。  
    2. **技术规范**：相关技术标准或方法，不用包含数字。  
    3. **军事标准**：适用的军事规范或要求。  
    4. **逻辑推理**：展示问题分析、解决方案推导的过程。  
    
    **输出格式如下：**  
    Q1: [问题正文]  
    A1: [答案正文，包含操作步骤、技术规范、军事标准及逻辑推理过程]  
    Q2: [问题正文]  
    A2: [答案正文，包含操作步骤、技术规范、军事标准及逻辑推理过程]  
    ...  
    
    **请确保回答：**  
    1. **简明扼要**：避免冗余信息，突出重点。  
    2. **准确无误**：符合技术规范和军事标准。  
    3. **逻辑清晰**：展示推理过程，体现问题解决能力。  
    4. **多样化**：针对不同类型问题，提供针对性解决方案。  
    
    ---
    """

    start_time = time.time()
    try:
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
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
            f.write(f"{answer_data['answer']}\n")
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
                "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "name": "DeepSeek-R1-Distill-Qwen-7B",
            },
            {
                "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "name": "DeepSeek-R1-Distill-Qwen-32B",
            },
            {
                "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                "name": "DeepSeek-R1-Distill-Qwen-14B",
            },
            {
                "id": "Qwen/Qwen2.5-32B-Instruct",
                "name": "Qwen2.5-32B-Instruct",
            },
            {
                "id": "Qwen/Qwen2.5-14B-Instruct",
                "name": "Qwen2.5-14B-Instruct",
            },
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "Qwen2.5-7B-Instruct",
            },
            {
                "id": "Qwen/QwQ-32B-Preview",
                "name": "QwQ-32B-Preview",
            },
            {
                "id": "THUDM/glm-4-9b-chat",
                "name": "glm-4-9b-chat",
            },
            {
                "id": "deepseek-ai/DeepSeek-R1",
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

