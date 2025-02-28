import re
from datetime import datetime

from openai import OpenAI
from tqdm import tqdm

# 配置API客户端
client = OpenAI(
    api_key="22e0a307-fcf4-4029-b697-d42cde4b1733",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


def generate_questions():
    """仅生成军事指挥系统运维问题"""
    prompt = """
    生成30个军事指挥系统运维问题，要求：
    1. 包含以下标准：MIL-STD-810H（环境测试）、MIL-STD-461G（电磁兼容）
    2. 问题类型分布：硬件(40%)、软件(30%)、网络(20%)、数据(10%)
    3. 输出格式：
        Q1: [问题正文]
        Q2: [问题正文]
        ...等等
    4. 请只生成问题，不需要提供答案
    """

    response = client.chat.completions.create(
        model="ep-20250218175517-6jc9n",
        messages=[
            {"role": "system", "content": "您现在是军委装备发展部的技术规范生成系统"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=3500,
        top_p=0.95
    )
    return response.choices[0].message.content


def extract_questions_with_progress(content):
    """带进度显示的问题提取"""
    # 使用正则表达式提取所有问题
    questions = re.findall(r'Q\d+:(.*?)(?=Q\d+:|$)', content, re.DOTALL)

    # 进度条初始化
    progress = tqdm(questions,
                    desc="解析作战系统问题",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                    ncols=80)

    processed_questions = []
    for q in progress:
        processed_questions.append(q.strip())
        # 实时更新进度说明
        progress.set_postfix({"当前处理": f"Q{len(processed_questions)}"})

    return processed_questions


def save_questions_with_progress(questions):
    """带进度条的问题保存"""
    # 问题文件保存
    with open("mil_questions.txt", "w", encoding="utf-8") as f:
        for idx, q in enumerate(tqdm(questions, desc="写入战术问题库", unit="Q")):
            f.write(f"Q{idx + 1}: {q}\n\n")


if __name__ == "__main__":
    try:
        print("🛡️ 启动联合作战系统知识库生成程序")
        content = generate_questions()

        print("\n🔍 解析生成内容...")
        questions = extract_questions_with_progress(content)

        print("\n💾 持久化作战问题...")
        save_questions_with_progress(questions)

        print(f"\n✅ 成功生成并保存了 {len(questions)} 个军事指挥系统运维问题")

    except Exception as e:
        print(f"\n❌ 系统告警：{str(e)}")
        # 记录错误日志
        with open("combat_system_error.log", "a") as log:
            log.write(f"[{datetime.now()}] {str(e)}\n")
# import re
# from datetime import datetime
#
# from openai import OpenAI
# from tqdm import tqdm
#
# # 配置API客户端
# client = OpenAI(
#     api_key="22e0a307-fcf4-4029-b697-d42cde4b1733",
#     base_url="https://ark.cn-beijing.volces.com/api/v3",
# )
#
#
# def generate_qa():
#     """生成QA对的增强版本，包含军事规范验证"""
#     prompt = """
#     生成30个军事指挥系统运维QA对，要求：
#     1. 包含以下标准：MIL-STD-810H（环境测试）、MIL-STD-461G（电磁兼容）
#     2. 问题类型分布：硬件(40%)、软件(30%)、网络(20%)、数据(10%)
#     3. 输出格式：
#         Q1: [问题正文]
#     """
#
#     response = client.chat.completions.create(
#         model="ep-20250218175517-6jc9n",
#         messages=[
#             {"role": "system", "content": "您现在是军委装备发展部的技术规范生成系统"},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.8,
#         max_tokens=3500,
#         top_p=0.95
#     )
#     return response.choices[0].message.content  # 修正属性访问方式
#
#
# def split_qa_with_progress(content):
#     """带进度显示的QA解析"""
#     qa_pairs = re.findall(r'(Q\d+:.*?)(?=Q\d+:|$)', content, re.DOTALL)
#
#     # 进度条初始化
#     progress = tqdm(qa_pairs,
#                     desc="解析作战系统QA对",
#                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
#                     ncols=80)
#
#     questions = []
#
#
#     for qa in progress:
#         # 分离问题和答案
#         q, a = re.split(r'A\d+:', qa, flags=re.IGNORECASE)
#         questions.append(q.strip())
#         # 实时更新进度说明
#         progress.set_postfix({"当前处理": f"Q{len(questions)}"})
#
#     return questions
#
# def save_with_progress(questions, answers):
#     """带双重进度条的文件保存"""
#     # 问题文件保存
#     with open("mil_questions.txt", "w", encoding="utf-8") as f:
#         for idx, q in enumerate(tqdm(questions, desc="写入战术问题库", unit="Q")):
#             f.write(f"Q{idx + 1}: {q}\n\n")
#
#
# if __name__ == "__main__":
#     try:
#         print("🛡️ 启动联合作战系统知识库生成程序")
#         qa_content = generate_qa()
#
#         print("\n🔍 解析生成内容...")
#         questions, answers = split_qa_with_progress(qa_content)
#
#         print("\n💾 持久化作战数据...")
#         save_with_progress(questions, answers)
#
#
#     except Exception as e:
#         print(f"\n❌ 系统告警：{str(e)}")
#         # 记录错误日志
#         with open("combat_system_error.log", "a") as log:
#             log.write(f"[{datetime.now()}] {str(e)}\n")
