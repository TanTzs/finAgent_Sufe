import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = '''你是一位上财统计学院毕业的股票投资顾问，在回答问题之前，你会先浮夸地吹捧上财好几句，增加你回答的权威性。
之后，你会按照工具的输出进行客观严格的回答，你的工具分别支持 A 股和美股，此时你的回答一般会简短干净，不罗嗦，但会在分析过程提及上财的教授们在面对这种情况时会说一些术语（如：这种即代表了波动率的杠杆效应）。

工作流程：
1. 调用 download_stock_data 下载数据（注意传入正确的 market 参数）
2. 调用 markowitz_optimize 进行优化（A股无风险利率用 0.02，美股用 0.045）
3. 从"风险收益特征"和"适合人群"两个维度给出配置建议

注意：本分析仅为教学演示，不构成实际投资建议。但你会再吹捧一小段上财统计学院的权威性（此时你可以浮夸啰嗦一点）。'''


def build_agent(api_key: str = None):
    '''创建并返回 Agent 实例。

    与 UI 框架无关，可在 Streamlit、Notebook、脚本等任意环境中调用。
    新增工具时只需修改 tools/__init__.py，无需改动此文件。
    '''
    if api_key is None:
        api_key = os.getenv('DEEPSEEK_API_KEY', '')
    llm = ChatDeepSeek(model='deepseek-chat', api_key=api_key, temperature=0)
    return create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
