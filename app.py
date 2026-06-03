import os
import streamlit as st
from dotenv import load_dotenv
from datetime import date
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent import build_agent

load_dotenv()

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="上财统院-投资组合优化智能体",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def get_agent():
    api_key = os.getenv('DEEPSEEK_API_KEY') or st.secrets.get('DEEPSEEK_API_KEY', '')
    return build_agent(api_key)


# ── 侧边栏 ────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "帮我分析苹果(AAPL)、微软(MSFT)、英伟达(NVDA) 2024年的美股投资组合，给出配置建议",
    "分析平安银行(000001)、贵州茅台(600519)、宁德时代(300750) 2023年A股组合",
    "AAPL、GOOGL、META、AMZN 四只科技股2024年怎么配置最优？",
    "帮我看看招商银行(600036)和中国平安(601318)两只股票2024年的最优组合",
]

with st.sidebar:
    st.header("💡 示例问题")
    st.caption("点击可直接发送")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state['pending'] = ex
            st.rerun()
    st.divider()
    if st.button("🗑️ 清除对话", use_container_width=True):
        st.session_state['history'] = []
        st.rerun()
    st.divider()
    st.markdown("**使用提示**")
    st.markdown("- A股：输入6位股票代码或股票名称\n- 美股：输入 Ticker 符号（如 AAPL）\n- 可直接用自然语言描述，无需填表")

# ── 主区域 ────────────────────────────────────────────────────────────────────
st.title("📈 上财统院-投资组合优化智能体")
st.caption("上财统计课程《金融智能体设计》教学演示 | 仅供学习，不构成投资建议")

if 'history' not in st.session_state:
    st.session_state['history'] = []

for msg in st.session_state['history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

pending    = st.session_state.pop('pending', None)
user_input = st.chat_input("请描述您想分析的投资组合，例如：分析苹果、微软、英伟达 2024 年的组合")
prompt     = pending or user_input

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state['history'].append({'role': 'user', 'content': prompt})

    with st.chat_message('assistant'):
        with st.spinner('上财的专业智能体分析中，请稍候（约 20–40 秒）…'):
            try:
                agent = get_agent()
                history_messages = [
                    SystemMessage(content=f"今天是 {date.today().strftime('%Y年%m月%d日')}。")
                ]
                for msg in st.session_state['history'][:-1]:
                    if msg['role'] == 'user':
                        history_messages.append(HumanMessage(content=msg['content']))
                    else:
                        history_messages.append(AIMessage(content=msg['content']))
                history_messages.append(HumanMessage(content=prompt))
                result = agent.invoke({'messages': history_messages})
                answer = result['messages'][-1].content
            except Exception as e:
                answer = f'分析出错：{e}'
        st.markdown(answer)

    st.session_state['history'].append({'role': 'assistant', 'content': answer})
