import os
import json
import numpy as np
import pandas as pd
import akshare as ak
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent

load_dotenv()

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="马科维茨投资组合优化智能体",
    page_icon="📈",
    layout="wide",
)

# ── 工具（与 notebook 保持一致） ──────────────────────────────────────────────
_portfolio_data: dict = {}

@tool
def download_stock_data(
    stock_codes: str,
    market: str = 'A',
    start_date: str = '20240101',
    end_date: str = '20241231',
) -> str:
    '''下载股票历史行情数据并计算日收益率，支持 A 股和美股。

    Args:
        stock_codes: 股票代码，多只用英文逗号分隔。
                     A股示例：000001,600519,300750
                     美股示例：AAPL,MSFT,GOOGL
        market: 市场选择，A 表示 A 股，US 表示美股，默认 A
        start_date: 起始日期，格式 YYYYMMDD，默认 20240101
        end_date: 结束日期，格式 YYYYMMDD，默认 20241231

    Returns:
        数据下载状态及基本统计摘要（JSON 字符串）
    '''
    codes = [c.strip() for c in stock_codes.split(',')]
    returns_df = _download_us(codes, start_date, end_date) if market.upper() == 'US' \
                 else _download_a(codes, start_date, end_date)

    if isinstance(returns_df, str):
        return returns_df

    _portfolio_data['returns'] = returns_df
    _portfolio_data['stocks'] = codes
    _portfolio_data['market'] = market.upper()

    summary = {
        '状态': '下载成功',
        '市场': 'A股' if market.upper() == 'A' else '美股',
        '股票代码': codes,
        '有效交易日数': len(returns_df),
        '日期范围': f'{str(returns_df.index[0])} 至 {str(returns_df.index[-1])}',
        '各股年化收益率（估算）': {
            code: f'{returns_df[code].mean() * 252:.2%}' for code in codes
        },
        '提示': '数据已保存，可调用 markowitz_optimize 进行优化',
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _download_a(codes, start_date, end_date):
    dfs = {}
    for code in codes:
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period='daily',
                start_date=start_date, end_date=end_date, adjust='qfq',
            )
            dfs[code] = df.set_index('日期')['涨跌幅'] / 100
        except Exception as e:
            return json.dumps({'错误': f'下载 {code} 失败：{e}'}, ensure_ascii=False)
    return pd.DataFrame(dfs).dropna()


def _download_us(codes, start_date, end_date):
    start = pd.to_datetime(start_date, format='%Y%m%d').strftime('%Y-%m-%d')
    end   = pd.to_datetime(end_date,   format='%Y%m%d').strftime('%Y-%m-%d')
    try:
        if len(codes) == 1:
            raw   = yf.download(codes[0], start=start, end=end, progress=False, auto_adjust=True)
            close = raw[['Close']].rename(columns={'Close': codes[0]})
        else:
            raw   = yf.download(codes, start=start, end=end, progress=False, auto_adjust=True)
            close = raw['Close']
            close.columns = [str(c) for c in close.columns]
        return close.pct_change().dropna()
    except Exception as e:
        return json.dumps({'错误': f'yfinance 下载失败：{e}'}, ensure_ascii=False)


@tool
def markowitz_optimize(risk_free_rate: float = 0.02) -> str:
    '''对已下载的股票数据进行马科维茨投资组合优化。

    同时求解最大夏普比率组合和最小方差组合。
    请先调用 download_stock_data 下载数据后再使用此工具。

    Args:
        risk_free_rate: 年化无风险利率，默认 0.02。美股建议 0.045。

    Returns:
        两种最优组合的权重及风险收益指标（JSON 字符串）
    '''
    if 'returns' not in _portfolio_data:
        return '错误：请先调用 download_stock_data 下载数据。'

    returns_df = _portfolio_data['returns']
    stocks     = _portfolio_data['stocks']
    n          = len(stocks)
    mean_ret   = returns_df.mean().values
    cov_mat    = returns_df.cov().values * 252

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds      = [(0.0, 1.0)] * n
    w0          = np.ones(n) / n

    def neg_sharpe(w):
        ret = float(np.dot(w, mean_ret) * 252)
        vol = float(np.sqrt(w @ cov_mat @ w))
        return -(ret - risk_free_rate) / vol

    def port_vol(w):
        return float(np.sqrt(w @ cov_mat @ w))

    res_sr = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    res_mv = minimize(port_vol,   w0, method='SLSQP', bounds=bounds, constraints=constraints)

    def stats(w):
        ret = float(np.dot(w, mean_ret) * 252)
        vol = float(np.sqrt(w @ cov_mat @ w))
        return ret, vol, (ret - risk_free_rate) / vol

    sr_ret, sr_vol, sr_sharpe = stats(res_sr.x)
    mv_ret, mv_vol, mv_sharpe = stats(res_mv.x)

    result = {
        '最大夏普比率组合（激进型）': {
            '权重配置': {stocks[i]: f'{res_sr.x[i]:.2%}' for i in range(n)},
            '年化预期收益': f'{sr_ret:.2%}',
            '年化波动率（风险）': f'{sr_vol:.2%}',
            '夏普比率': round(sr_sharpe, 4),
        },
        '最小方差组合（保守型）': {
            '权重配置': {stocks[i]: f'{res_mv.x[i]:.2%}' for i in range(n)},
            '年化预期收益': f'{mv_ret:.2%}',
            '年化波动率（风险）': f'{mv_vol:.2%}',
            '夏普比率': round(mv_sharpe, 4),
        },
        '参数说明': f'无风险利率 {risk_free_rate:.1%}，基于 {len(returns_df)} 个交易日历史数据',
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Agent（缓存，避免每次点击都重新初始化） ───────────────────────────────────
@st.cache_resource
def get_agent():
    api_key = os.getenv('DEEPSEEK_API_KEY') or st.secrets.get('DEEPSEEK_API_KEY', '')
    llm = ChatDeepSeek(model='deepseek-chat', api_key=api_key, temperature=0)
    system_prompt = '''你是一位上财毕业的股票投资顾问，回答问题时会按照工具的输出进行客观严格的回答，并且喜欢吹捧上财，擅长运用马科维茨现代投资组合理论进行资产配置分析，支持 A 股和美股。你的回答一般会简短干净，不罗嗦。

工作流程：
1. 调用 download_stock_data 下载数据（注意传入正确的 market 参数）
2. 调用 markowitz_optimize 进行优化（A股无风险利率用 0.02，美股用 0.045）
3. 从"风险收益特征"和"适合人群"两个维度给出配置建议

注意：本分析仅为教学演示，不构成实际投资建议。'''
    return create_agent(model=llm, tools=[download_stock_data, markowitz_optimize],
                        system_prompt=system_prompt)


# ── 页面 UI ───────────────────────────────────────────────────────────────────
st.title("📈 马科维茨投资组合优化智能体")
st.caption("课程《金融智能体设计》教学演示 | 仅供学习，不构成投资建议")

# ── 侧边栏：示例 + 清除 ───────────────────────────────────────────────────────
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

# ── 对话历史初始化 ────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ── 渲染历史消息 ──────────────────────────────────────────────────────────────
for msg in st.session_state['history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ── 处理"示例按钮"触发的预填问题 ──────────────────────────────────────────────
pending = st.session_state.pop('pending', None)

# ── 聊天输入框 ────────────────────────────────────────────────────────────────
user_input = st.chat_input("请描述您想分析的投资组合，例如：分析苹果、微软、英伟达 2024 年的组合")
prompt = pending or user_input

if prompt:
    # 显示用户消息
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state['history'].append({'role': 'user', 'content': prompt})

    # 调用智能体
    with st.chat_message('assistant'):
        with st.spinner('智能体分析中，请稍候（约 20–40 秒）…'):
            try:
                agent  = get_agent()
                result = agent.invoke({'messages': [HumanMessage(content=prompt)]})
                answer = result['messages'][-1].content
            except Exception as e:
                answer = f'分析出错：{e}'
        st.markdown(answer)

    st.session_state['history'].append({'role': 'assistant', 'content': answer})
