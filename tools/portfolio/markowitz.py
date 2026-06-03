import json
import numpy as np
from scipy.optimize import minimize
from langchain_core.tools import tool
from tools.state import portfolio_data


@tool
def markowitz_optimize(risk_free_rate: float = 0.02) -> str:
    '''对已下载的股票数据进行马科维茨投资组合优化。

    同时求解两种组合：
    - 最大夏普比率组合（风险调整后收益最优，适合积极型投资者）
    - 最小方差组合（组合波动率最低，适合保守型投资者）

    请先调用 download_stock_data 下载数据后再使用此工具。

    Args:
        risk_free_rate: 年化无风险利率，默认 0.02（2%）。美股建议 0.045。

    Returns:
        两种最优组合的权重及风险收益指标（JSON 字符串）
    '''
    if 'returns' not in portfolio_data:
        return '错误：请先调用 download_stock_data 下载数据。'

    returns_df = portfolio_data['returns']
    stocks     = portfolio_data['stocks']
    n          = len(stocks)

    if returns_df.empty or len(returns_df) < 5 or n == 0:
        return '错误：数据为空或不足，无法进行优化。请检查股票代码和日期范围后重试。'

    mean_ret = returns_df.mean().values
    cov_mat  = returns_df.cov().values * 252

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
