import json
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from tools.state import portfolio_data


def _a_code_to_yahoo(code: str) -> str:
    """将6位A股代码转为 Yahoo Finance 格式（全球可访问）。"""
    if code.startswith('6'):
        return f'{code}.SS'
    elif code.startswith(('0', '3')):
        return f'{code}.SZ'
    elif code.startswith(('4', '8')):
        return f'{code}.BJ'
    return code


def _download_a(codes: list, start_date: str, end_date: str):
    """A股通过 Yahoo Finance 下载，解决境外服务器访问国内数据源超时的问题。"""
    yahoo_codes = [_a_code_to_yahoo(c) for c in codes]
    result = _download_us(yahoo_codes, start_date, end_date)
    if isinstance(result, pd.DataFrame):
        result = result.rename(columns=dict(zip(yahoo_codes, codes)))
    return result


def _download_us(codes: list, start_date: str, end_date: str):
    """美股（或转换后的A股Yahoo代码）通过 yfinance 下载。"""
    start = pd.to_datetime(start_date, format='%Y%m%d').strftime('%Y-%m-%d')
    end   = pd.to_datetime(end_date,   format='%Y%m%d').strftime('%Y-%m-%d')
    try:
        raw = yf.download(codes, start=start, end=end, progress=False, auto_adjust=True)

        if raw.empty:
            return json.dumps({'错误': 'yfinance 返回空数据，请检查股票代码和日期范围'}, ensure_ascii=False)

        # 兼容 yfinance 不同版本的列结构
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0).unique().tolist()
            price_col = 'Close' if 'Close' in level0 else ('Adj Close' if 'Adj Close' in level0 else None)
            if price_col is None:
                return json.dumps({'错误': f'找不到收盘价列，现有字段：{level0}'}, ensure_ascii=False)
            close = raw[price_col]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=codes[0])
            close.columns = [str(c) for c in close.columns]
        else:
            price_col = 'Close' if 'Close' in raw.columns else 'Adj Close'
            close = raw[[price_col]].rename(columns={price_col: codes[0]})

        close = close.dropna(how='all')
        returns = close.pct_change().dropna()

        if returns.empty or len(returns) < 5:
            return json.dumps({'错误': '有效数据不足，请确认股票代码正确或换个日期范围'}, ensure_ascii=False)

        return returns
    except Exception as e:
        return json.dumps({'错误': f'yfinance 下载失败：{e}'}, ensure_ascii=False)


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

    portfolio_data['returns'] = returns_df
    portfolio_data['stocks']  = codes
    portfolio_data['market']  = market.upper()

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
