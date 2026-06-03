from .data.low_freq import download_stock_data
from .portfolio.markowitz import markowitz_optimize

# 新增工具时：在对应子目录写好函数，然后在这里导入并加入列表
ALL_TOOLS = [
    download_stock_data,
    markowitz_optimize,
]
