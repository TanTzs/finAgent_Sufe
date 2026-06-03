# 工具间共享的运行时状态
# download_stock_data 写入，markowitz_optimize / 未来其他工具读取
portfolio_data: dict = {}
