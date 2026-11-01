'第二问各计算入口共用的默认参数。每次搜索创建独立的 random.Random(DEFAULT_SEED)，HC、多步搜索和 MGR 运行时不重新设置模块级随机数生成器。'

# 命令行、接口、批量工作进程和搜索共同使用选定的参数。
DEFAULT_SEED = 1234
DEFAULT_ESCAPE_WEIGHT = 2.0
