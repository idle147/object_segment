# logger_setup.py
import sys

from loguru import logger


def logger_setup():
    # 配置 Loguru 日志，只执行一次
    logger.remove()  # 移除默认的控制台输出（如果需要自定义控制台输出，可以移除这行）

    # 添加文件日志处理器
    logger.add(
        "logs/app.log",
        rotation="10 MB",  # 当日志文件达到 10MB 时自动分割
        retention="10 days",  # 保留最近 10 天的日志
        compression="zip",  # 压缩旧日志
        level="INFO",  # 日志级别
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    )

    # 添加控制台日志处理器
    logger.add(
        sys.stdout,  # 将日志输出到控制台
        level="INFO",  # 日志级别
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        colorize=True,  # 启用彩色输出，提高可读性
    )
