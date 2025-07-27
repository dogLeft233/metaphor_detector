"""
日志处理器模块

该模块提供了与tqdm进度条兼容的日志处理器，确保日志输出不会干扰进度条的显示。

主要功能：
- TqdmLoggingHandler: 将logging输出重定向到tqdm.write，避免与进度条冲突
"""

import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """
    自定义日志处理器，将logging输出重定向到tqdm.write
    
    这个处理器解决了在使用tqdm进度条时，logging输出会破坏进度条显示的问题。
    通过使用tqdm.write()方法，确保日志消息能够正确显示而不会干扰进度条。
    
    Attributes:
        level: 日志级别，默认为NOTSET
    """
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        """
        发送日志记录到tqdm.write
        
        重写父类的emit方法，使用tqdm.write()输出日志消息，
        确保与进度条显示兼容。
        
        Args:
            record: 日志记录对象
        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)