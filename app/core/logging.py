"""로깅 설정 모듈"""
import sys
from loguru import logger
from app.core.config import settings

def setup_logging():
    """로깅 설정 초기화"""
    
    # 기본 로거 제거
    logger.remove()
    
    # 콘솔 로거 추가
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL
    )
    
    # 파일 로거 추가
    logger.add(
        settings.LOG_PATH / settings.LOG_FILE,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=settings.LOG_LEVEL
    )
    
    return logger

# 로거 초기화
log = setup_logging()
