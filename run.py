"""애플리케이션 실행 스크립트"""
import uvicorn
from app.core.config import Settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=Settings.API_HOST,
        port=Settings.API_PORT,
        reload=True if Settings.ENV == "development" else False,
        log_level=Settings.LOG_LEVEL.lower()
    )
