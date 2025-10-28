"""전역 설정 모듈"""
from pathlib import Path

class Settings:
    """애플리케이션 설정 클래스"""
    
    # ==================== 환경 설정 ====================
    ENV: str = "development"
    
    # ==================== API 설정 ====================
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_TITLE: str = "반도체 비즈니스 문서 QA API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "PDF 문서 기반 질의응답 시스템"
    
    # ==================== 모델 설정 ====================
    # LLM 모델 (아래 중 선택)
    # - "Qwen/Qwen2.5-0.5B-Instruct" (가장 빠름)
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (빠름)
    # - "microsoft/Phi-3-mini-4k-instruct" (균형)
    # - "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct" (한국어 최적화)
    LLM_MODEL: str = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    
    # 임베딩 모델
    EMBEDDING_MODEL: str = "dragonkue/multilingual-e5-small-ko"
    
    # ==================== 경로 설정 ====================
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    PDF_DOCUMENTS_PATH: Path = BASE_DIR / "data" / "pdf_documents"
    VECTOR_DB_PATH: Path = BASE_DIR / "data" / "vector_db"
    LOG_PATH: Path = BASE_DIR / "logs"
    
    # ==================== LLM 생성 설정 ====================
    MAX_NEW_TOKENS: int = 50        # 생성할 최대 토큰 수 (15-50 권장)
    TEMPERATURE: float = 0.1        # 낮을수록 일관된 답변 (0.0-1.0)
    
    # ==================== RAG 설정 ====================
    TOP_K: int = 3                  # 검색할 문서 수 (1-5)
    CHUNK_SIZE: int = 300           # 문서 청크 크기 (200-500)
    CHUNK_OVERLAP: int = 30         # 청크 오버랩 (10-50)
    
    # ==================== 로그 설정 ====================
    LOG_LEVEL: str = "INFO"         # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: str = "app.log"
    
    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리 생성"""
        cls.PDF_DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        cls.LOG_PATH.mkdir(parents=True, exist_ok=True)

# 디렉토리 생성
Settings.ensure_directories()

# 싱글톤 인스턴스
settings = Settings()
