"""FastAPI 메인 애플리케이션"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import upload, qa
from app.core.config import settings
from app.core.logging import log
from app.schemas.api_schemas import HealthResponse
from app.models.llm_manager import llm_manager

# FastAPI 앱 초기화
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(upload.router, prefix="/api/v1")
app.include_router(qa.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 이벤트"""
    log.info("=" * 60)
    log.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    log.info("=" * 60)
    
    # 모델 로드 (필수)
    try:
        log.info("Loading models (this may take a few minutes)...")
        
        # LLM 로드
        log.info("Step 1/2: Loading LLM...")
        llm = llm_manager.load_llm()
        if llm is None:
            raise RuntimeError("LLM pipeline is None after loading")
        log.info("✓ LLM loaded successfully")
        
        # 임베딩 모델 로드
        log.info("Step 2/2: Loading embeddings...")
        embeddings = llm_manager.load_embeddings()
        if embeddings is None:
            raise RuntimeError("Embedding model is None after loading")
        log.info("✓ Embeddings loaded successfully")
        
        log.info("=" * 60)
        log.info("All models loaded successfully - Server is ready!")
        log.info("=" * 60)
        
    except Exception as e:
        log.error("=" * 60)
        log.error(f"CRITICAL ERROR: Failed to load models")
        log.error(f"Error: {e}")
        log.error("Server cannot start without models - exiting...")
        log.error("=" * 60)
        import sys
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 이벤트"""
    log.info("Shutting down application")

@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "models_loaded": llm_manager.is_loaded()
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthResponse(
        status="healthy" if llm_manager.is_loaded() else "unhealthy",
        version=settings.API_VERSION,
        models_loaded=llm_manager.is_loaded()
    )
