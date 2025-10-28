"""API 요청/응답 스키마"""
from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionRequest(BaseModel):
    """질문 요청 스키마"""
    question: str = Field(
        ..., 
        min_length=1,
        max_length=500,
        description="사내 문서에 대한 질문",
        examples=["엔비디아와의 반도체 수출 계약 금액은?"]
    )

# answer만 반환하는 간단한 스키마
class AnswerResponse(BaseModel):
    """답변 응답 스키마 (answer만)"""
    answer: str = Field(..., description="답변")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "총 반도체 수출액: 980억 달러"
                }
            ]
        }
    }

# 선택사항: 상세 정보가 필요한 경우를 위한 스키마
class DetailedAnswerResponse(BaseModel):
    """답변 응답 스키마 (상세)"""
    question: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="답변")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    sources: List[str] = Field(default_factory=list, description="출처")

class UploadResponse(BaseModel):
    """PDF 업로드 응답 스키마"""
    message: str = Field(..., description="응답 메시지")
    filename: str = Field(..., description="업로드된 파일명")
    chunks_created: int = Field(..., ge=0, description="생성된 청크 수")
    status: str = Field(..., description="처리 상태")

class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    models_loaded: bool = Field(..., description="모델 로드 상태")

class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    error: str = Field(..., description="에러 타입")
    message: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 정보")
