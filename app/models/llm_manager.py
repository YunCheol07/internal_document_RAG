"""LLM 모델 관리 모듈 (답변 형식 개선)"""
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import log

class LLMManager:
    """LLM 모델 관리 클래스"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.device}")
        
        self.llm_pipeline = None
        self.embedding_model = None
        
        self._initialized = True
    
    def load_llm(self):
        """LLM 파이프라인 로드 (CPU 최적화)"""
        if self.llm_pipeline is not None:
            log.info("LLM pipeline already loaded")
            return self.llm_pipeline
        
        log.info(f"Loading LLM model: {settings.LLM_MODEL}")
        
        try:
            log.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL,
                trust_remote_code=True
            )
            
            # CPU 최적화
            if self.device == "cpu":
                log.info("Loading model optimized for CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    settings.LLM_MODEL,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model.eval()
            else:
                # GPU 최적화
                model = AutoModelForCausalLM.from_pretrained(
                    settings.LLM_MODEL,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            
            log.info("Creating pipeline...")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
            )
            
            if self.llm_pipeline is None:
                raise RuntimeError("Pipeline creation returned None")
            
            log.info(f"LLM model loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load LLM model: {e}")
            self.llm_pipeline = None
            raise
        
        return self.llm_pipeline
    
    def load_embeddings(self):
        """임베딩 모델 로드"""
        if self.embedding_model is not None:
            log.info("Embedding model already loaded")
            return self.embedding_model
        
        log.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        try:
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=self.device,
                trust_remote_code=True
            )
            
            if self.embedding_model is None:
                raise RuntimeError("SentenceTransformer creation returned None")
            
            log.info(f"Embedding model loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            raise
        
        return self.embedding_model
    
    def generate_answer(self, prompt: str) -> str:
        """답변 생성 (값 검증 강화)"""
        if self.llm_pipeline is None:
            log.warning("LLM pipeline not loaded, loading now...")
            self.load_llm()
        
        if self.llm_pipeline is None:
            raise RuntimeError("Failed to load LLM pipeline")
        
        log.info("Generating answer with LLM...")
        
        try:
            with torch.no_grad():
                result = self.llm_pipeline(
                    prompt,
                    max_new_tokens=settings.MAX_NEW_TOKENS,
                    temperature=settings.TEMPERATURE,
                    do_sample=settings.TEMPERATURE > 0,  # Temperature 0이면 샘플링 안 함
                    num_return_sequences=1,
                    return_full_text=True,
                    clean_up_tokenization_spaces=True,
                    pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
                )
            
            if not result or len(result) == 0:
                raise ValueError("LLM returned empty result")
            
            generated_text = result[0]['generated_text']
            
            # "답변:" 이후 텍스트 추출
            if "답변:" in generated_text:
                answer = generated_text.split("답변:")[-1].strip()
            else:
                if len(generated_text) > len(prompt):
                    answer = generated_text[len(prompt):].strip()
                else:
                    answer = generated_text.strip()
            
            # 답변 정제
            answer = self._clean_answer(answer)
            
            # 값이 있는지 검증
            if not self._has_value(answer):
                log.warning(f"Answer lacks value: {answer}")
                # 재생성 시도 (한 번만)
                return answer + " (값 정보 없음)"
            
            # 빈 답변 방지
            if not answer or len(answer) < 5:
                answer = "정보를 찾을 수 없습니다."
            
            log.info(f"Generated answer: {answer}")
            return answer
            
        except Exception as e:
            log.error(f"Failed to generate answer: {e}", exc_info=True)
            raise

    def _has_value(self, answer: str) -> bool:
        """답변에 값이 포함되어 있는지 검증
        
        Args:
            answer: 생성된 답변
            
        Returns:
            값이 있으면 True
        """
        # 숫자가 포함되어 있는지 체크
        has_number = any(char.isdigit() for char in answer)
        
        # 단위가 포함되어 있는지 체크
        units = ['억', '만', '달러', '원', '%', '대', '개', 'V', 'GB', 'TB']
        has_unit = any(unit in answer for unit in units)
        
        # 콜론이 있고 그 뒤에 값이 있는지 체크
        if ':' in answer:
            value_part = answer.split(':')[-1].strip()
            return len(value_part) > 0 and (has_number or has_unit)
        
        # 숫자와 단위가 있으면 OK
        return has_number and has_unit

    def _clean_answer(self, answer: str) -> str:
        """답변 정제"""
        
        # 줄 분리
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        # 첫 번째 줄만 사용
        answer = lines[0]
        
        # 불필요한 접두사 제거
        prefixes_to_remove = [
            '답변:', 'Answer:', 'A:', '정답:', '응답:',
            '문서에 따르면', '문서에서는', '위 문서에 따르면',
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # 하이픈/불릿 제거
        if answer.startswith('- '):
            answer = answer[2:].strip()
        
        # "항목:" 만 있고 값이 없는 경우 처리
        if answer.endswith(':') or answer.endswith('：'):
            log.warning(f"Answer ends with colon: {answer}")
            return answer  # 일단 반환 (검증에서 걸러짐)
        
        # 너무 짧은 경우 (10자 미만)
        if len(answer) < 10:
            # 첫 줄이 너무 짧으면 다음 줄도 포함
            if len(lines) > 1:
                answer = ' '.join(lines[:2])
        
        return answer
    
    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.llm_pipeline is not None and self.embedding_model is not None

# 싱글톤 인스턴스
llm_manager = LLMManager()
