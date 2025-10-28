"""PDF 문서 처리 서비스"""
import pypdf
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logging import log

class PDFProcessor:
    """PDF 문서 처리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        log.info(f"Extracting text from PDF: {pdf_path.name}")
        
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                log.info(f"Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    log.debug(f"Extracted page {page_num}/{total_pages}")
            
            log.info(f"Total text length: {len(text)} characters")
            
        except Exception as e:
            log.error(f"Failed to extract text from PDF: {e}")
            raise
        
        return text
    
    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """PDF를 청크로 분할
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Document 객체 리스트
        """
        log.info(f"Processing PDF: {pdf_path.name}")
        
        # 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError(f"No text extracted from PDF: {pdf_path.name}")
        
        # Document 객체 생성
        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "file_path": str(pdf_path)
            }
        )
        
        # 청크로 분할
        chunks = self.text_splitter.split_documents([doc])
        
        log.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
        
        return chunks

# 싱글톤 인스턴스
pdf_processor = PDFProcessor()
