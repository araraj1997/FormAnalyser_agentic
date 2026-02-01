"""
Document Processor Module

Handles raw document loading and text extraction from various formats.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExtractedDocument:
    """Represents raw extracted content from a document."""
    file_path: str
    file_type: str
    raw_text: str
    pages: List[str] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    images: List[bytes] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "unknown"
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DocumentProcessor:
    """
    Processes various document formats to extract raw text and structure.
    
    This is the first stage - just extraction, no intelligence yet.
    The LLM will handle understanding the content.
    """
    
    def __init__(self, ocr_enabled: bool = True, ocr_language: str = "eng"):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self._pdfplumber = None
        self._pytesseract = None
        self._Image = None
    
    def process(self, file_path: str) -> ExtractedDocument:
        """
        Process a document and extract its content.
        
        Args:
            file_path: Path to the document
            
        Returns:
            ExtractedDocument with raw content
        """
        file_path = str(Path(file_path).resolve())
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_type = self._detect_file_type(file_path)
        
        if file_type == "pdf":
            return self._process_pdf(file_path)
        elif file_type in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            return self._process_image(file_path)
        else:
            return self._process_text(file_path)
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension or magic bytes."""
        ext = Path(file_path).suffix.lower()
        
        type_map = {
            ".pdf": "pdf",
            ".png": "png", ".jpg": "jpg", ".jpeg": "jpeg",
            ".tiff": "tiff", ".tif": "tiff", ".bmp": "bmp",
            ".txt": "txt", ".text": "txt",
            ".json": "json", ".csv": "csv",
            ".html": "html", ".htm": "html",
            ".md": "markdown", ".markdown": "markdown"
        }
        
        return type_map.get(ext, "txt")
    
    def _process_pdf(self, file_path: str) -> ExtractedDocument:
        """Extract content from PDF."""
        if self._pdfplumber is None:
            try:
                import pdfplumber
                self._pdfplumber = pdfplumber
            except ImportError:
                raise ImportError("pdfplumber required: pip install pdfplumber")
        
        pages = []
        tables = []
        all_text = ""
        metadata = {}
        
        try:
            with self._pdfplumber.open(file_path) as pdf:
                metadata = {
                    "page_count": len(pdf.pages),
                    "pdf_info": pdf.metadata or {}
                }
                
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    pages.append(page_text)
                    all_text += f"\n--- Page {i+1} ---\n{page_text}"
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            cleaned = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                            tables.append(cleaned)
                
                # If no text, try OCR
                if not all_text.strip() and self.ocr_enabled:
                    all_text, pages = self._ocr_pdf(file_path)
                    metadata["ocr_used"] = True
        
        except Exception as e:
            metadata["error"] = str(e)
            if self.ocr_enabled:
                try:
                    all_text, pages = self._ocr_pdf(file_path)
                    metadata["ocr_used"] = True
                except:
                    pass
        
        return ExtractedDocument(
            file_path=file_path,
            file_type="pdf",
            raw_text=all_text.strip(),
            pages=pages,
            tables=tables,
            metadata=metadata,
            extraction_method="pdfplumber" if not metadata.get("ocr_used") else "ocr"
        )
    
    def _ocr_pdf(self, file_path: str) -> Tuple[str, List[str]]:
        """OCR a PDF document."""
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError:
            return "", []
        
        pages = []
        all_text = ""
        
        images = convert_from_path(file_path)
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang=self.ocr_language)
            pages.append(page_text)
            all_text += f"\n--- Page {i+1} (OCR) ---\n{page_text}"
        
        return all_text, pages
    
    def _process_image(self, file_path: str) -> ExtractedDocument:
        """Extract text from image using OCR."""
        if not self.ocr_enabled:
            return ExtractedDocument(
                file_path=file_path,
                file_type=self._detect_file_type(file_path),
                raw_text="",
                metadata={"error": "OCR disabled"}
            )
        
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            raise ImportError("Pillow and pytesseract required for image processing")
        
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang=self.ocr_language)
        
        # Get OCR confidence data
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 0
        
        return ExtractedDocument(
            file_path=file_path,
            file_type=self._detect_file_type(file_path),
            raw_text=text.strip(),
            pages=[text],
            metadata={
                "image_size": img.size,
                "image_mode": img.mode,
                "ocr_confidence": avg_confidence
            },
            extraction_method="ocr"
        )
    
    def _process_text(self, file_path: str) -> ExtractedDocument:
        """Process text-based files."""
        file_type = self._detect_file_type(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        
        tables = []
        
        # Extract tables from CSV
        if file_type == "csv":
            import csv
            from io import StringIO
            reader = csv.reader(StringIO(content))
            tables.append([row for row in reader])
        
        # Extract tables from JSON arrays
        elif file_type == "json":
            import json
            try:
                data = json.loads(content)
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    headers = list(data[0].keys())
                    table = [headers] + [[str(row.get(h, "")) for h in headers] for row in data]
                    tables.append(table)
            except:
                pass
        
        return ExtractedDocument(
            file_path=file_path,
            file_type=file_type,
            raw_text=content.strip(),
            pages=[content],
            tables=tables,
            metadata={"char_count": len(content), "line_count": content.count("\n") + 1},
            extraction_method="text"
        )
