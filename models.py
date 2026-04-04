from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class Table:
    table_id: str
    pages: List[int]
    headers: List[str]
    rows: List[List[str]]
    footnotes: List[str] = field(default_factory=list)
    caption: Optional[str] = None
    related_formulas: List[str] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps({
            "table_id": self.table_id,
            "pages": self.pages,
            "headers": self.headers,
            "rows": self.rows[:10],  # Limit rows for embedding
            "footnotes": self.footnotes
        })
    
    def to_natural_language(self) -> str:
        """Convert table to natural language description"""
        nl = f"Table {self.table_id} on pages {self.pages[0]}-{self.pages[-1]}: "
        if self.caption:
            nl += f"{self.caption}. "
        
        nl += f"Headers: {', '.join(self.headers)}. "
        
        # Add sample rows
        sample_rows = self.rows[:3]
        for row in sample_rows:
            nl += f"Row: {', '.join(row)}. "
        
        if self.footnotes:
            nl += f"Notes: {'; '.join(self.footnotes)}. "
        
        return nl
    
    def generate_qa_pairs(self) -> List[str]:
        """Generate question-answer pairs for better retrieval"""
        qa_pairs = []
        
        # Create questions from headers and rows
        for i, header in enumerate(self.headers):
            sample_values = [row[i] for row in self.rows[:3] if len(row) > i]
            if sample_values:
                qa_pairs.append(f"Q: What is the {header}? A: Examples include {', '.join(sample_values)}")
        
        return qa_pairs

@dataclass
class Formula:
    formula_id: str
    formula_text: str
    page: int
    section: str
    variables: Dict[str, str]
    conditions: List[str]
    example_calculation: Optional[str] = None
    related_tables: List[str] = field(default_factory=list)
    
    def to_natural_language(self) -> str:
        nl = f"Formula {self.formula_id} on page {self.page}: {self.formula_text}. "
        nl += f"Variables: {', '.join([f'{k}={v}' for k, v in self.variables.items()])}. "
        
        if self.conditions:
            nl += f"Conditions: {'; '.join(self.conditions)}. "
        
        if self.example_calculation:
            nl += f"Example: {self.example_calculation}. "
        
        return nl

@dataclass
class Section:
    section_id: str
    title: str
    page_start: int
    page_end: int
    content: str
    parent_section: Optional[str] = None
    subsections: List[str] = field(default_factory=list)
    
    def to_natural_language(self) -> str:
        return f"Section {self.section_id}: {self.title} (pages {self.page_start}-{self.page_end})"

@dataclass
class Paragraph:
    paragraph_id: str
    text: str
    page: int
    section_id: str
    section_title: str
    chunk_index: int
    
    def to_natural_language(self) -> str:
        return f"[Page {self.page}] {self.text}"

@dataclass
class IngestionProgress:
    total_pages: int = 0
    processed_pages: int = 0
    current_page: int = 0
    pages_processed: List[int] = field(default_factory=list)
    current_operation: str = ""
    errors: List[str] = field(default_factory=list)
    tables_found: int = 0
    formulas_found: int = 0
    paragraphs_extracted: int = 0
    paragraphs_chunked: int = 0
