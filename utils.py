import hashlib
import json
from typing import List, Dict, Any
import re

def generate_id(content: str, prefix: str = "") -> str:
    """Generate deterministic ID from content"""
    hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_val}" if prefix else hash_val

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def extract_table_number(text: str) -> str:
    """Extract table number from text"""
    match = re.search(r'table\s+(\d+[-\.]\d+)', text.lower())
    return match.group(1) if match else None

def extract_formula_variables(formula_text: str) -> Dict[str, str]:
    """Extract variables from formula text"""
    variables = {}
    pattern = r'([A-Z][a-z]*)\s*=\s*([\d\.]+)'
    matches = re.finditer(pattern, formula_text)
    
    for match in matches:
        variables[match.group(1)] = match.group(2)
    
    return variables

def validate_citation(chunk_id: str, cosmos_container) -> bool:
    """Validate that a chunk exists in Cosmos DB"""
    try:
        cosmos_container.read_item(item=chunk_id, partition_key="content")
        return True
    except:
        return False

def format_page_range(pages: List[int]) -> str:
    """Format page numbers for display"""
    if not pages:
        return "N/A"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"

class ProgressTracker:
    """Track progress through ingestion pipeline"""
    def __init__(self):
        self.current_step = ""
        self.step_progress = 0
        self.total_steps = 0
        self.errors = []
        
    def update(self, step: str, progress: int, total: int = 100):
        self.current_step = step
        self.step_progress = progress
        self.total_steps = total
    
    def add_error(self, error: str):
        self.errors.append(error)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "progress": self.step_progress / max(self.total_steps, 1),
            "errors": self.errors[-5:]  # Last 5 errors
        }
