import asyncio
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from llama_parse import LlamaParse
import tempfile
import os
import pymongo
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import AzureOpenAI
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import re

from config import Config
from models import Table, Formula, Section, Paragraph, IngestionProgress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def string_to_qdrant_id(string_id: str) -> str:
    """Convert string ID to valid Qdrant point ID (UUID format)"""
    # Use uuid5 with a fixed namespace for deterministic conversion
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, string_id))

class AMAGuidesIngestionPipeline:
    def __init__(self):
        """Initialize all Azure and Qdrant clients"""
        self.config = Config()
        self.progress = IngestionProgress()
        
        # Document parser selection
        self.use_llama_parse = self.config.USE_LLAMA_PARSE and bool(self.config.LLAMA_CLOUD_API_KEY)
        self.llama_parser = None

        if self.use_llama_parse:
            self.llama_parser = LlamaParse(
                result_type="markdown",
                num_workers=2,
                verbose=True
            )
            logger.info("✅ Using LlamaParse for document parsing")
        else:
            logger.info("🔁 Using Azure Document Intelligence for document parsing")

        # Initialize Azure clients
        self.doc_intelligence = DocumentIntelligenceClient(
            endpoint=self.config.DOC_INTELLIGENCE_ENDPOINT,
            credential=AzureKeyCredential(self.config.DOC_INTELLIGENCE_KEY)
        )
        
        self.openai_client = AzureOpenAI(
            azure_endpoint=self.config.OPENAI_ENDPOINT,
            api_key=self.config.OPENAI_KEY,
            api_version="2023-12-01-preview"
        )
        
        # Cosmos DB is optional — if connection fails, pipeline continues with Qdrant only
        self.cosmos_container = None
        try:
            self.cosmos_client = pymongo.MongoClient(
                self.config.COSMOS_CONNECTION_STRING,
                serverSelectionTimeoutMS=5000  # Fail fast instead of waiting 30s
            )
            # Ping to verify connection is actually reachable
            self.cosmos_client.admin.command('ping')
            self._init_cosmos()
            logger.info("✅ Cosmos DB connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Cosmos DB unavailable, skipping (Qdrant will still work): {e}")
            self.cosmos_client = None
        
        self.qdrant_client = QdrantClient(
            url=self.config.QDRANT_URL,
            api_key=self.config.QDRANT_API_KEY,
            timeout=60.0,  # Increased timeout for large batches
            grpc_options={
                "grpc.max_receive_message_length": 50 * 1024 * 1024,  # 50MB
                "grpc.max_send_message_length": 50 * 1024 * 1024,  # 50MB
            }
        )
        self._init_qdrant_collections()
        
    def _report_progress(self, callback=None):
        if callback:
            callback(self.progress)

    def _init_cosmos(self):
        """Initialize Cosmos DB containers"""
        database = self.cosmos_client[self.config.COSMOS_DATABASE]
        self.cosmos_container = database[self.config.COSMOS_CONTAINER]
    
    def _init_qdrant_collections(self):
        """Initialize Qdrant collections if they don't exist"""
        for collection_name in self.config.QDRANT_COLLECTIONS.values():
            try:
                self.qdrant_client.get_collection(collection_name)
            except:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _upsert_with_retry(self, collection_name: str, points: List[PointStruct]):
        """Upsert with retry logic and timeout handling"""
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
        except Exception as e:
            logger.error(f"❌ Upsert failed for {collection_name}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def parse_pdf(self, pdf_bytes: bytes, progress_callback=None) -> Dict[str, Any]:
        """Parse PDF using LlamaParse or Azure Document Intelligence."""
        if self.use_llama_parse:
            self.progress.current_operation = "Parsing PDF with LlamaParse..."
            self._report_progress(progress_callback)
            return await self._parse_with_llama(pdf_bytes, progress_callback)

        self.progress.current_operation = "Parsing PDF with Azure Document Intelligence..."
        self._report_progress(progress_callback)

        try:
            # Analyze document
            logger.info(f"📥 Analyzing PDF ({len(pdf_bytes)} bytes) with Azure Document Intelligence...")
            poller = self.doc_intelligence.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=pdf_bytes)
            )
            result = poller.result()
            
            if not result:
                logger.error("❌ Azure Document Intelligence returned no result")
                raise ValueError("Document analysis failed - no result returned")
            
            logger.info(f"✅ Document analysis complete. Result object: {type(result)}")
            logger.info(f"   - Pages: {len(result.pages) if result.pages else 0}")
            logger.info(f"   - Paragraphs: {len(result.paragraphs) if result.paragraphs else 0}")
            logger.info(f"   - Tables: {len(result.tables) if result.tables else 0}")
            
            self.progress.total_pages = len(result.pages) if result.pages else 0
            self.progress.pages_processed = []
            self.progress.processed_pages = 0
            self.progress.current_page = 0

            # Track page extraction during parse
            for page_index, _ in enumerate(result.pages or [], start=1):
                self.progress.current_page = page_index
                if page_index not in self.progress.pages_processed:
                    self.progress.pages_processed.append(page_index)
                self.progress.processed_pages = page_index
                self.progress.current_operation = f"Parsing page {page_index}/{self.progress.total_pages} with Azure"
                self._report_progress(progress_callback)

            # Extract structured content
            structured = {
                "tables": [],
                "paragraphs": [],
                "formulas": [],
                "sections": [],
                "pages": result.pages if result.pages else []
            }
            
            # Extract tables
            if result.tables:
                for table in result.tables:
                    table_obj = self._extract_table(table, result.pages)
                    structured["tables"].append(table_obj)
                    self.progress.tables_found += 1
                logger.info(f"✅ Extracted {len(result.tables)} tables")
            else:
                logger.warning("⚠️ No tables found in document")
            
            # Extract paragraphs
            para_count = 0
            if result.paragraphs:
                for para in result.paragraphs:
                    paragraph_obj = self._extract_paragraph(para, result)
                    if paragraph_obj:
                        structured["paragraphs"].append(paragraph_obj)
                        para_count += 1
                self.progress.paragraphs_extracted = para_count
                logger.info(f"✅ Extracted {para_count} paragraphs from {len(result.paragraphs)} raw paragraphs")
            else:
                logger.warning("⚠️ No paragraphs found in document")
            
            # Extract formulas (custom regex for AMA Guides)
            structured["formulas"] = self._extract_formulas(result)
            logger.info(f"✅ Extracted {len(structured['formulas'])} formulas")
            
            # Extract sections from headings
            structured["sections"] = self._extract_sections(result)
            logger.info(f"✅ Extracted {len(structured['sections'])} sections")
            
            self.progress.processed_pages = len(result.pages) if result.pages else 0
            self.progress.pages_processed = list(range(1, self.progress.processed_pages + 1))
            logger.info(f"📄 Parsed {self.progress.processed_pages} pages successfully")
            return structured
            
        except Exception as e:
            self.progress.errors.append(f"PDF parsing failed: {str(e)}")
            raise
    
    def _extract_table(self, table, pages) -> Table:
        """Extract table with multi-page handling"""
        # Find which page this table is on
        cell = table.cells[0] if table.cells else None
        page_num = cell.bounding_regions[0].page_number if cell and cell.bounding_regions else 1
        
        headers = []
        rows = []
        
        # Extract headers (first row typically)
        for cell in table.cells:
            if cell.row_index == 0:
                headers.append(cell.content or "")
            else:
                # Ensure row exists
                while len(rows) <= cell.row_index - 1:
                    rows.append([])
                rows[cell.row_index - 1].append(cell.content or "")
        
        # Clean up rows to have consistent columns
        max_cols = len(headers)
        for row in rows:
            while len(row) < max_cols:
                row.append("")
        
        # Generate table ID from content hash
        table_hash = hashlib.md5(str(headers).encode()).hexdigest()[:8]
        table_id = f"tbl_{page_num}_{table_hash}"
        
        logger.debug(f"  📊 Extracted table: {table_id} ({len(headers)} headers, {len(rows)} rows)")
        
        return Table(
            table_id=table_id,
            pages=[page_num],
            headers=headers,
            rows=rows,
            footnotes=[]  # Would need additional extraction
        )
    
    def _extract_paragraph(self, para, result) -> Optional[Paragraph]:
        """Extract paragraph with metadata"""
        # Skip empty or very short content
        if not para.content or len(para.content.strip()) < 3:
            return None
        
        content = para.content.strip()
        
        # Find page number
        page_num = 1
        if para.bounding_regions:
            page_num = para.bounding_regions[0].page_number
        
        # Find section (simplified - would need heading detection)
        section_id = "unknown"
        section_title = "General"
        
        # Generate paragraph ID
        para_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        paragraph_id = f"para_{page_num}_{para_hash}"
        
        logger.debug(f"  📝 Extracted paragraph: {paragraph_id} ({len(content)} chars)")
        
        return Paragraph(
            paragraph_id=paragraph_id,
            text=content,
            page=page_num,
            section_id=section_id,
            section_title=section_title,
            chunk_index=0
        )
    
    def _extract_formulas(self, result) -> List[Formula]:
        """Extract formulas using regex patterns"""
        formulas = []
        
        # Patterns for AMA Guides formulas
        formula_patterns = [
            r'(\w+\s*=\s*[\d\w\s\+\-\*\/\(\)\.]+)',  # Basic formula with decimals
            r'(?:Whole Person Impairment|Impairment)\s*(?:is\s+)?(?:calculated\s+as\s+)?(?:follows)?:?\s*([^\.]+\.)',  # Common AMA pattern
            r'(\w+\s*coefficient|factor|rating|percentage)',  # Look for key terms
        ]
        
        all_text = ""
        if result.paragraphs:
            all_text = " ".join([p.content for p in result.paragraphs if p.content])
        
        if not all_text.strip():
            logger.debug("  📐 No text found for formula extraction")
            return formulas
        
        # Find formulas
        formula_count = 0
        for pattern in formula_patterns:
            try:
                matches = re.finditer(pattern, all_text, re.IGNORECASE | re.MULTILINE)
                for i, match in enumerate(matches):
                    formula_text = match.group(1) if match.lastindex else match.group(0)
                    formula_text = formula_text.strip()
                    
                    # Skip very short matches
                    if len(formula_text) < 5:
                        continue
                    
                    # Extract variables (simplified)
                    variables = {}
                    var_pattern = r'([A-Z][a-z]*)\s*=\s*([\d\.]+)'
                    var_matches = re.finditer(var_pattern, formula_text)
                    for var_match in var_matches:
                        variables[var_match.group(1)] = var_match.group(2)
                    
                    formula = Formula(
                        formula_id=f"formula_{formula_count+1}",
                        formula_text=formula_text[:500],  # Limit length
                        page=1,  # Would need actual page
                        section="Unknown",
                        variables=variables,
                        conditions=[],
                        example_calculation=None
                    )
                    formulas.append(formula)
                    formula_count += 1
                    self.progress.formulas_found += 1
            except Exception as e:
                logger.warning(f"⚠️ Formula pattern matching error: {e}")
                continue
        
        logger.debug(f"  📐 Extracted {formula_count} formulas from patterns")
        return formulas
    
    def _extract_sections(self, result) -> List[Section]:
        """Extract sections from headings"""
        sections = []
        current_section = None
        section_count = 0
        
        if result.paragraphs:
            for para in result.paragraphs:
                # Check if this paragraph is a heading (simplified detection)
                is_heading = (
                    para.role == "heading" or
                    (para.content and len(para.content) < 100 and len(para.content) > 3 and para.content.isupper()) or
                    (para.content and len(para.content) < 80 and para.content.startswith(("Chapter", "Section", "Part", "**")))
                )
                
                if is_heading and para.content:
                    section_id = hashlib.md5(para.content.encode()).hexdigest()[:8]
                    page_num = para.bounding_regions[0].page_number if para.bounding_regions else 1
                    
                    section = Section(
                        section_id=f"sec_{section_id}",
                        title=para.content.strip(),
                        page_start=page_num,
                        page_end=page_num,
                        content=""
                    )
                    sections.append(section)
                    current_section = section
                    section_count += 1
                    logger.debug(f"  📂 Found section: {para.content[:50]}")
                elif current_section and para.content:
                    current_section.content += " " + para.content.strip()
                    if para.bounding_regions:
                        current_section.page_end = para.bounding_regions[0].page_number
        
        logger.debug(f"  📂 Extracted {section_count} sections")
        return sections

    async def _parse_with_llama(self, pdf_bytes: bytes, progress_callback=None) -> Dict[str, Any]:
        """Parse PDF bytes via LlamaParse and return structured content."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            file_path = tmp.name

        try:
            docs = self.llama_parser.load_data(file_path)
        finally:
            try:
                os.remove(file_path)
            except Exception:
                pass

        text = []
        tables = []
        paragraphs = []
        formulas = []
        sections = []
        pages = len(docs)

        self.progress.total_pages = pages
        self.progress.pages_processed = []
        self.progress.processed_pages = 0
        self.progress.current_page = 0

        for i, doc in enumerate(docs, start=1):
            self.progress.current_page = i
            if i not in self.progress.pages_processed:
                self.progress.pages_processed.append(i)
            self.progress.processed_pages = i
            self.progress.current_operation = f"Parsing page {i}/{pages} with LlamaParse"
            self._report_progress(progress_callback)

            text.append(f"\n--- Page {i} ---\n")
            doc_text = getattr(doc, 'text', str(doc))
            text.append(doc_text)

            # Parse markdown tables if available
            tables.extend(self._extract_tables_from_markdown(doc_text))
            
            # Extract paragraphs from this page
            page_paragraphs = self._extract_paragraphs_from_text(doc_text, page_num=i)
            paragraphs.extend(page_paragraphs)
            self.progress.paragraphs_extracted += len(page_paragraphs)

        # Extract formulas and sections from all text
        raw_text = "".join(text)
        formulas = self._extract_formulas_from_text(raw_text)
        sections = self._extract_sections_from_text(raw_text)
        
        self.progress.current_operation = "LlamaParse parsing complete"
        self._report_progress(progress_callback)

        return {
            "tables": tables,
            "paragraphs": paragraphs,
            "formulas": formulas,
            "sections": sections,
            "pages": pages,
            "raw_text": raw_text
        }

    def _extract_tables_from_markdown(self, text: str) -> List[Table]:
        """Extract tables from multiple formats: markdown, grid, CSV-like structures"""
        result = []
        table_count = 0
        
        # Try markdown table format (| header | header |)
        result.extend(self._extract_markdown_tables(text))
        markdown_count = len(result)
        
        # Try grid table format (using +----+ borders)
        result.extend(self._extract_grid_tables(text))
        grid_count = len(result) - markdown_count
        
        # Try CSV-like or space-separated table format
        result.extend(self._extract_csv_style_tables(text))
        csv_count = len(result) - markdown_count - grid_count
        
        # Try HTML table format
        result.extend(self._extract_html_tables(text))
        html_count = len(result) - markdown_count - grid_count - csv_count
        
        total_new = markdown_count + grid_count + csv_count + html_count
        if total_new > 0:
            logger.info(f"📊 Extracted {markdown_count} markdown + {grid_count} grid + {csv_count} CSV + {html_count} HTML tables = {total_new} total")
        
        return result

    def _extract_markdown_tables(self, text: str) -> List[Table]:
        """Extract pipe-delimited markdown tables"""
        result = []
        lines = text.splitlines()
        idx = 0
        table_count = 0

        while idx < len(lines):
            line = lines[idx].strip()
            # Check for pipe-delimited table (| header |)
            if line.startswith('|') and '|' in line and idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                # Check if next line is separator (| --- | --- |)
                if next_line.startswith('|') and '-' in next_line:
                    header = [h.strip() for h in line.split('|')[1:-1]]
                    if not header or all(not h for h in header):
                        idx += 1
                        continue
                    
                    idx += 2  # Skip header and separator
                    rows = []

                    while idx < len(lines) and lines[idx].strip().startswith('|'):
                        row_line = lines[idx].strip()
                        cells = [c.strip() for c in row_line.split('|')[1:-1]]
                        if cells and len(cells) == len(header):
                            rows.append(cells)
                        idx += 1

                    if rows:  # Only create table if we have rows
                        table_count += 1
                        table_hash = hashlib.md5(str(header).encode()).hexdigest()[:8]
                        table_id = f"tbl_md_{table_count}_{table_hash}"
                        
                        table = Table(
                            table_id=table_id,
                            pages=[1],
                            headers=header,
                            rows=rows,
                            footnotes=[]
                        )
                        result.append(table)
                        logger.debug(f"  📊 Markdown table: {table_id} ({len(header)} cols, {len(rows)} rows)")
                    continue
            idx += 1

        return result

    def _extract_grid_tables(self, text: str) -> List[Table]:
        """Extract grid/box-drawn tables (+----|----+)"""
        result = []
        lines = text.splitlines()
        idx = 0
        table_count = 0

        while idx < len(lines):
            line = lines[idx].strip()
            # Look for grid border: lines with + and -
            if line.startswith('+') and '-' in line and line.endswith('+'):
                table_lines = [line]
                idx += 1
                
                # Collect table content until closing border
                while idx < len(lines):
                    current = lines[idx]
                    table_lines.append(current)
                    if current.strip().startswith('+') and '-' in current and current.strip().endswith('+'):
                        break
                    idx += 1
                
                # Parse the grid table
                table = self._parse_grid_table(table_lines, table_count + 1)
                if table and table.rows:
                    result.append(table)
                    table_count += 1
                    logger.debug(f"  📊 Grid table: {table.table_id} ({len(table.headers)} cols, {len(table.rows)} rows)")
            idx += 1

        return result

    def _parse_grid_table(self, lines: List[str], table_num: int) -> Optional[Table]:
        """Parse grid-format table"""
        if len(lines) < 3:
            return None
        
        try:
            # Extract column positions from first border line
            border = lines[0]
            col_positions = [i for i, c in enumerate(border) if c == '+']
            
            if len(col_positions) < 3:  # Need at least 2 columns (3 positions)
                return None
            
            # Extract header row
            if len(lines) > 1:
                header_line = lines[1]
                headers = []
                for i in range(len(col_positions) - 1):
                    start = col_positions[i] + 1
                    end = col_positions[i + 1]
                    cell_text = header_line[start:end].strip()
                    headers.append(cell_text)
                
                # Extract data rows
                rows = []
                for line_idx in range(3, len(lines) - 1, 2):  # Skip borders
                    if line_idx < len(lines):
                        row_line = lines[line_idx]
                        row = []
                        for i in range(len(col_positions) - 1):
                            start = col_positions[i] + 1
                            end = col_positions[i + 1]
                            cell_text = row_line[start:end].strip()
                            row.append(cell_text)
                        if row and any(row):  # Only add non-empty rows
                            rows.append(row)
                
                if headers and rows:
                    table_hash = hashlib.md5(str(headers).encode()).hexdigest()[:8]
                    return Table(
                        table_id=f"tbl_grid_{table_num}_{table_hash}",
                        pages=[1],
                        headers=headers,
                        rows=rows,
                        footnotes=[]
                    )
        except Exception as e:
            logger.debug(f"  ⚠️ Grid table parsing error: {e}")
        
        return None

    def _extract_csv_style_tables(self, text: str) -> List[Table]:
        """Extract space or tab-separated table-like structures"""
        result = []
        lines = text.splitlines()
        idx = 0
        table_count = 0

        while idx < len(lines):
            line = lines[idx].strip()
            
            # Skip empty lines and non-table lines
            if not line or line.startswith('#'):
                idx += 1
                continue
            
            # Look for lines with consistent column structure
            # Must have multiple whitespace-separated values
            tokens = line.split()
            
            if len(tokens) >= 3:  # At least 3 columns
                # Check if next few lines have similar structure
                potential_rows = [tokens]
                check_idx = idx + 1
                consistent_cols = len(tokens)
                
                while check_idx < min(idx + 10, len(lines)):
                    next_line = lines[check_idx].strip()
                    if not next_line:
                        check_idx += 1
                        continue
                    
                    next_tokens = next_line.split()
                    # Allow some flexibility in column count (±1)
                    if len(next_tokens) >= max(2, consistent_cols - 2):
                        potential_rows.append(next_tokens)
                        check_idx += 1
                    else:
                        break
                
                # If we found at least 3-4 consistent rows, treat as table
                if len(potential_rows) >= 2:
                    # Use first row as header
                    headers = potential_rows[0][:consistent_cols]
                    rows = []
                    
                    for row_tokens in potential_rows[1:]:
                        # Pad or trim to match header count
                        row = (row_tokens[:consistent_cols] + [''] * consistent_cols)[:consistent_cols]
                        rows.append(row)
                    
                    if rows:
                        table_count += 1
                        table_hash = hashlib.md5(str(headers).encode()).hexdigest()[:8]
                        table = Table(
                            table_id=f"tbl_csv_{table_count}_{table_hash}",
                            pages=[1],
                            headers=headers,
                            rows=rows,
                            footnotes=[]
                        )
                        result.append(table)
                        logger.debug(f"  📊 CSV-style table: {table.table_id} ({len(headers)} cols, {len(rows)} rows)")
                        idx = check_idx
                        continue
            
            idx += 1

        return result

    def _extract_html_tables(self, text: str) -> List[Table]:
        """Extract HTML table format (if LlamaParse outputs HTML)"""
        result = []
        table_count = 0
        
        # Simple HTML table pattern matching
        import re
        table_pattern = r'<table[^>]*>(.*?)</table>'
        
        for table_match in re.finditer(table_pattern, text, re.IGNORECASE | re.DOTALL):
            table_html = table_match.group(1)
            
            # Extract headers
            header_pattern = r'<th[^>]*>([^<]*)</th>'
            headers = [h.strip() for h in re.findall(header_pattern, table_html, re.IGNORECASE)]
            
            # Extract rows
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            cell_pattern = r'<td[^>]*>([^<]*)</td>'
            
            rows = []
            for row_match in re.finditer(row_pattern, table_html, re.IGNORECASE | re.DOTALL):
                row_html = row_match.group(1)
                cells = [c.strip() for c in re.findall(cell_pattern, row_html, re.IGNORECASE)]
                if cells:
                    rows.append(cells)
            
            if headers and rows:
                table_count += 1
                table_hash = hashlib.md5(str(headers).encode()).hexdigest()[:8]
                table = Table(
                    table_id=f"tbl_html_{table_count}_{table_hash}",
                    pages=[1],
                    headers=headers,
                    rows=rows,
                    footnotes=[]
                )
                result.append(table)
                logger.debug(f"  📊 HTML table: {table.table_id} ({len(headers)} cols, {len(rows)} rows)")
        
        return result

    def _extract_paragraphs_from_text(self, text: str, page_num: int = 1) -> List[Paragraph]:
        """Extract paragraphs from plain text by splitting on blank lines."""
        paragraphs = []
        
        # Split by multiple newlines to create paragraph boundaries
        para_texts = re.split(r'\n\n+', text.strip())
        
        para_index = 0
        for para_text in para_texts:
            para_text = para_text.strip()
            
            # Skip very short content
            if not para_text or len(para_text) < 10:
                continue
            
            # Skip markdown code blocks
            if para_text.startswith('```') or para_text.startswith('---'):
                continue
            
            # Generate paragraph ID
            para_hash = hashlib.md5(para_text.encode()).hexdigest()[:12]
            paragraph_id = f"para_{page_num}_{para_hash}"
            
            paragraph = Paragraph(
                paragraph_id=paragraph_id,
                text=para_text,
                page=page_num,
                section_id="general",
                section_title="Document Content",
                chunk_index=para_index
            )
            paragraphs.append(paragraph)
            para_index += 1
            logger.debug(f"  📝 Extracted paragraph: {paragraph_id} ({len(para_text)} chars)")
        
        return paragraphs

    def _extract_formulas_from_text(self, text: str) -> List[Formula]:
        """Extract formulas with improved pattern matching and context"""
        formulas = []
        formula_count = 0
        processed = set()
        
        # Comprehensive formula patterns
        patterns = [
            # Pattern 1: "Formula/Equation: <formula text>"
            (r'(?:Formula|Equation|Calculation|Method)[\s:]+([^\n\.]+(?:\n(?:\s{2,}|\t)[^\n]+)*)', "labeled_formula"),
            
            # Pattern 2: "X = expression" or "Variable = calculation"
            (r'([A-Z][a-zA-Z\s]*?)\s*=\s*([^=\n\.]+(?:\n\s+[^=\n]+)*)', "assignment_formula"),
            
            # Pattern 3: Text mentioning calculations
            (r'(?:calculated?\s+(?:as|by|using|is)[\s:]*|To\s+calculate)([^\n\.]+(?:\n[^\n]+){0,2})', "calculation_text"),
            
            # Pattern 4: Whole Person Impairment specific
            (r'(?:Whole\s+Person\s+Impairment|WPI|Combined\s+Impairment)[\s:]*([^\n\"]+(?:\n[^\n\"]+)*)', "impairment_formula"),
            
            # Pattern 5: Percentage/rating formulas  
            (r'(?:percentage|rating|score)[\s:]*([^\n]+)', "rating_formula"),
        ]
        
        for pattern, pattern_type in patterns:
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    formula_text = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                    formula_text = formula_text.strip()[:400]  # Limit length
                    
                    # Skip empty and very short formulas
                    if len(formula_text) < 5 or formula_text in processed:
                        continue
                    
                    # Skip non-formula text
                    if formula_text.lower().startswith(('the', 'a ', 'an ')):
                        continue
                    
                    processed.add(formula_text)
                    
                    # Extract variables from formula
                    variables = self._extract_variables_from_formula(formula_text)
                    
                    formula = Formula(
                        formula_id=f"formula_{formula_count+1}",
                        formula_text=formula_text,
                        page=1,
                        section=pattern_type,
                        variables=variables,
                        conditions=[],
                        example_calculation=None
                    )
                    formulas.append(formula)
                    formula_count += 1
                    self.progress.formulas_found += 1
                    logger.debug(f"  📐 Found {pattern_type}: {formula_text[:50]}...")
            
            except Exception as e:
                logger.debug(f"⚠️ Formula pattern '{pattern_type}' error: {e}")
        
        logger.info(f"✅ Extracted {formula_count} formulas from text")
        return formulas

    def _extract_variables_from_formula(self, formula_text: str) -> Dict[str, str]:
        """Extract variable definitions from formula text"""
        variables = {}
        
        # Pattern: "Variable (X) = value" or "X = something"
        patterns = [
            r'([A-Z]\w*)\s*\((.+?)\)\s*=',  # X (name) = 
            r'\b([A-Z]\w*)\s*=\s*([\d\.]+)',  # X = number
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, formula_text):
                var_name = match.group(1)
                var_desc = match.group(2) if match.lastindex >= 2 else "unknown"
                variables[var_name] = var_desc
        
        return variables

    def _extract_sections_from_text(self, text: str) -> List[Section]:
        """Extract sections with improved heading detection"""
        sections = []
        section_count = 0
        
        # Markdown-style headers (# ## ### etc)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        # Alternative heading patterns
        alt_patterns = [
            r'^([A-Z\s]+)$',  # ALL CAPS lines (potential headings)
            r'^(Chapter|Section|Part|Appendix)\s+([A-Za-z0-9]+)[\s:]*(.*)$',  # Labeled sections
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            is_heading = False
            section_title = None
            
            # Check markdown headers
            header_match = re.match(header_pattern, line)
            if header_match:
                is_heading = True
                section_title = header_match.group(2).strip()
            else:
                # Check alternative patterns
                if len(line.strip()) > 3 and len(line.strip()) < 100:
                    all_caps_test = line.strip()
                    # Check if it's a reasonable heading (not ALL TEXT)
                    if all_caps_test.isupper() and ' ' in all_caps_test:
                        is_heading = True
                        section_title = all_caps_test
                    
                    # Check labeled section pattern
                    for alt_pattern in alt_patterns[1:]:
                        alt_match = re.match(alt_pattern, line)
                        if alt_match:
                            is_heading = True
                            section_title = line.strip()
                            break
            
            if is_heading and section_title:
                # Save previous section
                if current_section and current_content:
                    current_section.content = '\n'.join(current_content).strip()
                    if current_section.content:  # Only save if has content
                        sections.append(current_section)
                        section_count += 1
                
                # Create new section
                section_id = hashlib.md5(section_title.encode()).hexdigest()[:8]
                current_section = Section(
                    section_id=f"sec_{section_id}",
                    title=section_title,
                    page_start=1,
                    page_end=1,
                    content=""
                )
                current_content = []
                logger.debug(f"  📂 Found section: {section_title[:60]}")
            
            elif current_section and line.strip():
                # Add content to current section
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            current_section.content = '\n'.join(current_content).strip()
            if current_section.content:
                sections.append(current_section)
                section_count += 1
        
        logger.info(f"✅ Extracted {section_count} sections from text")
        return sections

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_DEPLOYMENT,
                input=text[:8000]  # Truncate to avoid token limits
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * self.config.EMBEDDING_DIMENSIONS
    
    async def store_in_cosmos(self, structured_data: Dict[str, Any], progress_callback=None):
        """Store all content in Cosmos DB (skipped if unavailable)"""
        if self.cosmos_container is None:
            logger.warning("Cosmos DB not available, skipping storage step")
            self.progress.current_operation = "Skipping Cosmos DB (unavailable)..."
            self._report_progress(progress_callback)
            return
        
        self.progress.current_operation = "Storing in Cosmos DB..."
        self._report_progress(progress_callback)
        
        try:
            # Helper for mongo upsert
            def upsert_mongo(doc):
                doc["_id"] = doc["id"]
                self.cosmos_container.update_one({"_id": doc["id"]}, {"$set": doc}, upsert=True)

            # Store tables
            for table in structured_data["tables"]:
                document = {
                    "id": table.table_id,
                    "content_type": "table",
                    "data": table.__dict__,
                    "natural_language": table.to_natural_language(),
                    "pages": table.pages,
                    "timestamp": datetime.utcnow().isoformat()
                }
                upsert_mongo(document)
            
            # Store paragraphs
            for para in structured_data["paragraphs"]:
                document = {
                    "id": para.paragraph_id,
                    "content_type": "paragraph",
                    "data": para.__dict__,
                    "text": para.text,
                    "page": para.page,
                    "section_id": para.section_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                upsert_mongo(document)
            
            # Store formulas
            for formula in structured_data["formulas"]:
                document = {
                    "id": formula.formula_id,
                    "content_type": "formula",
                    "data": formula.__dict__,
                    "formula_text": formula.formula_text,
                    "page": formula.page,
                    "timestamp": datetime.utcnow().isoformat()
                }
                upsert_mongo(document)
            
            # Store sections
            for section in structured_data["sections"]:
                document = {
                    "id": section.section_id,
                    "content_type": "section",
                    "data": section.__dict__,
                    "title": section.title,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "timestamp": datetime.utcnow().isoformat()
                }
                upsert_mongo(document)
                
        except Exception as e:
            self.progress.errors.append(f"Cosmos DB storage failed: {str(e)}")
            raise
    
    async def store_in_qdrant(self, structured_data: Dict[str, Any], progress_callback=None):
        """Store embeddings in Qdrant"""
        self.progress.current_operation = "Generating embeddings and storing in Qdrant..."
        self._report_progress(progress_callback)
        
        points = []
        
        # Process tables
        for table in structured_data["tables"]:
            # Multiple representations for better retrieval
            representations = [
                table.to_natural_language(),
                *table.generate_qa_pairs(),
                table.to_json()
            ]
            
            for i, rep in enumerate(representations[:3]):  # Limit to 3 representations
                embedding = await self.generate_embedding(rep)
                
                point = PointStruct(
                    id=string_to_qdrant_id(f"{table.table_id}_rep_{i}"),
                    vector=embedding,
                    payload={
                        "type": "table",
                        "table_id": table.table_id,
                        "text": rep,
                        "pages": table.pages
                    }
                )
                points.append(point)

                # Batch upsert to avoid memory issues (reduced batch size for reliability)
                if len(points) >= 50:
                    try:
                        self._upsert_with_retry(
                            collection_name=self.config.QDRANT_COLLECTIONS["tables"],
                            points=points
                        )
                        self.progress.paragraphs_chunked += len(points)
                        self.progress.current_operation = f"Upserted {self.progress.paragraphs_chunked} points in tables"
                        self._report_progress(progress_callback)
                    except Exception as e:
                        self.progress.errors.append(f"Failed to upsert table points: {str(e)}")
                        logger.error(f"❌ Table upsert failed: {e}")
                    points = []
        
        # Process paragraphs
        for para in structured_data["paragraphs"]:
            embedding = await self.generate_embedding(para.text)
            
            point = PointStruct(
                id=string_to_qdrant_id(para.paragraph_id),
                vector=embedding,
                payload={
                    "type": "paragraph",
                    "paragraph_id": para.paragraph_id,
                    "text": para.text,
                    "page": para.page,
                    "section_id": para.section_id,
                    "section_title": para.section_title
                }
            )
            points.append(point)

            if len(points) >= 50:
                try:
                    self._upsert_with_retry(
                        collection_name=self.config.QDRANT_COLLECTIONS["paragraphs"],
                        points=points
                    )
                    self.progress.paragraphs_chunked += len(points)
                    self.progress.current_operation = f"Upserted {self.progress.paragraphs_chunked} paragraph points"
                    self._report_progress(progress_callback)
                except Exception as e:
                    self.progress.errors.append(f"Failed to upsert paragraph points: {str(e)}")
                    logger.error(f"❌ Paragraph upsert failed: {e}")
                points = []
        
        # Process formulas
        for formula in structured_data["formulas"]:
            embedding = await self.generate_embedding(formula.to_natural_language())
            
            point = PointStruct(
                id=string_to_qdrant_id(formula.formula_id),
                vector=embedding,
                payload={
                    "type": "formula",
                    "formula_id": formula.formula_id,
                    "formula_text": formula.formula_text,
                    "variables": formula.variables,
                    "conditions": formula.conditions
                }
            )
            points.append(point)
        
        # Final upsert
        if points:
            try:
                collection = self.config.QDRANT_COLLECTIONS["paragraphs"]
                if points[0].payload["type"] == "table":
                    collection = self.config.QDRANT_COLLECTIONS["tables"]
                elif points[0].payload["type"] == "formula":
                    collection = self.config.QDRANT_COLLECTIONS["formulas"]
                
                self._upsert_with_retry(
                    collection_name=collection,
                    points=points
                )
                self.progress.paragraphs_chunked += len(points)
                self.progress.current_operation = f"Final Qdrant upsert done ({len(points)} points, total {self.progress.paragraphs_chunked})"
                self._report_progress(progress_callback)
            except Exception as e:
                self.progress.errors.append(f"Final upsert failed: {str(e)}")
                logger.error(f"❌ Final upsert failed: {e}")
    
    async def run_pipeline(self, pdf_bytes: bytes, progress_callback=None) -> IngestionProgress:
        """Run complete ingestion pipeline"""
        try:
            # Step 1: Parse PDF
            structured_data = await self.parse_pdf(pdf_bytes, progress_callback=progress_callback)
            
            # Step 2: Store in Cosmos DB
            await self.store_in_cosmos(structured_data, progress_callback=progress_callback)
            
            # Step 3: Generate embeddings and store in Qdrant
            await self.store_in_qdrant(structured_data, progress_callback=progress_callback)
            
            self.progress.current_operation = "Ingestion complete!"
            self.progress.processed_pages = self.progress.total_pages
            self._report_progress(progress_callback)
            
        except Exception as e:
            self.progress.errors.append(f"Pipeline failed: {str(e)}")
            raise
        
        return self.progress
