from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from openai import AzureOpenAI
from qdrant_client import QdrantClient
import pymongo
import re

from config import Config
from models import Table, Formula

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMARetrievalEngine:
    def __init__(self):
        self.config = Config()
        
        # Initialize clients
        self.openai_client = AzureOpenAI(
            azure_endpoint=self.config.OPENAI_ENDPOINT,
            api_key=self.config.OPENAI_KEY,
            api_version="2023-12-01-preview"
        )
        
        self.qdrant_client = QdrantClient(
            url=self.config.QDRANT_URL,
            api_key=self.config.QDRANT_API_KEY
        )
        
        self.cosmos_client = pymongo.MongoClient(
            self.config.COSMOS_CONNECTION_STRING
        )
        self.database = self.cosmos_client[self.config.COSMOS_DATABASE]
        self.container = self.database[self.config.COSMOS_CONTAINER]
    
    def is_populated(self) -> bool:
        """Check if Qdrant collections have data"""
        try:
            for collection_name in self.config.QDRANT_COLLECTIONS.values():
                try:
                    # Check if the collection exists and has points
                    count_result = self.qdrant_client.count(collection_name=collection_name)
                    if count_result.count > 0:
                        return True
                except Exception:
                    continue
            return False
        except Exception as e:
            logger.warning(f"Error checking Qdrant collections: {e}")
            return False

    def classify_intent(self, query: str) -> str:
        """Classify the intent of the query - more permissive"""
        query_lower = query.lower()
        
        # Table lookup patterns (expanded)
        table_patterns = ["table", "category", "dre", "impairment %", "percentage", "rating", "scale", "level", "grade"]
        if any(pattern in query_lower for pattern in table_patterns):
            return "table_lookup"
        
        # Formula patterns (expanded)
        formula_patterns = ["calculate", "formula", "computation", "whole person", "combined", "wpi", "impairment rating", "composite"]
        if any(pattern in query_lower for pattern in formula_patterns):
            return "formula_calculation"
        
        # Comparison patterns
        comparison_patterns = ["compare", "difference", "versus", "vs", "between", "versus"]
        if any(pattern in query_lower for pattern in comparison_patterns):
            return "comparison"
        
        # Default to concept explanation with multi-collection search
        return "concept_explanation"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_DEPLOYMENT,
                input=text[:8000]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * self.config.EMBEDDING_DIMENSIONS
    
    async def semantic_search(self, query: str, collection: str, limit: int = 5) -> List[Dict]:
        """Perform semantic search in Qdrant with detailed logging"""
        try:
            query_embedding = await self.generate_embedding(query)

            # qdrant-client >= 1.7 uses query_points(); older versions use search()
            try:
                response = self.qdrant_client.query_points(
                    collection_name=collection,
                    query=query_embedding,
                    limit=limit,
                    score_threshold=0.3,
                )
                results = response.points
            except AttributeError:
                # Fallback for qdrant-client < 1.7
                results = self.qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=0.3,
                )

            logger.info(f"🔍 Semantic search in '{collection}': Found {len(results)} results for query '{query}'")

            if results:
                for i, hit in enumerate(results):
                    logger.debug(f"  Result {i+1}: score={hit.score:.3f}, type={hit.payload.get('type', 'unknown')}")

            return [hit.payload for hit in results]
        except Exception as e:
            logger.error(f"❌ Semantic search failed in '{collection}': {e}")
            return []
    
    def prune_irrelevant_sentences(self, text: str, query: str, max_removal_ratio: float = 0.3) -> str:
        """Remove sentences that don't match query context to prevent chunk contamination.
        
        Args:
            text: Text to prune
            query: Original query for relevance checking
            max_removal_ratio: Maximum ratio of text to remove (safety limit)
        
        Returns:
            Pruned text or original if removal would exceed threshold
        """
        if not text or not query:
            return text
        
        query_terms = set(query.lower().split())
        # Remove common stopwords from query
        stopwords = {"the", "a", "an", "and", "or", "to", "in", "of", "for", "is", "are", "as", "by"}
        query_terms = {term for term in query_terms if term not in stopwords and len(term) > 2}
        
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_terms = set(sentence.lower().split())
            # Check if sentence shares significant terms with query or has high term overlap
            overlap = len(query_terms & sentence_terms)
            is_contextual = overlap > 0 or any(
                term in sentence.lower() for term in query_terms
            )
            
            if is_contextual or not query_terms:  # Keep if matches query or no filtering needed
                relevant_sentences.append(sentence)
        
        pruned_text = " ".join(relevant_sentences)
        
        # Safety check: don't remove more than max_removal_ratio
        if len(pruned_text) < len(text) * (1 - max_removal_ratio):
            logger.warning(f"⚠️ Pruning would remove >{max_removal_ratio*100}% of text. Keeping original.")
            return text
        
        removed_count = len(sentences) - len(relevant_sentences)
        if removed_count > 0:
            logger.info(f"  ✂️ Removed {removed_count} irrelevant sentence(s)")
        
        return pruned_text
    
    def filter_results_by_section(self, results: List[Dict], preferred_sections: List[str]) -> List[Dict]:
        """Filter retrieval results to preferred sections to prevent contamination.
        
        Args:
            results: Retrieved results
            preferred_sections: List of section IDs/names to prioritize (e.g., ["2.5f", "2.5"])
        
        Returns:
            Filtered results prioritizing preferred sections
        """
        if not preferred_sections or not results:
            return results
        
        # Separate results by whether they match preferred sections
        matched = []
        other = []
        
        for result in results:
            section = result.get("section", "") or result.get("section_title", "")
            section_lower = section.lower()
            
            # Check if this result's section is in preferred_sections
            if any(pref.lower() in section_lower for pref in preferred_sections):
                matched.append(result)
            else:
                other.append(result)
        
        # Return matched results first, then others if needed
        filtered = matched + other
        if len(filtered) < len(results):
            logger.info(f"  🎯 Filtered to {len(matched)} results from preferred sections: {preferred_sections}")
        
        return filtered
    
    def detect_context_completeness(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """Detect if context is complete/sufficient for query or if evidence boundaries should be highlighted.
        
        Returns:
            Dict with keys: is_complete, confidence, suggestions
        """
        if not results:
            return {"is_complete": False, "confidence": 0.0, "reason": "no_results"}
        
        # Score based on number and type of results
        total_score = 0
        doc_count = len(results)
        
        # More results = more likely complete
        if doc_count >= 3:
            total_score += 0.4
        elif doc_count >= 2:
            total_score += 0.2
        
        # Check result quality (tables + formulas are high-quality)
        has_table = any(r.get("type") == "table" for r in results)
        has_formula = any(r.get("type") == "formula" for r in results)
        has_example = any("example" in r.get("text", "").lower() for r in results)
        
        if has_table:
            total_score += 0.3
        if has_formula:
            total_score += 0.2
        if has_example:
            total_score += 0.1
        
        # Check for section variety (multiple sections = better coverage)
        sections = set(r.get("section", r.get("section_title", "")) for r in results)
        if len(sections) >= 2:
            total_score += 0.2
        
        # Confidence threshold: 0.7 = complete, < 0.5 = show boundary
        is_complete = total_score >= 0.7
        
        return {
            "is_complete": is_complete,
            "confidence": min(total_score, 1.0),
            "reason": "sufficient_coverage" if is_complete else "incomplete_coverage",
            "has_examples": has_example,
            "doc_count": doc_count,
            "section_variety": len(sections)
        }
    
    def extract_section_info(self, result: Dict) -> Tuple[str, str]:
        """Extract section number/title from a result for better citations.
        
        Returns:
            Tuple of (section_id, section_title)
        """
        section_id = result.get("section", result.get("section_id", "Unknown"))
        section_title = result.get("section_title", result.get("title", "Unknown Section"))
        return section_id, section_title
    
    async def deterministic_lookup(self, table_id: str) -> Optional[Table]:
        """Fetch exact table by ID from Cosmos DB"""
        try:
            response = self.container.find_one({"_id": table_id, "content_type": "table"})
            if response and response.get("content_type") == "table":
                return Table(**response["data"])
            return None
        except Exception as e:
            logger.error(f"Deterministic lookup failed: {e}")
            return None
    
    async def retrieve_table_direct(self, query: str) -> List[Dict]:
        """Direct table retrieval for table-specific queries"""
        query_lower = query.lower()
        
        # Try to extract table number
        import re
        table_match = re.search(r'table\s+(\d+[-\.]\d+)', query_lower)
        if table_match:
            table_id = f"tbl_{table_match.group(1)}"
            table = await self.deterministic_lookup(table_id)
            if table:
                return [{
                    "type": "table",
                    "content": table.to_natural_language(),
                    "table_id": table.table_id,
                    "pages": table.pages,
                    "data": table
                }]
        
        # Fall back to semantic search on tables
        return await self.semantic_search(query, self.config.QDRANT_COLLECTIONS["tables"])
    
    async def retrieve_formula_with_context(self, query: str) -> Dict:
        """Retrieve formula and related tables for calculations"""
        # Step 1: Find relevant formulas
        formula_results = await self.semantic_search(
            query, 
            self.config.QDRANT_COLLECTIONS["formulas"],
            limit=2
        )
        
        if not formula_results:
            return {"error": "No formula found"}
        
        # Step 2: Get formula details from Cosmos
        formula_id = formula_results[0].get("formula_id")
        if formula_id:
            try:
                formula_doc = self.container.find_one({"_id": formula_id, "content_type": "formula"})
                if formula_doc is None:
                    raise Exception("Formula not found")
                formula = Formula(**formula_doc["data"])
                
                # Step 3: Fetch related tables
                tables = []
                for table_id in formula.related_tables:
                    table = await self.deterministic_lookup(table_id)
                    if table:
                        tables.append(table)
                
                return {
                    "type": "formula",
                    "formula": formula,
                    "tables": tables,
                    "explanation": formula.to_natural_language()
                }
            except Exception as e:
                logger.error(f"Formula retrieval failed: {e}")
        
        return formula_results[0] if formula_results else {}
    
    async def retrieve_comparison_context(self, query: str) -> List[Dict]:
        """Retrieve multiple items for comparison"""
        # Search multiple collections
        paragraphs = await self.semantic_search(
            query,
            self.config.QDRANT_COLLECTIONS["paragraphs"],
            limit=3
        )
        
        tables = await self.semantic_search(
            query,
            self.config.QDRANT_COLLECTIONS["tables"],
            limit=2
        )
        
        return paragraphs + tables
    
    async def retrieve_calculation_context(self, query: str) -> Dict:
        """Complete calculation context with formula, tables, and examples"""
        # Step 1: Identify formula
        formula_context = await self.retrieve_formula_with_context(query)
        
        if "error" in formula_context:
            return formula_context
        
        # Step 2: Find calculation examples from paragraphs
        examples = await self.semantic_search(
            f"example calculation {query}",
            self.config.QDRANT_COLLECTIONS["paragraphs"],
            limit=2
        )
        
        # Step 3: Generate calculation steps
        formula = formula_context.get("formula")
        if formula:
            steps = self.generate_calculation_steps(formula, query)
            formula_context["calculation_steps"] = steps
        
        formula_context["examples"] = examples
        
        return formula_context
    
    def generate_calculation_steps(self, formula: Formula, query: str) -> List[str]:
        """Generate step-by-step calculation guide"""
        steps = []
        
        steps.append(f"Step 1: Identify the correct formula: {formula.formula_text}")
        
        # Add variable explanation
        if formula.variables:
            steps.append("Step 2: Determine variable values:")
            for var, desc in formula.variables.items():
                steps.append(f"  - {var}: {desc}")
        
        # Add conditions
        if formula.conditions:
            steps.append("Step 3: Check conditions:")
            for condition in formula.conditions:
                steps.append(f"  - {condition}")
        
        # Add calculation
        steps.append("Step 4: Apply the formula to calculate impairment")
        
        # Add example if available
        if formula.example_calculation:
            steps.append(f"Example: {formula.example_calculation}")
        
        return steps
    
    async def retrieve_with_citations(self, query: str) -> Dict[str, Any]:
        """Main retrieval method with citations"""
        # Step 1: Classify intent
        intent = self.classify_intent(query)
        
        # Step 2: Route to appropriate retrieval method
        if intent == "table_lookup":
            results = await self.retrieve_table_direct(query)
            return {
                "intent": intent,
                "results": results,
                "citations": self.extract_citations(results)
            }
        
        elif intent == "formula_calculation":
            results = await self.retrieve_calculation_context(query)
            return {
                "intent": intent,
                "results": results,
                "citations": self.extract_citations([results])
            }
        
        elif intent == "comparison":
            results = await self.retrieve_comparison_context(query)
            return {
                "intent": intent,
                "results": results,
                "citations": self.extract_citations(results)
            }
        
        else:  # concept_explanation - search all collections
            logger.info(f"📚 Multi-collection concept search for: '{query}'")
            
            # Search all collections for comprehensive results
            paragraph_results = await self.semantic_search(
                query,
                self.config.QDRANT_COLLECTIONS["paragraphs"],
                limit=3
            )
            
            section_results = await self.semantic_search(
                query,
                self.config.QDRANT_COLLECTIONS["sections"],
                limit=2
            )
            
            table_results = await self.semantic_search(
                query,
                self.config.QDRANT_COLLECTIONS["tables"],
                limit=2
            )
            
            formula_results = await self.semantic_search(
                query,
                self.config.QDRANT_COLLECTIONS["formulas"],
                limit=1
            )
            
            # Combine all results
            all_results = paragraph_results + section_results + table_results + formula_results
            
            logger.info(f"📊 Total results from all collections: {len(all_results)}")
            
            # Step 3: Prune irrelevant sentences to prevent chunk contamination
            for result in all_results:
                if "text" in result:
                    result["text"] = self.prune_irrelevant_sentences(result["text"], query)
                elif "content" in result:
                    result["content"] = self.prune_irrelevant_sentences(result["content"], query)
            
            return {
                "intent": intent,
                "results": all_results,
                "citations": self.extract_citations(all_results)
            }
    
    def extract_citations(self, results: List[Dict]) -> List[Dict]:
        """Extract citations from results with improved grounding including section numbers"""
        citations = []
        
        for result in results:
            if isinstance(result, dict):
                if "pages" in result:
                    pages = result["pages"]
                    if isinstance(pages, list):
                        pages = pages[0] if pages else None
                    
                    # Improved citation with section ID and title
                    section_id = result.get("section", result.get("section_id", ""))
                    section_title = result.get("section_title", result.get("title", ""))
                    citation = {
                        "type": result.get("type", "unknown"),
                        "section_id": section_id,
                        "section_title": section_title,
                        "pages": pages,
                        "section_reference": self._format_section_reference(section_id, section_title, pages),
                        "content": result.get("content", result.get("text", ""))[:200]
                    }
                    citations.append(citation)
                elif "payload" in result:
                    payload = result["payload"]
                    section_id = payload.get("section", payload.get("section_id", ""))
                    section_title = payload.get("section_title", payload.get("title", ""))
                    citation = {
                        "type": payload.get("type", "unknown"),
                        "section_id": section_id,
                        "section_title": section_title,
                        "pages": payload.get("pages", payload.get("page")),
                        "section_reference": self._format_section_reference(section_id, section_title, payload.get("pages", payload.get("page"))),
                        "content": payload.get("text", "")[:200]
                    }
                    citations.append(citation)
        
        return citations
    
    def _format_section_reference(self, section_id: str, section_title: str, pages: Any) -> str:
        """Format section reference for improved citation grounding with section numbers"""
        # Build reference: [Section 2.5f – Using Assistive Devices – p. XX]
        if section_id and section_title:
            page_str = f" – p. {pages}" if pages else ""
            return f"[Section {section_id} – {section_title}{page_str}]"
        elif section_id:
            page_str = f" – p. {pages}" if pages else ""
            return f"[Section {section_id}{page_str}]"
        elif section_title:
            page_str = f" – p. {pages}" if pages else ""
            return f"[{section_title}{page_str}]"
        else:
            return f"[p. {pages}]" if pages else "[Source location not specified]"
    
    async def answer_query(self, query: str) -> Dict[str, Any]:
        """Generate answer with grounding context, anti-hallucination constraints, and evidence boundaries"""
        # Step 1: Retrieve relevant content
        retrieval_results = await self.retrieve_with_citations(query)
        
        logger.info(f"✅ Retrieved {len(retrieval_results.get('results', []))} documents for query: '{query}'")
        
        # Step 2: Check if we have results
        if not retrieval_results.get("results"):
            logger.warning(f"⚠️ No results found for query: '{query}'")
            return {
                "query": query,
                "answer": f"I was unable to find information about '{query}' in the AMA Guides. Please try a different search term or be more specific about what you're looking for.",
                "citations": [],
                "intent": retrieval_results.get("intent", "unknown"),
                "results_found": 0,
                "evidence_boundary": {"status": "NO_CONTEXT_FOUND", "explicitly_stated": True}
            }
        
        # Step 3: Assess context completeness for adaptive verbosity
        context_completeness = self.detect_context_completeness(retrieval_results["results"], query)
        context_text, evidence_map = self.format_context_with_tracking(retrieval_results["results"])
        
        # Build system prompt with core anti-hallucination constraints
        system_prompt = """You are an expert medical assistant specializing in the AMA Guides to the Evaluation of Permanent Impairment, 5th Edition.

CRITICAL CONSTRAINTS - ANTI-HALLUCINATION:
1. ONLY state differences, facts, or details that are EXPLICITLY mentioned in the provided context.
2. Do NOT infer, assume, or extrapolate information not clearly stated in the context.
3. Do NOT use external knowledge, medical training, or general assumptions.
4. If a detail could be interpreted multiple ways, state that ambiguity explicitly.

CITATION REQUIREMENTS:
1. Use format: [Section 2.5f – p. 45] with actual section numbers and page numbers.
2. Every factual claim MUST be grounded in a specific section reference.
3. If information spans multiple sections, cite all relevant sections.

Be precise, professional, and transparent."""
        
        # Adaptive verbosity: only request evidence boundary discussion if context is incomplete
        if context_completeness["is_complete"]:
            # Context is complete - focus on clean, direct answer
            user_prompt = f"""Context from AMA Guides (marked with section references):
{context_text}

Question: {query}

GUIDELINES:
1. Provide a clear, direct answer based ONLY on the context above.
2. For each claim, cite the specific section and page numbers.
3. Keep the answer focused and avoid unnecessary caveats.
4. If you can fully answer from the context, do so without disclaimers.

Answer directly based on the provided context."""
        else:
            # Context is incomplete - request explicit boundary marking
            user_prompt = f"""Context from AMA Guides (marked with section references):
{context_text}

Question: {query}

GUIDELINES:
1. Answer based ONLY on the context provided.
2. For each claim, cite the specific section and page numbers.
3. If information is MISSING or INCOMPLETE, explicitly state: "The context does not provide [what is missing]"
4. Mark any limitations clearly.
5. Structure as: [Explicit facts] → [What the context doesn't cover] → [If needed: reasonable interpretations]

Be transparent about context boundaries."""
        
        # Step 4: Generate answer
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for stricter adherence to constraints
                max_completion_tokens=1500
            )
            
            answer = response.choices[0].message.content
            
            # Build response with adaptive evidence_boundary
            response_dict = {
                "query": query,
                "answer": answer,
                "citations": retrieval_results["citations"],
                "intent": retrieval_results["intent"],
                "grounding_context": retrieval_results["results"],
                "results_found": len(retrieval_results.get("results", [])),
                "context_quality": {
                    "is_complete": context_completeness["is_complete"],
                    "confidence": context_completeness["confidence"],
                    "doc_count": context_completeness["doc_count"],
                    "has_examples": context_completeness.get("has_examples", False)
                }
            }
            
            # Only include evidence_boundary if context is incomplete (adaptive verbosity)
            if not context_completeness["is_complete"]:
                response_dict["evidence_boundary"] = {
                    "status": "INCOMPLETE_CONTEXT",
                    "explicitly_stated": True,
                    "reason": context_completeness["reason"],
                    "sections_referenced": list(set(evidence_map.values())) if evidence_map else []
                }
            
            return response_dict
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while generating an answer: {str(e)}",
                "error": str(e),
                "citations": retrieval_results["citations"],
                "results_found": len(retrieval_results.get("results", [])),
                "context_quality": {
                    "is_complete": context_completeness["is_complete"],
                    "confidence": context_completeness["confidence"]
                }
            }
    
    def format_context(self, results: List[Dict]) -> str:
        """Format retrieval results for prompt (legacy method)"""
        formatted, _ = self.format_context_with_tracking(results)
        return formatted
    
    def format_context_with_tracking(self, results: List[Dict]) -> Tuple[str, Dict[int, str]]:
        """Format retrieval results with section tracking for evidence boundaries
        
        Returns:
            Tuple of (formatted_context_string, evidence_map[index -> section_id])
        """
        context_parts = []
        evidence_map = {}  # Track which section each result comes from
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                # Extract section number and title
                section_id = result.get("section", result.get("section_id", ""))
                section_title = result.get("section_title", result.get("title", "Unknown Section"))
                page_info = result.get("pages", result.get("page", "N/A"))
                
                # Build citation: [Section 2.5f – Using Assistive Devices – p. XX]
                if section_id:
                    section_ref = f"[Section {section_id}"
                    if section_title and section_title.lower() != "unknown section":
                        section_ref += f" – {section_title}"
                    if page_info != "N/A":
                        section_ref += f" – p. {page_info}"
                    section_ref += "]"
                    evidence_map[i] = f"{section_id}: {section_title}"
                else:
                    # Fallback if no section_id
                    section_ref = f"[{section_title}"
                    if page_info != "N/A":
                        section_ref += f" – p. {page_info}"
                    section_ref += "]"
                    evidence_map[i] = section_title
                
                # Extract content
                content_text = ""
                if "content" in result:
                    content_text = result['content']
                elif "text" in result:
                    content_text = result['text']
                elif "payload" in result and "text" in result["payload"]:
                    content_text = result['payload']['text']
                elif "formula" in result:
                    content_text = result['formula'].to_natural_language()
                
                # Mark result type for clarity
                result_type = result.get("type", "content").upper()
                type_marker = f"[{result_type}]" if result_type != "CONTENT" else ""
                
                # Combine: [Section 2.5f – Title – p. XX] [TYPE]
                # Content
                formatted_result = f"{section_ref}"
                if type_marker:
                    formatted_result += f" {type_marker}"
                formatted_result += f"\n{content_text}"
                
                context_parts.append(formatted_result)
        
        return "\n\n".join(context_parts), evidence_map
