import streamlit as st
import asyncio
import tempfile
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

from ingestion_pipeline import AMAGuidesIngestionPipeline
from retrieval_engine import AMARetrievalEngine

# Page configuration
st.set_page_config(
    page_title="AMA Guides Ingestion System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #3B82F6;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
    }
    .error-box {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
    }
    .info-box {
        background-color: #DBEAFE;
        border-left: 5px solid #3B82F6;
    }
    .citation {
        font-size: 0.85rem;
        color: #6B7280;
        border-left: 2px solid #3B82F6;
        padding-left: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'retrieval_engine' not in st.session_state:
    st.session_state.retrieval_engine = AMARetrievalEngine()
if 'ingestion_complete' not in st.session_state:
    st.session_state.ingestion_complete = st.session_state.retrieval_engine.is_populated()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class StreamlitProgress:
    """Progress tracking for Streamlit UI"""
    def __init__(self):
        self.progress_bar = None
        self.status_text = None
    
    def update(self, progress):
        if self.progress_bar:
            self.progress_bar.progress(progress.processed_pages / max(progress.total_pages, 1))
        if self.status_text:
            self.status_text.text(progress.current_operation)

async def run_ingestion_with_progress(pdf_bytes, progress_callback):
    """Run ingestion pipeline with progress updates"""
    pipeline = AMAGuidesIngestionPipeline()
    await pipeline.run_pipeline(pdf_bytes, progress_callback=progress_callback)
    return pipeline

def main():
    # Sidebar for configuration
    with st.sidebar:

        st.markdown("### System Status")
        
        if st.session_state.ingestion_complete:
            st.success("✅ AMA Guides Ingested")
        else:
            st.warning("⚠️ No document ingested yet")
        
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("""
        - 📄 Intelligent PDF parsing
        - 📊 Multi-page table extraction
        - 🧮 Formula detection
        - 🔍 Hybrid retrieval
        - 📝 Citation validation
        - 💬 Contextual Q&A
        """)
    
    # Main content
    st.markdown('<div class="main-header">AMA Guides Ingestion System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Document Processing for AMA Guides 5th Edition</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📥 Ingest Document", "💬 Ask Questions", "📊 Statistics"])
    
    # Tab 1: Ingest Document
    with tab1:
        st.markdown("### Upload AMA Guides PDF")
        st.markdown("Upload the AMA Guides 5th Edition PDF for ingestion into the system.")
        
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=['pdf'],
            help="Upload AMA Guides 5th Edition PDF"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if uploaded_file and st.button("🚀 Start Ingestion", type="primary"):
                with st.spinner("Processing..."):
                    # Create progress placeholders
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    page_text = st.empty()

                    def progress_callback(progress):
                        progress_value = progress.processed_pages / max(progress.total_pages, 1)
                        progress_bar.progress(progress_value)
                        status_text.text(f"{progress.current_operation} (page {progress.current_page}/{progress.total_pages})")
                        if progress.pages_processed:
                            page_text.text(f"Pages processed: {', '.join(map(str, progress.pages_processed))}")
                        else:
                            page_text.text("Pages processed: 0")

                    try:
                        pipeline = asyncio.run(run_ingestion_with_progress(uploaded_file.getvalue(), progress_callback))
                        st.session_state.pipeline = pipeline
                        st.session_state.ingestion_complete = True
                        st.session_state.retrieval_engine = AMARetrievalEngine()

                        status_text.text("✅ Ingestion complete!")
                        progress_bar.progress(1.0)
                        st.success("✅ AMA Guides successfully ingested!")

                        # Show summary
                        with st.expander("Ingestion Summary", expanded=True):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Tables Extracted", pipeline.progress.tables_found)
                            with col_b:
                                st.metric("Formulas Found", pipeline.progress.formulas_found)
                            with col_c:
                                st.metric("Paragraphs Extracted", pipeline.progress.paragraphs_extracted)

                            if pipeline.progress.errors:
                                st.warning(f"⚠️ {len(pipeline.progress.errors)} errors occurred")
                                for error in pipeline.progress.errors[-5:]:
                                    st.code(error)

                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")
                        status_text.text("❌ Ingestion failed")
        
        with col2:
            if st.session_state.ingestion_complete:
                st.markdown("### Ingestion Status")
                st.markdown("✅ Document is ready for querying")
                st.markdown("💡 You can now ask questions about the AMA Guides in the 'Ask Questions' tab")
    
    # Tab 2: Ask Questions
    with tab2:
        if not st.session_state.ingestion_complete:
            st.warning("⚠️ Please ingest a document first in the 'Ingest Document' tab")
        else:
            st.markdown("### Ask Questions about AMA Guides")
            st.markdown("Ask questions about impairment ratings, calculations, or specific conditions.")
            
            # Example questions
            with st.expander("Example Questions"):
                st.markdown("""
                - What is the impairment rating for DRE Category II lumbosacral?
                - How do you calculate whole person impairment?
                - What's the difference between upper and lower extremity impairment?
                - Show me Table 5-3 for spine impairment
                - Calculate combined impairment for upper extremity 25% and lower extremity 15%
                """)
            
            # Query input
            query = st.text_input("Your Question", placeholder="e.g., What is the impairment rating for DRE Category II?")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🔍 Ask", type="primary", use_container_width=True):
                    if query:
                        with st.spinner("Retrieving answer..."):
                            try:
                                async def get_answer():
                                    engine = st.session_state.retrieval_engine
                                    return await engine.answer_query(query)
                                
                                result = asyncio.run(get_answer())
                                
                                # Store in chat history
                                st.session_state.chat_history.append({
                                    "query": query,
                                    "answer": result.get("answer", "No answer generated"),
                                    "citations": result.get("citations", []),
                                    "intent": result.get("intent", "unknown"),
                                    "timestamp": datetime.now()
                                })
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("### Conversation History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
                    with st.container():
                        st.markdown(f"**You:** {chat['query']}")
                        st.markdown(f"**Assistant:** {chat['answer']}")
                        
                        # Show citations
                        if chat.get('citations'):
                            with st.expander(f"📚 Citations ({len(chat['citations'])})"):
                                for citation in chat['citations']:
                                    st.markdown(f"""
                                    <div class="citation">
                                    <strong>Type:</strong> {citation.get('type', 'unknown')}<br>
                                    <strong>Pages:</strong> {citation.get('pages', 'N/A')}<br>
                                    <strong>Content:</strong> {citation.get('content', '')[:200]}...
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.markdown(f"<small>Intent: {chat.get('intent', 'N/A')} | {chat['timestamp'].strftime('%H:%M:%S')}</small>", unsafe_allow_html=True)
                        st.markdown("---")
    
    # Tab 3: Statistics
    with tab3:
        if not st.session_state.ingestion_complete:
            st.warning("⚠️ Please ingest a document first to see statistics")
        else:
            st.markdown("### Ingestion Statistics")
            
            pipeline = st.session_state.pipeline
            
            if pipeline:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pages", pipeline.progress.total_pages)
                with col2:
                    st.metric("Tables Found", pipeline.progress.tables_found)
                with col3:
                    st.metric("Formulas Found", pipeline.progress.formulas_found)
                with col4:
                    st.metric("Paragraphs", pipeline.progress.paragraphs_chunked)
                
                # Progress chart
                st.markdown("### Processing Progress")
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Processed Pages', 'Tables', 'Formulas', 'Paragraphs'],
                        y=[
                            pipeline.progress.processed_pages,
                            pipeline.progress.tables_found,
                            pipeline.progress.formulas_found,
                            pipeline.progress.paragraphs_chunked
                        ],
                        marker_color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
                    )
                ])
                fig.update_layout(title="Extracted Content", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Errors
                if pipeline.progress.errors:
                    st.markdown("### Errors")
                    for error in pipeline.progress.errors:
                        st.error(error)
                
                # Page detail
                st.markdown("### Pages Processed")
                st.write(pipeline.progress.pages_processed)

                # Status
                st.markdown("### System Status")
                st.json({
                    "ingestion_complete": st.session_state.ingestion_complete,
                    "last_operation": pipeline.progress.current_operation,
                    "total_errors": len(pipeline.progress.errors)
                })

if __name__ == "__main__":
    main()
