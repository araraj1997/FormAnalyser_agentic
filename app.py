"""
Streamlit Web Interface for Intelligent Form Agent

An interactive web UI for processing forms with LLM-powered intelligence.
"""

import streamlit as st
import json
import os
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import IntelligentFormAgent, ProcessedForm


# Page config
st.set_page_config(
    page_title="Intelligent Form Agent",
    page_icon="📄",
    layout="wide"
)


def init_session_state():
    """Initialize session state."""
    if "agent" not in st.session_state:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            st.session_state.agent = IntelligentFormAgent(api_key=api_key, verbose=True)
        else:
            st.session_state.agent = None
    
    if "forms" not in st.session_state:
        st.session_state.forms = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "llm_calls" not in st.session_state:
        st.session_state.llm_calls = 0


def render_api_key_input():
    """Render API key input if not set."""
    st.sidebar.header("🔑 API Configuration")
    
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Enter your Anthropic API key"
    )
    
    if api_key and (st.session_state.agent is None or st.sidebar.button("Update API Key")):
        st.session_state.agent = IntelligentFormAgent(api_key=api_key, verbose=True)
        st.sidebar.success("✅ Agent initialized!")
    
    return api_key


def render_sidebar():
    """Render sidebar with settings and document list."""
    with st.sidebar:
        st.header("📚 Loaded Forms")
        
        if st.session_state.forms:
            for i, form in enumerate(st.session_state.forms):
                with st.expander(f"📄 {Path(form.file_path).name}"):
                    st.write(f"**Type:** {form.form_type}")
                    st.write(f"**Confidence:** {form.extraction_confidence:.1%}")
                    st.write(f"**Fields:** {len(form.extracted_fields)}")
                    
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.forms.pop(i)
                        st.rerun()
        else:
            st.info("No forms loaded yet")
        
        st.divider()
        
        # Statistics
        st.header("📊 Statistics")
        if st.session_state.agent:
            stats = st.session_state.agent.get_stats()
            st.metric("LLM Calls", stats["total_llm_calls"])
            st.metric("Cached Forms", stats["cached_forms"])
        
        # Clear button
        if st.button("🗑️ Clear All"):
            st.session_state.forms = []
            st.session_state.chat_history = []
            if st.session_state.agent:
                st.session_state.agent.clear_cache()
            st.rerun()


def render_upload_section():
    """Render file upload section."""
    st.header("📤 Upload Forms")
    
    uploaded_files = st.file_uploader(
        "Upload form documents",
        type=["pdf", "png", "jpg", "jpeg", "txt", "json", "csv"],
        accept_multiple_files=True,
        help="Supported: PDF, images, text files"
    )
    
    if uploaded_files and st.button("🔄 Process Forms", type="primary"):
        if not st.session_state.agent:
            st.error("Please enter your API key first!")
            return
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status.text(f"Processing: {uploaded_file.name}...")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                form = st.session_state.agent.process_form(tmp_path)
                form.file_path = uploaded_file.name  # Use original name
                st.session_state.forms.append(form)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            finally:
                os.unlink(tmp_path)
            
            progress.progress((i + 1) / len(uploaded_files))
        
        status.text("✅ Processing complete!")
        st.rerun()


def render_qa_section():
    """Render Q&A section."""
    st.header("❓ Ask Questions")
    
    if not st.session_state.forms:
        st.warning("Please upload and process some forms first.")
        return
    
    if not st.session_state.agent:
        st.error("Please enter your API key!")
        return
    
    # Form selection
    form_names = [Path(f.file_path).name for f in st.session_state.forms]
    selected = st.multiselect("Select forms to query", form_names, default=form_names)
    
    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is the total income? Who is the applicant?"
    )
    
    if st.button("🔍 Ask", type="primary") and question:
        selected_forms = [f for f in st.session_state.forms 
                         if Path(f.file_path).name in selected]
        
        with st.spinner("Thinking..."):
            if len(selected_forms) == 1:
                result = st.session_state.agent.ask(question, selected_forms[0])
                
                st.success(f"**Answer:** {result.answer}")
                
                col1, col2 = st.columns(2)
                with col1:
                    confidence_color = "green" if result.confidence > 0.7 else "orange"
                    st.markdown(f"**Confidence:** :{confidence_color}[{result.confidence:.1%}]")
                with col2:
                    if result.evidence:
                        with st.expander("📋 Evidence"):
                            for e in result.evidence:
                                st.write(f"• {e}")
            else:
                result = st.session_state.agent.analyze(question, selected_forms)
                
                st.success(f"**Answer:** {result.answer}")
                
                if result.insights:
                    st.subheader("💡 Insights")
                    for insight in result.insights:
                        st.write(f"• {insight}")
                
                if result.statistics:
                    st.subheader("📊 Statistics")
                    st.json(result.statistics)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result.answer if hasattr(result, "answer") else str(result)
        })


def render_summary_section():
    """Render summary section."""
    st.header("📝 Generate Summary")
    
    if not st.session_state.forms:
        st.warning("Please upload forms first.")
        return
    
    if not st.session_state.agent:
        st.error("Please enter your API key!")
        return
    
    form_names = [Path(f.file_path).name for f in st.session_state.forms]
    selected = st.selectbox("Select form to summarize", form_names)
    style = st.selectbox("Summary style", ["detailed", "brief", "bullet_points"])
    
    if st.button("📄 Generate Summary", type="primary"):
        form = next(f for f in st.session_state.forms 
                   if Path(f.file_path).name == selected)
        
        with st.spinner("Generating summary..."):
            summary = st.session_state.agent.summarize(form, style=style)
        
        st.subheader("Summary")
        st.write(summary.summary)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Points")
            for point in summary.key_points:
                st.write(f"• {point}")
        
        with col2:
            st.subheader("Important Values")
            for k, v in summary.important_values.items():
                st.write(f"**{k}:** {v}")


def render_extraction_section():
    """Render extraction details section."""
    st.header("🔍 Extraction Details")
    
    if not st.session_state.forms:
        st.warning("No forms loaded.")
        return
    
    form_names = [Path(f.file_path).name for f in st.session_state.forms]
    selected = st.selectbox("Select form", form_names, key="extract_select")
    
    form = next(f for f in st.session_state.forms 
               if Path(f.file_path).name == selected)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fields", "Raw Text", "Tables", "Export"])
    
    with tab1:
        st.subheader("Extracted Fields")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Form Type", form.form_type or "Unknown")
        with col2:
            st.metric("Confidence", f"{form.extraction_confidence:.1%}")
        
        st.json(form.extracted_fields)
    
    with tab2:
        st.subheader("Raw Text")
        st.text_area("", form.raw_text, height=400)
    
    with tab3:
        st.subheader("Tables")
        if form.tables:
            for i, table in enumerate(form.tables):
                st.write(f"**Table {i+1}**")
                if table:
                    import pandas as pd
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        st.dataframe(df)
                    except:
                        st.write(table)
        else:
            st.info("No tables found")
    
    with tab4:
        st.subheader("Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download JSON",
                form.to_json(),
                f"{Path(form.file_path).stem}_extracted.json",
                "application/json"
            )
        
        with col2:
            if st.session_state.agent:
                if st.button("📥 Generate Report"):
                    report = st.session_state.agent.export_summary(form)
                    st.download_button(
                        "Download Report",
                        report,
                        f"{Path(form.file_path).stem}_report.md",
                        "text/markdown"
                    )


def render_analysis_section():
    """Render cross-form analysis section."""
    st.header("📊 Cross-Form Analysis")
    
    if len(st.session_state.forms) < 2:
        st.warning("Upload at least 2 forms for cross-form analysis.")
        return
    
    if not st.session_state.agent:
        st.error("Please enter your API key!")
        return
    
    question = st.text_input(
        "Analysis question:",
        placeholder="e.g., Compare salaries across all employees"
    )
    
    if st.button("🔬 Analyze", type="primary") and question:
        with st.spinner("Analyzing..."):
            result = st.session_state.agent.analyze(question, st.session_state.forms)
        
        st.success(f"**Answer:** {result.answer}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💡 Insights")
            for insight in result.insights:
                st.info(insight)
        
        with col2:
            if result.statistics:
                st.subheader("📈 Statistics")
                st.json(result.statistics)
        
        if result.comparisons:
            st.subheader("🔄 Comparisons")
            st.json(result.comparisons)


def main():
    """Main app."""
    init_session_state()
    
    st.title("📄 Intelligent Form Agent")
    st.markdown("*LLM-powered form processing, Q&A, and analysis*")
    
    # API key input
    api_key = render_api_key_input()
    
    if not api_key and not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to get started.")
        st.info("""
        **Getting Started:**
        1. Enter your Anthropic API key in the sidebar
        2. Upload form documents (PDF, images, or text)
        3. Ask questions, generate summaries, or analyze multiple forms
        
        The agent uses Claude to intelligently understand and extract information from any form type.
        """)
        return
    
    # Sidebar
    render_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload",
        "❓ Q&A",
        "📝 Summary",
        "🔍 Details",
        "📊 Analysis"
    ])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_qa_section()
    
    with tab3:
        render_summary_section()
    
    with tab4:
        render_extraction_section()
    
    with tab5:
        render_analysis_section()


if __name__ == "__main__":
    main()
