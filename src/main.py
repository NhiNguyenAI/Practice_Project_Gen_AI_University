import os
import streamlit as st
from langchain.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory

class RobustChunkingAgent:
    def __init__(self):
        self.llm = Ollama(model="llama2", temperature=0.7)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=['\n\n', '\n', '(?<=\. )', ' ']
        )
        
    def group_by_context(self, text):
        """Intelligently group text chunks with error handling"""
        chunks = self.splitter.split_text(text)
        
        tools = [
            Tool(
                name="ContextAnalyzer",
                func=self.analyze_relationship,
                description="Determines if text segments should be grouped (respond ONLY with 'yes' or 'no')"
            )
        ]
        
        agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,  # Critical for robustness
            max_iterations=3,  # Prevent infinite loops
            early_stopping_method="generate"
        )
        
        grouped_chunks = []
        current_group = []
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                try:
                    response = agent.run(
                        f"Analyze ONLY whether these two text segments should be grouped. "
                        f"Respond ONLY with 'yes' or 'no'.\n"
                        f"Segment 1: {chunks[i-1][:200]}...\n"
                        f"Segment 2: {chunk[:200]}...\n"
                        "Answer:"
                    )
                    
                    # Normalize response
                    response = response.strip().lower()
                    if "yes" in response:
                        current_group.append(chunk)
                    else:
                        if current_group:
                            grouped_chunks.append("\n\n".join(current_group))
                        current_group = [chunk]
                        
                except Exception as e:
                    st.warning(f"⚠️ Analysis skipped for chunk {i}: {str(e)}")
                    current_group.append(chunk)  # Default to grouping on error
            else:
                current_group.append(chunk)
        
        if current_group:
            grouped_chunks.append("\n\n".join(current_group))
            
        return grouped_chunks
    
    def analyze_relationship(self, input_text):
        """Strict relationship analysis with output control"""
        response = self.llm(input_text + "\nRespond ONLY with 'yes' or 'no':")
        return response[:3].strip().lower()  # Force short response

# Streamlit UI Setup
st.title("UniBox: Robust Research Assistant 🛡️")
st.sidebar.title("Configuration")

# Settings with validation
model_name = st.sidebar.selectbox("LLM Model", ["llama2", "mistral"])
chunk_size = st.sidebar.slider("Base Chunk Size", 400, 1200, 800, step=50)
temperature = st.sidebar.slider("Creativity", 0.1, 1.0, 0.5, step=0.1)

# Document Processing
urls = [st.text_input(f"Source URL {i+1}") for i in range(3)]
process_btn = st.button("Process Documents")

if process_btn and any(urls):
    with st.spinner("🔍 Processing with enhanced error handling..."):
        try:
            # Load documents with timeout
            loader = UnstructuredURLLoader(urls=urls, timeout=30)
            raw_docs = loader.load()
            
            if not raw_docs:
                st.error("No content loaded from URLs")
                st.stop()
            
            # Process with robust agent
            chunk_agent = RobustChunkingAgent()
            processed_docs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, doc in enumerate(raw_docs):
                status_text.text(f"Processing document {i+1}/{len(raw_docs)}...")
                grouped_chunks = chunk_agent.group_by_context(doc.page_content)
                
                for chunk in grouped_chunks:
                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "group_id": f"doc_{i}_group_{len(processed_docs)}",
                            "source": doc.metadata.get("source", "direct_input")
                        }
                    ))
                progress_bar.progress((i + 1) / len(raw_docs))
            
            # Create and save vector store
            embeddings = OllamaEmbeddings(
                model=model_name,
                timeout=60  # Increased timeout
            )
            
            with st.spinner("Building semantic index..."):
                vectorstore = FAISS.from_documents(processed_docs, embeddings)
                vectorstore.save_local("robust_index")
            
            st.success(f"✅ Successfully processed {len(processed_docs)} contextual groups!")
            
            # Show sample output
            with st.expander("Sample Document Groups"):
                cols = st.columns(2)
                for i, group in enumerate(processed_docs[:4]):
                    cols[i%2].text_area(
                        f"Group {i+1} ({group.metadata['source'][:20]}...)",
                        group.page_content[:300] + "...",
                        height=150
                    )
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.info("Try reducing chunk size or checking URL accessibility")

# Query System
if os.path.exists("robust_index"):
    st.divider()
    query = st.text_input("Enter your research question:", key="query")
    
    if query:
        with st.spinner("🧠 Generating response..."):
            try:
                # Load index with validation
                embeddings = OllamaEmbeddings(model=model_name)
                if not os.path.exists("robust_index/index.faiss"):
                    st.error("Index files not found")
                    st.stop()
                
                vectorstore = FAISS.load_local("robust_index", embeddings)
                
                # Configure reliable QA agent
                qa_tool = Tool(
                    name="DocumentQA",
                    func=lambda q: RetrievalQA.from_chain_type(
                        llm=Ollama(
                            model=model_name,
                            temperature=temperature,
                            num_ctx=2048  # Increased context window
                        ),
                        chain_type="map_reduce",
                        retriever=vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={"k": 3, "fetch_k": 10}
                        ),
                        return_source_documents=True
                    )(q),
                    description="Answers questions based on processed documents"
                )
                
                agent = initialize_agent(
                    [qa_tool],
                    Ollama(model=model_name, temperature=temperature),
                    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=ConversationBufferMemory(memory_key="chat_history"),
                    handle_parsing_errors=True,
                    max_execution_time=30,
                    verbose=True
                )
                
                # Execute with timeout handling
                result = agent.run({
                    "input": query[:500],  # Truncate very long queries
                    "timeout": 20  # Seconds
                })
                
                st.subheader("📚 Answer")
                st.write(result.get("output", result))
                
                # Show sources
                if hasattr(result, "source_documents"):
                    st.subheader("🔍 Source Contexts")
                    for doc in result.source_documents:
                        with st.expander(f"Source: {doc.metadata.get('source', 'unknown')}"):
                            st.write(doc.page_content[:500] + "...")
                
            except Exception as e:
                st.error(f"⚠️ Query failed: {str(e)}")
                st.info("Try simplifying your question or reprocessing documents")