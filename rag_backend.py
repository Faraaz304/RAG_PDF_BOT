import os
import warnings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_document(file_path):
    """Load PDF or TXT document"""
    print(f"Loading document: {file_path}")
    
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Use .pdf or .txt files only")
    
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def split_text(documents):
    """Split documents into chunks"""
    print("Splitting text into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_embeddings():
    """Create embedding model"""
    return OllamaEmbeddings(model="llama3.2")

def create_vectorstore(chunks, embeddings):
    """Create vector database"""
    print("Creating vector database...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Vector store created!")
    return vectorstore

def create_llm():
    """Create language model"""
    return ChatOllama(model="llama3.2", temperature=0.2)

def setup_qa_chain(vectorstore, llm):
    """Setup question-answering chain"""
    print("Setting up QA system...")
    
    prompt_template = """
    Use the context below to answer the question.
    If you don't know the answer, say you don't know.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    print("QA system ready!")
    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question to the system"""
    print(f"\nQuestion: {question}")
    print("Searching for answer...")
    
    # result = qa_chain({"query": question})
    result = qa_chain.invoke({"query": question})
    
    answer = result["result"]
    sources = len(result["source_documents"])
    
    print(f"\nAnswer: {answer}")
    print(f"Used {sources} source chunks")
    
    return answer

def main():
    """Main pipeline"""
    print("Simple RAG System with Llama 3.2")
    print("=" * 40)
    
    # Step 1: Get document path
    file_path = input("Enter your document path: ").strip()
    
    if not os.path.exists(file_path):
        print("File not found!")
        return
    
    try:
        # Step 2: Load document
        documents = load_document(file_path)
        
        # Step 3: Split into chunks
        chunks = split_text(documents)
        
        # Step 4: Create embeddings
        embeddings = create_embeddings()
        
        # Step 5: Create vector store
        vectorstore = create_vectorstore(chunks, embeddings)
        
        # Step 6: Create LLM
        llm = create_llm()
        
        # Step 7: Setup QA chain
        qa_chain = setup_qa_chain(vectorstore, llm)
        
        print("\n✅ Setup complete! Ready for questions!")
        print("-" * 40)
        
        # Step 8: Interactive Q&A
        while True:
            question = input("\nYour question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            ask_question(qa_chain, question)
        
        print("\nGoodbye!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# doucment = load_document("D:\Faraaz\AI\Langchain\pdf lanchain bot\JARO EDUCATION .pdf")
# chunks = split_text(doucment)
# print(chunks[0].page_content)
# vector = create_vectorstore(chunks, create_embeddings())
# llm = create_llm()
# qa_chain = setup_qa_chain(vector, llm)
# question = "What is the mission of Jaro Education?"
# answer = ask_question(qa_chain, question)
# print(f"Answer: {answer}")