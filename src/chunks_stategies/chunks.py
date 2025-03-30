"""
************************************************************************
 *
 * chunks.py
 *
 * Initial Creation:
 *    Author      Nhi Nguyen
 *    Created on  2025-27-03
 *
 ************************************************************************
"""
#--------------------------------------------------------------------------------#
#                                 Libraries                                      #
#--------------------------------------------------------------------------------#


from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

local_llm = ChatOllama(model="mistral")

#--------------------------------------------------------------------------------#
#                            Retrieval Augmented Generation                      #
#--------------------------------------------------------------------------------#
def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model='nomic-embed-text'),  # Fixed
    )
    retriever = vectorstore.as_retriever()

    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {question} """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )
    result = chain.invoke("What is the use of Text Splitting?")
    print(result)
    
#--------------------------------------------------------------------------------#
#                            Splitting                                           #
#--------------------------------------------------------------------------------#
# 1. Character Text Splitting
print("#### Character Text Splitting ####")

text = "Text splitting in LangChain is a critical feature that facilitates the division of large texts into smaller, manageable segments. "

# Manual Splitting
chunks = []
chunk_size = 35 # Characters
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
print(documents)

# Call the rag function with the documents
rag(documents, "test_collection")

# 2 Automatic Text Slitting 

text_splitter = CharacterTextSplitter(chunk_size = 35, chunk_overlap=0, separator='', strip_whitespace=False)
documents = text_splitter.create_documents([text])
print(documents)