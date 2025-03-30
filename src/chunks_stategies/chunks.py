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
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub
from agentic_chunker import AgenticChunker
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.text_splitter import PythonCodeTextSplitter


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
# 1.1 Character Text Splitting
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

# 1.2 Automatic Text Slitting 

text_splitter = CharacterTextSplitter(chunk_size = 35, chunk_overlap=0, separator='', strip_whitespace=False)
documents = text_splitter.create_documents([text])
print(documents)

#--------------------------------------------------------------------------------#
#                            Recursive Character Text Splitting                  #
#--------------------------------------------------------------------------------#



with open('../../data/chunks/content.txt', 'r', encoding='utf-8') as file:
    text = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 65, chunk_overlap=0) # ["\n\n", "\n", " ", ""] 65,450
print(text_splitter.create_documents([text])) 

#--------------------------------------------------------------------------------#
#                            Document Specific Splitting                         #
#--------------------------------------------------------------------------------#

# Document Specific Splitting - Markdown

splitter = MarkdownTextSplitter(chunk_size = 40, chunk_overlap=0)
markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""
print(splitter.create_documents([markdown_text]))

# Document Specific Splitting - Python

python_text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

for i in range(10):
    print (i)
"""
python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)
print(python_splitter.create_documents([python_text]))

# Document Specific Splitting - Javascript

javascript_text = """
// Function is called, the return value will end up in x
let x = myFunction(4, 3);

function myFunction(a, b) {
// Function returns the product of a and b
  return a * b;
}
"""
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=65, chunk_overlap=0
)
print(js_splitter.create_documents([javascript_text]))

#--------------------------------------------------------------------------------#
#                            Use embeddings for Semantic Chunking                #
#--------------------------------------------------------------------------------#

# Percentile - all differences between sentences are calculated, and then any difference greater than the X percentile is split
text_splitter = SemanticChunker(OpenAIEmbeddings())
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile"
)
documents = text_splitter.create_documents([text])
print(documents)

#--------------------------------------------------------------------------------#
#                            Agentic Chunking ->  For LLM                        #
#--------------------------------------------------------------------------------#
obj = hub.pull("wfh/proposal-indexing")
llm = ChatOpenAI(model='gpt-3.5-turbo')
runnable = obj | llm

class Sentences(BaseModel):
    sentences: List[str]
    
# Extraction
extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
def get_propositions(text):
    runnable_output = runnable.invoke({
    	"input": text
    }).content
    propositions = extraction_chain.invoke(runnable_output)["text"][0].sentences
    return propositions
    
paragraphs = text.split("\n\n")
text_propositions = []
for i, para in enumerate(paragraphs[:5]):
    propositions = get_propositions(para)
    text_propositions.extend(propositions)
    print (f"Done with {i}")

print (f"You have {len(text_propositions)} propositions")
print(text_propositions[:10])

print("#### Agentic Chunking ####")
ac = AgenticChunker()
ac.add_propositions(text_propositions)
print(ac.pretty_print_chunks())

#--------------------------------------------------------------------------------#
#                            Agentic Chunking for Optimised Grouping             #
#--------------------------------------------------------------------------------#
ac = AgenticChunker()
ac.add_propositions(text_propositions)
print(ac.pretty_print_chunks())
chunks = ac.get_chunks(get_type = 'List of strings')
print(chunks)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
rag(documents, "agentic-chunks-optimised")

