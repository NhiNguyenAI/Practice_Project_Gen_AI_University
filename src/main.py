"""
************************************************************************
 *
 * main.py - Modified for Google Colab
 *
 * Modified by: [Your Name]
 * Modified on: 2025-27-03
 *
 ************************************************************************
"""

#--------------------------------------------------------------------------------#
#                                 Libraries                                      #
#--------------------------------------------------------------------------------#
import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama

#--------------------------------------------------------------------------------#
#                                 Load the urls                                  #
#--------------------------------------------------------------------------------#
urls_list = ["https://www.hs-fulda.de/en/", 
             "https://www.hs-fulda.de/en/studyprogrammes"]

# Load Data
loader = UnstructuredURLLoader(urls=urls_list)
data = loader.load()
len(data)

#--------------------------------------------------------------------------------#
#                                 splitting                                      #
#--------------------------------------------------------------------------------#
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
docs = text_splitter.split_documents(data)
docs[0]

