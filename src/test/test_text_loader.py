from langchain.document_loaders import TextLoader

loader = TextLoader("../../data/test/test.docx")
data = loader.load()
data