from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader

local_path = "updated data.pdf"

if local_path:
  loader = PyPDFLoader("data.pdf")
  data = loader.load()
else:
  print("please upload a file")

#print( data[0].page_content)

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

vector_db = Chroma.from_documents (
  documents=chunks,
  embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
  collection_name="local-rag"
)


from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_model = "mistral"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
  input_variables=["question"],
  template="""You are an AI language model assistant.  Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector
  database. Don't try to exactly match the contents or keywords. By generating multiple perspectives  on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.  
  Provide these alternative questions seperated by newlines.
  Original question: {question}""", 
)
retriever = MultiQueryRetriever.from_llm(
  vector_db.as_retriever(),
  llm,
  prompt=QUERY_PROMPT
)

template = """Answer the question directly and avoid using phrases like 'from the given context' or 'based on the context' or 'from the information provided'. Provide a clear, concise, and informative response.:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
  {"context": retriever, "question": RunnablePassthrough()}
   | prompt
   | llm
   | StrOutputParser()
)

#output = chain.invoke(input())
#print(output)

def reply (question):
  try:
    output = chain.invoke(question)
  except Exception as e:
    #output = e
    output = "I am food safety bot where I can answer questions only related to food. If your question is related to food sure will try to update myself."
  finally:
    return output
