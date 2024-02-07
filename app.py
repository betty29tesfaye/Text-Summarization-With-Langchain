import warnings
import time
warnings.filterwarnings('ignore')

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_path = '/content/drive/MyDrive/Colab Notebooks/Comparing_apples_and_oranges_.pdf'
loaded_file = PyPDFLoader(file_path=file_path)

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000,
   chunk_overlap=100,
)
data_chunks = loaded_file.load_and_split(text_splitter=text_splitter)

#Defining the large language Model


llm = ChatOpenAI()


chain = load_summarize_chain(
   llm=llm,
   chain_type='refine'
)
chain.run(data_chunks)
