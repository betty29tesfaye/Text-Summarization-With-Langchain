import warnings
import time
warnings.filterwarnings('ignore')

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from PIL import Image

file_path = '/content/drive/MyDrive/Colab Notebooks/Comparing_apples_and_oranges_.pdf'
loaded_file = PyPDFLoader(file_path=file_path)

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

def temporarly_save_uploaded_file(uploadedfile):
     with open(uploadedfile.name,"wb") as f:
         f.write(uploadedfile.getbuffer())

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000,
   chunk_overlap=100,
)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def add_background_image(image_file):
  with open(image_file, "rb") as image_file:
     encoded_string = base64.b64encode(image_file.read())
  st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_background_image('bgi.png')   
st.markdown(f'<span style="background-color:#DFF2FF;color:#0F52BA;font-family:book-antiqua;font-size:24px;">AI App For Text Summarizing</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(' ')

data_chunks = loaded_file.load_and_split(text_splitter=text_splitter)

#Defining the large language Model


llm = ChatOpenAI()


chain = load_summarize_chain(
   llm=llm,
   chain_type='refine'
)
chain.run(data_chunks)
