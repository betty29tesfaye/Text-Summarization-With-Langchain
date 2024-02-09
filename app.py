import warnings
import time
import os 
import base64

warnings.filterwarnings('ignore')

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from PIL import Image

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000,
   chunk_overlap=100,
)

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
def temporarly_save_uploaded_file(uploadedfile):
     with open(uploadedfile.name,"wb") as f:
         f.write(uploadedfile.getbuffer())
add_background_image('bgi.png')   

st.markdown(f'<span style="background-color:#DFF2FF;color:#0F52BA;font-family:book-antiqua;font-size:24px;">AI App For Text Summarizing</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(' ')

if uploaded_file is not None:
 
          
   temporarly_save_uploaded_file(uploaded_file)

   loaded_file = PyPDFLoader(str(uploaded_file.name))
   data_chunks = loaded_file.load_and_split(text_splitter=text_splitter)
   #data_chunks=text_splitter.split_text(uploaded_file)

#Defining the large language Model

   llm = ChatOpenAI()

   chain = load_summarize_chain(
        llm=llm,
        chain_type='refine'
        )
   print(chain.run(data_chunks))
