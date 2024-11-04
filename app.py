import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import test
import torch
#from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
#from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification, AdamW

from transformers import AutoModel
import loader
import train

st.set_page_config(
    page_title="Doc Classifier App", page_icon="📊", initial_sidebar_state="expanded"
)


def predict(img):
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
#    model = LayoutLMv2ForSequenceClassification.from_pretrained("saved_model")
    model.to(device);
    query = '../content/email/doc_000042.png'
    image = Image.open(query).convert("RGB")
    encoded_inputs = processor(image, return_tensors="pt").to(device)
    outputs = model(**encoded_inputs)
    preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
    pred_labels = {label:pred for label, pred in zip(label2idx.keys(), preds)}
    pred_labels
    return pred_labels

st.image("C:/Users/Lenovo/Desktop/classifi/download.jpg", width=600)

st.write(
    """
# Doc Classifier App
Upload your document to see classification results.
"""
)

uploaded_file = st.file_uploader("Upload image", type=".png")

ab_default = None
result_default = None



if uploaded_file:
    img=Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    text=(str)(test.predict(img))
    st.markdown("### Document is "+text)
   #highest_key_value_pair st.write(text)
    

    st.markdown("### Is prediction correct?")
    v1=st.radio("",['YES','NO'],index=1)
    title = st.text_input("Enter the correct Doc Type", "Enter Type")
    st.write("The correct Doc Type is", title)
    num=0
    loader.df_load(img,text,num)
    num=num+1
    
    if st.button("Train Model"):
        st.write("training model")
        train.train_mod()
