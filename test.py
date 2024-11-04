import torch
#from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification, AdamW
from transformers import LayoutLMv2ImageProcessor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification
import torch
from PIL import Image
# from transformers import LayoutLMv3Processor
import pytesseract

feature_extractor = LayoutLMv2ImageProcessor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(feature_extractor, tokenizer)
label2idx = {".config": 0, "resume": 1, "scientific_publication": 2,"document-classification-dataset.zip":3,"email":4,"sample_data":5} # Replace with your actual label mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LayoutLMv2ForSequenceClassification.from_pretrained("saved_model")
model.to(device);
def predict(img):
    #query = 'C:/Users/Lenovo/Desktop/classifi/Screenshot.png'
    image = img.convert("RGB")
    #image=Image.open('C:/Users/Lenovo/Desktop/classifi/Screenshot.png').convert("RGB")
    encoded_inputs = processor(image, return_tensors="pt").to(device)
    outputs = model(**encoded_inputs)
    preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
    pred_labels = {label:pred for label, pred in zip(label2idx.keys(), preds)}
    highest_key_value_pair = max(pred_labels.items(), key=lambda item: item[1])
    return (highest_key_value_pair[0])

