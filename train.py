from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification, AdamW
from transformers import LayoutLMv2ImageProcessor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification
import torch
from PIL import Image
# from transformers import LayoutLMv3Processor
import pytesseract
import pandas as pd
import os
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from tqdm.auto import tqdm
import streamlit as st


feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(feature_extractor, tokenizer)

label2idx={'.config': 0,
 'resume': 1,
 'scientific_publication': 2,
 'document-classification-dataset.zip': 3,
 'email': 4,
 'sample_data': 5}

training_features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
})    



def train_mod():

    train_data=get_data()
    if len(train_data)<20:
        st.write("Insufficient Data")
        return
    train_dataloader = training_dataloader_from_df(train_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",  num_labels=len(label2idx))
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3


    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        training_loss = 0.0
        training_correct = 0
        #put the model in training mode
        model.train()
        for batch in tqdm(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            training_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            training_correct += (predictions == batch['labels']).float().sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Training Loss:", training_loss / batch["input_ids"].shape[0])
        training_accuracy = 100 * training_correct / len(train_data)
        print("Training accuracy:", training_accuracy.item())

        validation_loss = 0.0
        validation_correct = 0
        for batch in tqdm(valid_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            validation_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            validation_correct += (predictions == batch['labels']).float().sum()

        print("Validation Loss:", validation_loss / batch["input_ids"].shape[0])
        validation_accuracy = 100 * validation_correct / len(valid_data)
        print("Validation accuracy:", validation_accuracy.item())
    #udating model weights
    model.save_pretrained('saved_model1/')    

def get_data():
    # Get a list of image files in the directory
    df=pd.DataFrame()
    for i in label2idx:
  #image_files = [f for f in os.listdir('/content/'+i) if os.path.isfile(os.path.join('/content/'+i, f))]
        df=pd.concat([df,make_image_dataframe('C:/Users/Lenovo/Desktop/classifi/'+i,i)])    
    return df

def make_image_dataframe(folder_path,label):
  image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp') # Add more if needed
  image_paths = []
  for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        image_paths.append(os.path.join(folder_path, filename))
  df = pd.DataFrame({'image_path': image_paths,'label':label})
  return df




def encode_training_example(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    encoded_inputs = processor(images, padding="max_length", truncation=True)
    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]

    return encoded_inputs


def training_dataloader_from_df(data):
    dataset = Dataset.from_pandas(data)

    encoded_dataset = dataset.map(
        encode_training_example, remove_columns=dataset.column_names, features=training_features,
        batched=True, batch_size=2
    )
    encoded_dataset.set_format(type='torch', device=device)
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    return dataloader
