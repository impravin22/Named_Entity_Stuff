import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

# Define a BERT model class with EWC support
class BertForSequenceClassificationEWC(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.ewc_lambda = 5000  # Regularization term strength
        self.fisher_matrix = {}
        self.optimal_params = {}

    def compute_ewc_loss(self):
        """Compute the EWC loss."""
        ewc_loss = 0
        for name, param in self.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                optimal_param = self.optimal_params[name]
                ewc_loss += torch.sum(fisher * (param - optimal_param) ** 2)
        return self.ewc_lambda * ewc_loss

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=4)
model = BertForSequenceClassificationEWC(config)

# Load and preprocess the AG News dataset
dataset_ag_news = load_dataset("ag_news")

def tokenize_function(examples):
    # Updated to return attention masks
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt", return_attention_mask=True)

tokenized_datasets_ag_news = dataset_ag_news.map(tokenize_function, batched=True)

# Convert Hugging Face dataset to PyTorch DataLoader
class HF_Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset['input_ids']
        self.attention_masks = hf_dataset['attention_mask']  # Store attention masks
        self.labels = hf_dataset['label']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.hf_dataset[idx]),
            'attention_mask': torch.tensor(self.attention_masks[idx]),  # Include attention mask
            'labels': torch.tensor(self.labels[idx])
        }
        return item

train_dataset_ag_news = HF_Dataset(tokenized_datasets_ag_news['train'])
train_dataloader_ag_news = DataLoader(train_dataset_ag_news, batch_size=8)

# Function to calculate the Fisher Information Matrix and optimal parameters
def update_ewc_params(model, dataloader, device):
    model.eval()
    fisher_matrix = {}
    for name, param in model.named_parameters():
        fisher_matrix[name] = torch.zeros_like(param)

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        model.zero_grad()
        loss.backward()
        
        for name, param in model.named_parameters():
            fisher_matrix[name] += param.grad ** 2 / len(dataloader)

    # Update the fisher_matrix and optimal_params in the model
    model.fisher_matrix = {name: fisher.detach() for name, fisher in fisher_matrix.items()}
    model.optimal_params = {name: param.detach() for name, param in model.named_parameters()}

# Custom training loop to include EWC loss and print progress
def custom_train(model, train_dataloader, device, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss + model.compute_ewc_loss()  # Include EWC loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Update EWC parameters after the first task (AG News)
update_ewc_params(model, train_dataloader_ag_news, device)

# Load and preprocess the Emotion dataset (simulating the Twitter dataset)
dataset_emotion = load_dataset("emotion")
tokenized_datasets_emotion = dataset_emotion.map(tokenize_function, batched=True)
train_dataset_emotion = HF_Dataset(tokenized_datasets_emotion['train'])
train_dataloader_emotion = DataLoader(train_dataset_emotion, batch_size=8)

# Train on the new task (Emotion)
custom_train(model, train_dataloader_emotion, device, epochs=3)

# Directory for saving the model
model_dir = "./model"
# Create the directory if it does not exist
os.makedirs(model_dir, exist_ok=True)

# Save the model, tokenizer, and EWC parameters
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
torch.save({
    'fisher_matrix': model.fisher_matrix,
    'optimal_params': model.optimal_params
}, os.path.join(model_dir, "ewc_state.pt"))









# ------------- FP 16 mixed precision-----------------


# import torch
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
# from datasets import load_dataset
# import numpy as np
# from tqdm import tqdm
# import os

# # Define a BERT model class with EWC support
# class BertForSequenceClassificationEWC(BertForSequenceClassification):
#     def __init__(self, config):
#         super().__init__(config)
#         self.ewc_lambda = 5000  # Regularization term strength
#         self.fisher_matrix = {}
#         self.optimal_params = {}

#     def compute_ewc_loss(self):
#         """Compute the EWC loss."""
#         ewc_loss = 0
#         for name, param in self.named_parameters():
#             if name in self.fisher_matrix:
#                 fisher = self.fisher_matrix[name]
#                 optimal_param = self.optimal_params[name]
#                 ewc_loss += torch.sum(fisher * (param - optimal_param) ** 2)
#         return self.ewc_lambda * ewc_loss

# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# config = BertConfig.from_pretrained('bert-base-uncased', num_labels=4)
# model = BertForSequenceClassificationEWC(config)

# # Load and preprocess the AG News dataset
# dataset_ag_news = load_dataset("ag_news")

# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt", return_attention_mask=True)

# tokenized_datasets_ag_news = dataset_ag_news.map(tokenize_function, batched=True)

# # Convert Hugging Face dataset to PyTorch DataLoader
# class HF_Dataset(torch.utils.data.Dataset):
#     def __init__(self, hf_dataset):
#         self.hf_dataset = hf_dataset['input_ids']
#         self.attention_masks = hf_dataset['attention_mask']
#         self.labels = hf_dataset['label']

#     def __len__(self):
#         return len(self.hf_dataset)

#     def __getitem__(self, idx):
#         item = {
#             'input_ids': torch.tensor(self.hf_dataset[idx]),
#             'attention_mask': torch.tensor(self.attention_masks[idx]),
#             'labels': torch.tensor(self.labels[idx])
#         }
#         return item

# train_dataset_ag_news = HF_Dataset(tokenized_datasets_ag_news['train'])
# train_dataloader_ag_news = DataLoader(train_dataset_ag_news, batch_size=8)

# # Function to calculate the Fisher Information Matrix and optimal parameters
# def update_ewc_params(model, dataloader, device):
#     model.eval()
#     fisher_matrix = {}
#     for name, param in model.named_parameters():
#         fisher_matrix[name] = torch.zeros_like(param)

#     for batch in dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         model.zero_grad()
#         loss.backward()
        
#         for name, param in model.named_parameters():
#             fisher_matrix[name] += param.grad ** 2 / len(dataloader)

#     model.fisher_matrix = {name: fisher.detach() for name, fisher in fisher_matrix.items()}
#     model.optimal_params = {name: param.detach() for name, param in model.named_parameters()}

# # Custom training loop to include EWC loss and print progress
# def custom_train(model, train_dataloader, device, epochs=3):
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
#     scaler = torch.cuda.amp.GradScaler()  # NEW: Initialize GradScaler for mixed precision

#     model.train()
#     for epoch in range(epochs):
#         print(f"Epoch {epoch+1}/{epochs}")
#         progress_bar = tqdm(train_dataloader, desc="Training")
#         for batch in progress_bar:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             optimizer.zero_grad()

#             # Use autocast for the forward pass
#             with torch.cuda.amp.autocast():
#                 outputs = model(**batch)
#                 loss = outputs.loss + model.compute_ewc_loss()

#             # Scale loss and call backward
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             progress_bar.set_postfix(loss=loss.item())

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Update EWC parameters after the first task (AG News)
# update_ewc_params(model, train_dataloader_ag_news, device)

# # Load and preprocess the Emotion dataset (simulating the Twitter dataset)
# dataset_emotion = load_dataset("emotion")
# tokenized_datasets_emotion = dataset_emotion.map(tokenize_function, batched=True)
# train_dataset_emotion = HF_Dataset(tokenized_datasets_emotion['train'])
# train_dataloader_emotion = DataLoader(train_dataset_emotion, batch_size=8)

# # Train on the new task (Emotion)
# custom_train(model, train_dataloader_emotion, device, epochs=3)

# # Directory for saving the model
# model_dir = "./model"
# os.makedirs(model_dir, exist_ok=True)

# # Save the model, tokenizer, and EWC parameters
# model.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)
# torch.save({
#     'fisher_matrix': model.fisher_matrix,
#     'optimal_params': model.optimal_params
# }, os.path.join(model_dir, "ewc_state.pt"))

