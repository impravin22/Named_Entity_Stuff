# Assignment for WondersAI

# Tackling Catastrophic Forgetting:

**Catastrophic Forgetting**

Catastrophic forgetting refers to a phenomenon observed in machine learning models, particularly in neural networks, where the model loses previously learned knowledge or experiences significant degradation in performance on previously learned tasks when trained on new tasks or datasets.

[Catastrophic Forgetting in Neural Networks](https://www.theaidream.com/post/catastrophic-forgetting-in-neural-networks#:~:text=Catastrophic%20forgetting%20is%20observed%20when,during%20the%20initial%20training%20phase.)

**Elastic weight Consolidation**

**EWC** is a technique introduced by Kirkpatrick et al. in the paper "Overcoming catastrophic forgetting in neural networks". It addresses the problem of catastrophic forgetting by adding a regularization term to the loss function. This term penalizes changes to parameters that are important for previous tasks, based on the Fisher Information Matrix.

[Overcoming catastrophic forgetting in neural networks | Proceedings of the National Academy of Sciences](https://www.pnas.org/doi/10.1073/pnas.1611835114)

### Fisher Information Matrix

- The **Fisher Information Matrix** quantifies the importance of each parameter of the model concerning the tasks it has already learned. In the context of EWC, it is used to determine how much each parameter should contribute to the regularization term. The Fisher Information is calculated as the square of the gradients of the log-likelihood with respect to the parameters, averaged over the data distribution.

### PyTorch and Transformers

- **PyTorch** is an open-source machine learning library used for applications such as computer vision and natural language processing. It provides the core functions for defining and training neural networks.
- **Transformers** library, developed by Hugging Face, provides thousands of pre-trained models to perform tasks on texts such as classification, information extraction, question answering, and more. It also provides easy-to-use interfaces to work with these models.

### Key Functions and Methods

- `BertTokenizer` and `BertForSequenceClassification`: These are classes from the Transformers library. The tokenizer is used to convert text into a format that the BERT model can understand (tokenization), and `BertForSequenceClassification` is a BERT model adapted for text classification tasks.
- `load_dataset`: A function from the `datasets` library (also by Hugging Face) used to load and preprocess datasets.
- `DataLoader` and `Dataset`: PyTorch classes for handling batches of data, making it easier to iterate over data during training.
- `torch.optim.Adam`: An optimizer from PyTorch, used for updating model parameters based on gradients.
- `torch.cuda.amp`: Automatic Mixed Precision (AMP) package from PyTorch, used for faster training with reduced precision arithmetic.

### Custom Classes and Functions

- `BertForSequenceClassificationEWC`: A custom class inheriting from `BertForSequenceClassification`, extended to include EWC support by adding methods to compute the EWC loss and update the Fisher Information Matrix and optimal parameters.
- `HF_Dataset`: A custom PyTorch `Dataset` class to handle data from the Hugging Face `datasets` library.
- `update_ewc_params`: A function to update the Fisher Information Matrix and optimal parameters after training on a task.
- `custom_train`: A custom training loop that includes the computation of the EWC loss along with the standard loss.

**Code**

```other
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
```

(I tried to run this on CoLab but could not run it as the GPU resources are limited, I tried using FP16 precision as well. But it was training for more than 8 hours, So I could not save the model.)

⟶ The above code helps to solve the catastrophic forgetting problem during continuous pre-training for domain-specific knowledge. If you observe the code, it is intended to pre-train on the news dataset, and emotion dataset, which are of two different domains.

## **Some other potential solutions**

Did not get the time to implement these. I just did a fair bit of research on the other solutions.

### Progressive Neural Networks (PNNs)

- **PNNs** address catastrophic forgetting by maintaining separate neural networks, called "experts," for each task encountered. When a new task is introduced, a new expert network is added to the architecture, initialized with weights from the existing experts. The parameters of the new expert network are updated during training on the new task, while the parameters of the existing experts are kept frozen. This allows the network to specialize in the new task without disrupting performance on previously learned tasks.

### Optimized Fixed Expansion Layers (OFELs)

- **OFELs** introduce additional layers into the neural network architecture that are specifically designed to preserve knowledge learned from previous tasks while allowing the network to adapt to new tasks. These additional layers, known as "expansion layers," are fixed in size and are not updated during the training process, serving as memory banks. Alongside these, OFELs include trainable adaptation layers responsible for adapting the network's parameters to the current tasks.

### Meta-learning Approaches

- **Meta-learning** involves training models to learn how to learn, enabling them to quickly adapt to new tasks without significant degradation in performance on previously learned tasks. **Model-agnostic meta-learning (MAML)** and **Reptile** are two popular meta-learning algorithms. MAML aims to learn a good initialization of model parameters for rapid convergence on new tasks, while Reptile updates the model parameters towards the direction of the average gradient across multiple tasks, encouraging the model to find a robust set of parameters that generalize well across tasks.

### Transfer Learning

- **Transfer Learning** involves fine-tuning a model trained on one task to perform a related task. This method leverages the knowledge the model has already acquired, such as features useful for recognizing images of animals in general, to quickly learn to recognize new but related categories.

[Catastrophic Forgetting in Machine Learning](https://codelabsacademy.com/blog/catastrophic-forgetting-in-machine-learning)

### Ensemble Methods

- **Ensemble Methods** train multiple models to solve different tasks, and their outputs are combined to make a final prediction. This approach helps to prevent catastrophic forgetting by leveraging the specialized knowledge of each model in the ensemble to make more informed predictions on a variety of tasks.

# Expanding context length to above 128k

BERT and similar transformer models have a limitation on the maximum sequence length due to their self-attention mechanism, which has a quadratic complexity concerning the sequence length. The original BERT model has a maximum sequence length of 512 tokens.

### Strategies for Handling Longer Texts with BERT-like Models

1. **Chunking and Aggregating**: Split the long text into smaller chunks that fit within the model's maximum input size, process each chunk independently, and then aggregate the results.
2. **Extended Position Embeddings**: This involves initializing additional position embeddings and training them, allowing the model to process longer sequences than it was originally designed for. **Limitation:** However, this requires additional training and might not scale to very long contexts like 128K tokens without significant computational resources.

[How to use Bert for long text classification?](https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification)

**Using different models**

**LLAMA-2:**

LLAMA-2-7B can only handle context length above 32K.
LLAMA-2-13B can handle context length above 128K.

[NousResearch/Yarn-Llama-2-13b-128k · Hugging Face](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-128k)

[Llama-2 with 128k context length thanks to YaRN](https://www.reddit.com/r/LocalLLaMA/comments/166je92/llama2_with_128k_context_length_thanks_to_yarn/)

**code:**

The above code can be adjusted as follows

```other
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-13b-128k")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-13b-128k")

def tokenize_function(examples):
    # Adjusted for longer context; consider dynamic adjustment based on actual text length and memory constraints
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128000, return_tensors="pt", return_attention_mask=True)
```

**Methods to improve context length to 128K:**

[Extending Context Length in Large Language Models](https://towardsdatascience.com/extending-context-length-in-large-language-models-74e59201b51f)

Position Interpolation: This method extends the context window of LLMs by interpolating the positional embeddings. It allows the model to handle longer sequences without increasing the number of parameters.

[Extending the Context length of Language Models: A Deep Dive into Positional Interpolation](https://medium.com/@jain.sm/extending-the-context-length-of-language-models-a-deep-dive-into-positional-interpolation-a93140c69f6a)

NTK-Aware and Dynamic NTK: These methods leverage the Neural Tangent Kernel (NTK) to efficiently extend the context length. The NTK-Aware method optimizes pre-training by considering the NTK, while the Dynamic NTK method adjusts the NTK during fine-tuning to handle longer sequences.

*As our main goal is to increase the context legth during pre-training, we should consider NTK-aware Training.*

[Aman&#x27;s AI Journal • NLP • LLM Context Length Extension](https://aman.ai/primers/ai/context-length-extension/)

# Training a NER model

**Code:**

```other
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
import spacy.matcher
from sklearn.model_selection import KFold
import random

# Step 1: Data Collection
def collect_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text_data = " ".join([p.get_text() for p in soup.find_all("p")])
    return text_data

# Step 2: Data Preprocessing
def preprocess_data(text_data):
    nlp = spacy.blank("en")
    doc = nlp(text_data)
    preprocessed_data = " ".join([token.text.lower() for token in doc if not token.is_punct])
    return preprocessed_data

def annotate_data(nlp, text_data, named_entities):
    matcher = spacy.matcher.PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(entity) for entity in named_entities]
    matcher.add("TerminologyList", patterns)
    
    doc = nlp(text_data)
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    filtered_spans = spacy.util.filter_spans(spans)
    entities = [(span.start_char, span.end_char, "TERMINOLOGY") for span in filtered_spans]
    
    return Example.from_dict(doc, {"entities": entities})

# Model Training
def train_model(annotated_data, iterations):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    ner.add_label("TERMINOLOGY")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            losses = {}
            examples = [example for example in annotated_data]
            nlp.update(examples, sgd=optimizer, drop=0.5, losses=losses)
            print(f"Iteration {itn + 1}: Loss = {losses['ner']:.4f}")
    return nlp

# Model Evaluation
def evaluate_model(nlp, test_data):
    scores = nlp.evaluate(test_data)
    return scores

def tag_raw_data(model, raw_data):
    doc = model(raw_data)
    tagged_data = [(ent.text, ent.label_) for ent in doc.ents]
    return tagged_data

def main():
    nlp = spacy.blank("en")
    training_urls = [
        "https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/toolkit",
        "https://en.wikipedia.org/wiki/TensorFlow",
        "https://www.tensorflow.org/learn",
        "https://en.wikipedia.org/wiki/Google_Brain",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
            # Add more URLs with relevant content
    ]
    named_entities = ["TensorFlow", "Google", "Python", "Machine Learning"]

    annotated_data = []
    for url in training_urls:
        text_data = collect_data(url)
        preprocessed_data = preprocess_data(text_data)
        annotated_data.append(annotate_data(nlp, preprocessed_data, named_entities))

    # Shuffle and split the data for cross-validation

    # random.shuffle(annotated_data)
    # Shuffle and prepare for cross-validation
    random.shuffle(annotated_data)
    kf = KFold(n_splits=5)
    
    best_f1_score = -1
    best_model_path = ""
    
    for fold, (train_index, test_index) in enumerate(kf.split(annotated_data)):
        train_data = [annotated_data[i] for i in train_index]
        test_data = [annotated_data[i] for i in test_index]
        
        # Train model on this fold's training data
        model = train_model(train_data, iterations=100)
        
        # Evaluate model on this fold's test data
        evaluation_results = evaluate_model(model, test_data)
        print(f"Fold {fold+1} Evaluation Results: {evaluation_results}")
        
        # Check if this model is the best so far
        if evaluation_results['ents_f'] > best_f1_score:
            best_f1_score = evaluation_results['ents_f']
            best_model_path = f"best_model_fold_{fold+1}"
            model.to_disk(best_model_path)
    
    print(f"Best F1 Score: {best_f1_score}")
    
    # Load the best model from disk
    best_model = spacy.load(best_model_path)
    
    # Model Deployment and Tagging with the best model
    sample_text = "TensorFlow, developed by Google, is widely used in Python programming for machine learning projects, often requiring GPU acceleration."
    tagged_data = tag_raw_data(best_model, sample_text)
    print("Tagged Data:", tagged_data)

    # kf = KFold(n_splits=5)  # 5-fold cross-validation
    # for train_index, test_index in kf.split(annotated_data):
    #     train_data = [annotated_data[i] for i in train_index]
    #     test_data = [annotated_data[i] for i in test_index]
    #     model = train_model(train_data, iterations=30)
    #     evaluation_results = evaluate_model(model, test_data)
    #     print(f"Evaluation Results: {evaluation_results}")

    # # After cross-validation, train a final model on all data
    # final_model = train_model(annotated_data, iterations=200)

    # # Model Deployment and Tagging
    # sample_text = "TensorFlow, developed by Google, is widely used in Python programming for machine learning projects, often requiring GPU acceleration."
    # tagged_data = tag_raw_data(final_model, sample_text)
    # print("Tagged Data:", tagged_data)

if __name__ == "__main__":
    main()
```

for results, check my GitHub repo:

[Github](https://github.com/impravin22/Named_Entity_Stuff/blob/main/NER.ipynb)

### Overview

I developed this script to tackle a Named Entity Recognition (NER) task using the `spaCy` library, a powerful tool for natural language processing (NLP). The goal is to identify and then classify named entities (like "TensorFlow", "Google", "Python", and "Machine Learning".) within text data collected from various web sources.

Overall, this proposed methodology can

### a. Identify Domain-Specific Terminology

By manually specifying a list of named entities such as "TensorFlow", "Google", "Python", and "Machine Learning", I focused on a specific domain (in this case, technology and programming languages). The use of `spaCy`'s `PhraseMatcher` in the `annotate_data` function allows for the efficient identification of these terms within the preprocessed text data.

### b. Used in Tagging Raw Data Sourced from the Internet or Clients

The `collect_data` function shows how to fetch and process text data from web pages, and tells the model's ability to work with internet-sourced data. And, the `tag_raw_data` function explains how the trained model can be applied to new, raw text data to identify and label named entities. This is very important for practical applications as it allows the model to process and extract valuable information from unstructured data, doesn't matter if it is collected from the internet or the clients.

# What does NER model has to do to with pretraining?

[Named Entity Recognition: The Mechanism, Methods, Use Cases,](https://www.altexsoft.com/blog/named-entity-recognition/)

[A Beginner&#x27;s Introduction to NER (Named Entity Recognition)](https://www.analyticsvidhya.com/blog/2021/11/a-beginners-introduction-to-ner-named-entity-recognition/)

The significance of NER models in the context of pre-training lies in their ability to enhance the performance and efficiency of named entity recognition tasks. Here are some key points:

1. Transfer learning: Pre-trained models like GPT-4 or RoBERTa can be adapted for specific NER tasks, saving computational effort and often leading to better performance compared to training from scratch.
2. Automatic feature learning: Deep learning-based NER models can learn features automatically from the data, reducing the reliance on extensive manual feature engineering required in traditional methods. This capability makes deep learning models more efficient and effective.
3. Handling large datasets: Deep learning models, including transformer architectures like GPT, can handle vast datasets and complex structures, often outperforming traditional approaches when there is a wealth of training data available.
4. Capturing sequential information: Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, which are commonly used in NER, can capture sequential information, making them suitable for processing textual data with context.
5. Non-linear representation learning: Deep learning approaches map input data to non-linear representations, enabling the learning of complex relations present in the input data. This capability helps in building state-of-the-art NER systems.
6. Reduced feature engineering: By using deep learning techniques, a significant amount of time and resources spent on feature engineering, which is required for traditional approaches, can be avoided.

In summary, pre-trained NER models offer significant advantages in terms of performance, efficiency, and the ability to handle complex data, making them a valuable tool in the field of named entity recognition.

