# ***LayoutLMv3-Based Document Classification and LLaMA for Summarization**  
#### This project integrates LayoutLMv3 for document classification and LLaMA for text summarization into a web application. It focuses on utilizing advanced multi-modal features and efficient text extraction methods to deliver accurate and robust document processing capabilities

## **why LayoutLMv3?**
### We chose LayoutLMv3 because of its:
#### **Unified Multi-modal Transformer:** Combines textual, visual, and layout information effectively.
#### **Vision Transformer (ViT) Backbone:** Enables fine-grained visual feature extraction.
![Screenshot 2024-12-25 233751](https://github.com/user-attachments/assets/88dfef03-8edf-4b4c-af88-69f48dbb30bc)

## **Dataset**
####   Custom Dataset: We collected and annotated our own dataset with multiple document classes tailored for classification tasks

## **Process Overview**
####   Initially, the dataset files are converted into image format to align with the requirements of LayoutLMv3

## **Text Extraction**
####   Although LayoutLMv3 includes built-in **OCR** (TesseractOCR), we opted for EasyOCR due to its superior speed and efficiency.
####   The extracted text is stored in JSON format for further processing.

## Model Setup
### **feature Extraction and Tokenization**
#### We used the following setup to prepare the model:
![output](https://github.com/user-attachments/assets/b193abac-6bfe-412b-aadd-428ecc0b2aaa)

#### 
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)


## **Weights:**
####  Base Model: 133M parameters
####  Large Model: 368M parameters

## **Data Structuring**
### After loading the weights, we structured the dataset and split it into training and testing sets.

## **Training**
### We implemented a training pipeline with PyTorch Lightning:

###
model_checkpoint = ModelCheckpoint(
    filename="{epoch}-{step}-{val_loss:.4f}", save_last=True, save_top_k=3, monitor="val_loss", mode="min"
)

trainer = pl.Trainer(
    accelerator="gpu",
    precision=16,
    devices=1,
    max_epochs=5,
    callbacks=[model_checkpoint],
)
trainer.fit(model_module, train_data_loader, test_data_loader)
![image](https://github.com/user-attachments/assets/dcfc4f26-8c04-4ca6-a3b4-68d5cf813ba6)


### **Parameter Tuning:** Adjust epochs and other parameters based on your computational resources.
## output Example
### 
c:\Users\tharshananth N\.conda\envs\fast_gpu\lib\site-packages\lightning_fabric\connector.py:572: `precision=16` is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs

## **Architecture**
### The architecture integrates multi-modal processing, leveraging LayoutLMv3's unified Transformer for effective document classification.
![Screenshot 2024-12-26 090201](https://github.com/user-attachments/assets/c1c3f69b-d153-41e9-b5c1-ad6c526e6ddf)

## **Limitations**
###
Hardware Constraints:
The system used for this project (i5-12500H, 8GB RAM, RTX 3050 4GB) took approximately 15 minutes for text extraction using EasyOCR.
Training was constrained due to limited computational resources.
Next Steps:
Exploring alternative setups to complete the training phase efficiently.

## **References**
### LayoutLMv3 **Documentation:** 
#### https://github.com/microsoft/unilm/tree/master/layoutlmv3


