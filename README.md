# Multi-Modal-AI-System-For-Wild-Heap-Fire-Monitoring-and-Analysis

## 📦 Download BERT QA and CNN Models
The fine-tuned `bert_qa_final` folder (used for wildfire-related question answering) and CNN models for classification and segmentation are too large to host on GitHub.  
👉 [Download from Google Drive](https://drive.google.com/drive/folders/1iXNmxAAIJir4ABhPLnCYDC9b9gxWdtNm?usp=sharing)  

---

## 🌐 Overview  
This project is a **multi-modal AI system** designed to **monitor**, **classify**, **segment**, and **interact** with wildfire data, combining the power of deep learning, NLP, and interactive UI.

Built with 🔥passion and precision, this system blends **image classification**, **pixel-level segmentation**, and **natural language understanding** into a single streamlined platform.

---

## 🧠 Image Classification

Using the [🔥 Flame Dataset (IEEE Dataport)] for fire vs. no-fire detection:

- 📚 **Models Trained**:
  - **DenseNet121** (pretrained, then fine-tuned last 30 layers unfrozen)
  - **ResNet50** (same fine-tuning strategy)
  - **EfficientNet**
  - **MobileNet**
  - **Custom CNN**

- ✅ **Top Performer**:  
  `DenseNet121 (fine-tuned)` -- achieved **94% accuracy** and proved to be robust in distinguishing flames from clutter.

---

## 🧩 Segmentation

Using a **U-Net** architecture for **pixel-level segmentation** of fire regions:

- Input: Images + custom ground-truth **masks**
- Output: Pixel-wise fire detection maps
- Purpose: To go beyond image-level classification and **visually localize fire areas**.

---

## 💬 Wildfire Question Answering

- **Model**: Fine-tuned `bert-base-uncased` on a wildfire Q/A dataset
- **Use Case**: Ask technical or practical wildfire-related questions like:
  - *"How to prevent heap fires?"*
  - *"Is open burning allowed in dry weather?"*

- **Why BERT?** Because we don’t just classify — we converse.

---

## 🤖 Gemini Chatbot

A Gemini-powered chatbot handles **wildfire-related queries** intelligently:

- Uses Gemini for **friendly, real-time answers**
- Backs up where BERT may fall short
- Great for **casual interaction** and public education

---

## 🌟 Streamlit App (Interactive GUI)

Built using [Streamlit](https://streamlit.io/) with **3 awesome tabs**:

1. **Image Classification**  
   Upload an image → Get fire prediction → See model explainability (Grad-CAM coming soon!)

2. **Segmentation**  
   Upload an image + mask → Visual fire region mapping

3. **Wildfire Chat**  
   Ask BERT or Gemini anything about fire safety, prevention, or detection.

---

## 🚀 Tech Stack

- 🐍 Python
- 🧠 TensorFlow / Keras
- 🤗 HuggingFace Transformers
- 🎈 Streamlit
- 🧪 Numpy, OpenCV, Scikit-Learn
- 🧠 Gemini API

  ---

  ## ✨ Final Thoughts

This isn’t just another AI project — it’s a **smart firefighter assistant** 🔥  
Whether you’re detecting fire from a drone cam, segmenting it for localization, or just curious about safety, this system’s got you covered.

---

> Created with ❤️ by [Bilal Antall](https://github.com/BilalAntall)
