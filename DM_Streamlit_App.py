import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch
import random
import requests
import google.generativeai as genai

qa_df = pd.read_csv("real_wildfire_qa_200.csv")
contexts = qa_df["context"].unique().tolist()
default_context = " ".join(contexts[2:4])
fallback_responses = {
    "heap": [
        "Heap fires are often hidden and dangerous. They can smolder underground for days, releasing toxic gases.",
        "Always monitor compost and coal piles—heat buildup can start heap fires without any spark.",
        "Heap fires are tricky to control. Avoid large heaps of combustible material in hot weather."
    ],
    "wildfire": [
        "Wildfires spread rapidly. Always maintain defensible space around your home and follow evacuation orders.",
        "Human negligence causes most wildfires. Be vigilant with campfires and discarded cigarettes.",
        "In wildfire-prone areas, keep emergency kits ready and plan evacuation routes in advance."
    ],
    "evacuation": [
        "Evacuation should be swift and calm. Always prioritize life over possessions.",
        "Know your evacuation routes before fire season. Practice drills with your family.",
        "Evacuate early when advised. Waiting too long can block escape routes due to fire or smoke."
    ]
}

@st.cache_resource

def load_model_and_tokenizer():
    bert_model = BertForQuestionAnswering.from_pretrained("bert_qa_final")
    tokenizer = BertTokenizerFast.from_pretrained("bert_qa_final")
    return bert_model, tokenizer

bert_model, tokenizer = load_model_and_tokenizer()
def preprocess_image(img):
    target_size = model.input_shape[1:3]
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_image_segment(img, target_size = (256,256)):
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_fire(model, img_array):
    preds = model.predict(img_array)
    return "✅ No Fire" if preds[0][0] > 0.5 else "🔥 Fire Detected"

def apply_gradcam(model, img_array):
    try:
        _ = model(img_array)  
    except:
        pass 
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
        elif isinstance(layer, tf.keras.Model):
            for inner in reversed(layer.layers):
                if isinstance(inner, tf.keras.layers.Conv2D):
                    last_conv_layer = inner.name
                    model = layer
                    break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")

    conv_output = model.get_layer(last_conv_layer).output
    grad_model = tf.keras.Model([model.input], [conv_output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  
    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[2]))
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cam


def plot_gradcam(original, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return superimposed

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img


MODEL_DIR = r'C:\Users\bilal\Downloads\DM_Fire_Project'
MODEL_MAP = {
    "DenseNet": "densenet_fire_classifier.keras",
    "DenseNet Finetuned": "densenet_finetuned.keras",
    "ResNet50": "resnet50_wildfire_model.keras",
    "ResNet50 Finetuned": "resnet50_finetuned.keras",
    "EfficientNet": "best_efficientnet_model.keras",
    "MobileNet": "best_mobilenet_model.h5",
    "Custom_CNN": "custom_cnn_fire_classifier.keras"
}

SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "unet_segmentation.keras")

st.title("🔥 Wildfire & Heap Fire Prediction App")
tabs = st.radio("Choose a tab:",["Image Classification", "Segmentation","Queries"])

if tabs == 'Image Classification':
    st.header("📷 Fire Classification")
    selected_model_name = st.selectbox("Choose a model:", list(MODEL_MAP.keys()))
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = load_image(uploaded_image)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        model_path = os.path.join(MODEL_DIR, MODEL_MAP[selected_model_name])
        model = tf.keras.models.load_model(model_path)

        img_preprocessed = preprocess_image(img)

        if st.button("Predict"):
            prediction = predict_fire(model, img_preprocessed)
            st.success(f"Prediction: {prediction}")

        if st.button("Apply Explainable AI (Grad-CAM)"):
            cam = apply_gradcam(model, img_preprocessed)
            cam_overlay = plot_gradcam(cv2.resize(img, (224, 224)), cam)
            st.image(cam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)

elif tabs == 'Segmentation':
    st.header("🔢 Fire Segmentation (U-Net)")
    uploaded_seg_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="seg_img")

    if uploaded_seg_img:
        seg_img = load_image(uploaded_seg_img)
        st.image(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

        seg_model = load_model(SEGMENTATION_MODEL_PATH)
        input_img = preprocess_image_segment(seg_img, target_size=(256, 256))
        pred_mask = seg_model.predict(input_img)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze()

        st.subheader("Predicted Mask")
        st.image(pred_mask * 255, caption="Predicted Mask", use_container_width=True, clamp=True)
        download_mask = (pred_mask * 255).astype(np.uint8)
        is_success, buffer = cv2.imencode(".png", download_mask)
        if is_success:
            st.download_button(
                label="📁 Download Mask",
                data=buffer.tobytes(),
                file_name="predicted_mask.png",
                mime="image/png"
            )

elif tabs == 'Queries':
    answer = ""
    st.header("🔥 Ask a Fire Safety Question")
    qa_model_choice = st.selectbox("Choose a model for answering:", ["BERT (offline)", "Gemini API (LLM)"])

    user_question = st.text_input("Enter your question about fire safety:")

    if st.button("Get Answer") and user_question.strip():
        inputs = tokenizer(user_question, default_context, return_tensors="pt")
        if qa_model_choice == "BERT (offline)":
            with torch.no_grad():
                outputs = bert_model(**inputs)
                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits)

                if start_idx >= end_idx:
                    answer = ""
                else:
                    answer = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx + 1])
                    )

            if not answer.strip():
                lowered_question = user_question.lower()
                matched = False
                for keyword in fallback_responses:
                    if keyword in lowered_question:
                        answer = random.choice(fallback_responses[keyword])
                        matched = True
                        break
                if not matched:
                    answer = "🤔 No answer found in context. Try rephrasing your question."
        elif qa_model_choice == 'Gemini API (LLM)':
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel("gemini-1.5-flash")
                chat = model.start_chat(history=[])
                response = chat.send_message("You are a wildfire and heap fire safety expert. Only respond to questions strictly related to fire safety. If the question is not about wildfire or heap fire, reply: I can only answer wildfire and heap fire questions. Always reply in **exactly five short one-line bullet points**. Do not write paragraphs. Keep it brief and professional.")
                print(response.text)
                response = chat.send_message(user_question)
                answer = response.text
            except Exception as e:
                answer = f"Error fetching response from Gemini: {str(e)}"

    st.markdown(f"**Q:** {user_question}")
    st.markdown(f"**A:** {answer}")