# =====================================================
# STREAMLIT APP — AI CANCER DIAGNOSIS SYSTEM + XAI
# =====================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import densenet121
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown

from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="AI Cancer Diagnosis", layout="centered")

st.title("🧬 AI Cancer Diagnosis System")
st.write("Upload a histopathology/MRI image for AI-based cancer prediction.")

# =====================================================
# DEVICE (FOR STREAMLIT CLOUD -> CPU ONLY)
# =====================================================
device = torch.device("cpu")

# =====================================================
# CREATE MODEL DIRECTORY
# =====================================================
os.makedirs("saved_model", exist_ok=True)

# =====================================================
# GOOGLE DRIVE MODEL LINKS
# =====================================================
MODEL_URLS = {
    "Any": "https://drive.google.com/uc?id=1VrYCI3MKJ2C3T-Zot-XwhGEZIYMYNEy6",
    "Acute Lymphoblastic Leukemia": "https://drive.google.com/uc?id=1hfpae28q2dX5hT8I9es-v__f5ReiTsIi",
    "Brain": "https://drive.google.com/uc?id=1Q6ZFkEEzDwyEYGNGmuJZnpgKLeoBUwkG",
    "Breast": "https://drive.google.com/uc?id=18D42Zjq-UGwXxk5RkncjDGfsfls0O5nF",
    "Cervical": "https://drive.google.com/uc?id=163nIIzV5xMqgLHgjiR0DgSupN_sDgCek",
    "Kidney": "https://drive.google.com/uc?id=1lbbmEAmwuCdantM40cfyg2vJOMuFSkvX",
    "Lung": "https://drive.google.com/uc?id=1cUSlV-KToG5s-qOC6LBaxp0mc1Oc06pa",
    "Colon": "https://drive.google.com/uc?id=1vSfbasUNufThd7oy0sAbYQGs95A2mt1H",
    "Lymphoma": "https://drive.google.com/uc?id=1R4aqGH0YsEaza-6k92_CFF949LoLT1P_",
    "Oral": "https://drive.google.com/uc?id=15DazGaJAOIdp143VF3a8ZYBm0jgJS9_L",
}

# =====================================================
# MODEL PATH MAP (LOCAL)
# =====================================================
MODEL_MAP = {
    k: f"saved_model/{k.replace(' ', '_').lower()}.pth"
    for k in MODEL_URLS.keys()
}

# =====================================================
# DOWNLOAD MODEL IF NOT EXISTS
# =====================================================
def download_model(model_key):
    model_path = MODEL_MAP[model_key]

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10_000_000:
        with st.spinner(f"⬇️ Downloading {model_key} model..."):
            gdown.download(MODEL_URLS[model_key], model_path, quiet=False)

    return model_path

# =====================================================
# CLASS INFO
# =====================================================
CLASS_INFO = {
    "all_benign": ("Benign", "Non-cancerous, healthy cells"),
    "all_early": ("Early", "Early stages of leukemia"),
    "all_pre": ("Pre", "Pre-stage abnormal cells"),
    "all_pro": ("Pro", "Advanced leukemia cells"),

    "brain_glioma": ("Glioma", "Most common brain tumor"),
    "brain_menin": ("Meningioma", "Tumors affecting brain membranes"),
    "brain_tumor": ("Pituitary Tumor", "Tumors affecting the pituitary gland"),

    "breast_benign": ("Benign", "Non-cancerous breast tissues"),
    "breast_malignant": ("Malignant", "Cancerous breast tissues"),

    "cervix_dyk": ("Dyskeratotic", "Abnormal cell growth"),
    "cervix_koc": ("Koilocytotic", "HPV infection related cells"),
    "cervix_mep": ("Metaplastic", "Precancerous cell changes"),
    "cervix_pab": ("Parabasal", "Immature squamous cells"),
    "cervix_sfi": ("Superficial-Intermediate", "Mature squamous cells"),

    "kidney_normal": ("Normal", "Healthy kidney tissues"),
    "kidney_tumor": ("Tumor", "Tumor-affected kidney tissues"),

    "colon_aca": ("Colon Adenocarcinoma", "Cancerous colon cells"),
    "colon_bnt": ("Colon Benign Tissue", "Healthy colon tissues"),

    "lung_aca": ("Lung Adenocarcinoma", "Cancerous lung cells"),
    "lung_bnt": ("Lung Benign Tissue", "Healthy lung tissues"),
    "lung_scc": ("Lung Squamous Cell Carcinoma", "Aggressive lung cancer"),

    "lymph_cll": ("CLL", "Slow blood cancer"),
    "lymph_fl": ("Follicular Lymphoma", "Slow-growing lymphoma"),
    "lymph_mcl": ("Mantle Cell Lymphoma", "Aggressive lymphoma"),

    "oral_normal": ("Normal", "Healthy oral tissues"),
    "oral_scc": ("Oral SCC", "Cancerous oral cells"),
}

# =====================================================
# ORGAN CONFIG
# =====================================================
ORGAN_CONFIG = {
    "Any": {"classes": list(CLASS_INFO.keys()), "topk": 3},
    "Brain": {"classes": ["brain_glioma","brain_menin","brain_tumor"], "topk": 3},
    "Breast": {"classes": ["breast_benign","breast_malignant"], "topk": 2},
    "Kidney": {"classes": ["kidney_normal","kidney_tumor"], "topk": 2},
    "Colon": {"classes": ["colon_aca","colon_bnt"], "topk": 2},
    "Lung": {"classes": ["lung_aca","lung_bnt","lung_scc"], "topk": 3},
    "Lymphoma": {"classes": ["lymph_cll","lymph_fl","lymph_mcl"], "topk": 3},
    "Oral": {"classes": ["oral_normal","oral_scc"], "topk": 2},
    "Acute Lymphoblastic Leukemia":
        {"classes": ["all_benign","all_early","all_pre","all_pro"], "topk": 3},
    "Cervical":
        {"classes": ["cervix_dyk","cervix_koc","cervix_mep","cervix_pab","cervix_sfi"], "topk": 3},
}

# =====================================================
# SELECT ORGAN
# =====================================================
selected_organ = st.selectbox("🧠 Select Organ Type", list(MODEL_URLS.keys()))

# =====================================================
# LOAD MODEL (CACHED)
# =====================================================
@st.cache_resource
def load_model(model_key):

    model_path = download_model(model_key)

    checkpoint = torch.load(model_path, map_location=device)

    model = densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

    img_size = checkpoint["img_size"]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    return model, idx_to_class, transform


if "current_model" not in st.session_state or st.session_state["current_model"] != selected_organ:
    model, idx_to_class, transform = load_model(selected_organ)
    st.session_state["model"] = model
    st.session_state["idx_to_class"] = idx_to_class
    st.session_state["transform"] = transform
    st.session_state["current_model"] = selected_organ
else:
    model = st.session_state["model"]
    idx_to_class = st.session_state["idx_to_class"]
    transform = st.session_state["transform"]

# =====================================================
# PREDICTION
# =====================================================
def predict_by_organ(image):

    config = ORGAN_CONFIG[selected_organ]
    allowed = config["classes"]
    topk = config["topk"]

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(img_tensor), dim=1)[0]

    results = []
    for i, p in enumerate(probs):
        cname = idx_to_class[i]
        if cname in allowed:
            results.append((cname, p.item()))

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:topk]

    output = []
    for cname, prob in results:
        subclass, desc = CLASS_INFO[cname]
        output.append((cname, subclass, desc, prob*100))

    return output, img_tensor

# =====================================================
# GRADCAM
# =====================================================
def run_gradcam(img_tensor, pred_class):

    cam = GradCAM(model=model, target_layers=[model.features[-1]])

    heatmap = cam(
        input_tensor=img_tensor,
        targets=[ClassifierOutputTarget(pred_class)]
    )[0]

    heatmap = (heatmap - heatmap.min()) / (heatmap.max()+1e-8)

    img_np = img_tensor.squeeze().permute(1,2,0).cpu().numpy()
    img_np = (img_np-img_np.min())/(img_np.max()-img_np.min())

    return img_np, heatmap

# =====================================================
# LIME
# =====================================================
def predict_fn(images):

    batch=[]
    for img in images:
        t = transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(device)
        batch.append(t)

    batch=torch.cat(batch)

    with torch.no_grad():
        probs = F.softmax(model(batch),dim=1).cpu().numpy()

    return probs

def run_lime(img_np):

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=predict_fn,
        top_labels=1,
        num_samples=100   # reduced for speed
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=1,
        hide_rest=False
    )

    return mark_boundaries(temp, mask)

# =====================================================
# UI
# =====================================================
uploaded_file = st.file_uploader("Upload Image", ["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🔬 Run AI Diagnosis"):

        with st.spinner("Running diagnosis..."):
            results, img_tensor = predict_by_organ(image)

        st.session_state["results"] = results
        st.session_state["img_tensor"] = img_tensor

        st.success("Diagnosis Complete")

    if "results" in st.session_state:

        st.subheader("🧾 Diagnosis Report")

        for i,(c,sub,desc,prob) in enumerate(st.session_state["results"],1):
            st.markdown(f"""
            ### {i}. {c}
            **Subclass:** {sub}  
            **Probability:** {prob:.2f}%  
            **Description:** {desc}
            """)
            st.progress(prob/100)

        st.divider()

        if st.button("🧠 Run XAI Explanation"):

            with st.spinner("Generating explanations..."):

                img_tensor = st.session_state["img_tensor"]
                pred_class = torch.argmax(model(img_tensor),1).item()

                img_np, heatmap = run_gradcam(img_tensor, pred_class)
                lime_img = run_lime((img_np*255).astype(np.uint8))

            st.subheader("🧠 Explainable AI (XAI)")

            fig,axs=plt.subplots(1,3,figsize=(15,5))

            axs[0].imshow(img_np)
            axs[0].set_title("Original")

            axs[1].imshow(img_np)
            axs[1].imshow(heatmap,cmap="jet",alpha=0.5)
            axs[1].set_title("Grad-CAM")

            axs[2].imshow(lime_img)
            axs[2].set_title("LIME")

            for ax in axs:
                ax.axis("off")

            st.pyplot(fig)
