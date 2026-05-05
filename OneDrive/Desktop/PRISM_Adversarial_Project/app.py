import streamlit as st
import torch
from PIL import Image
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

from models.resnet import load_model
from src.preprocess import preprocess_image

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="AegisAI Dashboard", layout="wide")

# ---------------------- FULL DARK UI ----------------------
st.markdown("""
<style>

/* FULL BACKGROUND */
html, body, .stApp {
    background-color: #0a192f !important;
    color: white !important;
}

/* REMOVE HEADER */
header {visibility: hidden;}
.block-container {padding-top: 1rem;}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #112240 !important;
}

/* TEXT */
h1, h2, h3, h4, h5, h6, p, span {
    color: white !important;
}

/* CARDS */
.card {
    background-color: #112240;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
}

/* BUTTON */
.stButton>button {
    background-color: #1f6feb;
    color: white;
    border-radius: 8px;
}

/* UPLOADER */
.stFileUploader {
    background-color: #112240;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------- MODEL ----------------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ---------------------- LABELS ----------------------
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return urllib.request.urlopen(url).read().decode("utf-8").splitlines()

labels = load_labels()

# ---------------------- FGSM ATTACK ----------------------
def fgsm_attack(model, image, epsilon, target_class=None):
    image = image.clone().detach().requires_grad_(True)

    output = model(image)

    if target_class is None:
        label = output.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(output, label)
        direction = 1
    else:
        target = torch.tensor([target_class])
        loss = torch.nn.functional.cross_entropy(output, target)
        direction = -1

    model.zero_grad()
    loss.backward()

    adv = image + direction * epsilon * image.grad.sign()
    adv = torch.clamp(adv, 0, 1)

    return adv.detach()

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("🛡️ AegisAI Dashboard")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Navigation", ["Dashboard", "Run Attack"])

epsilon = st.sidebar.slider("Noise ε", 0.0, 0.1, 0.02, 0.005)
targeted = st.sidebar.checkbox("Targeted Attack")

target_class = None
if targeted:
    target_label = st.sidebar.selectbox("Target Class", labels)
    target_class = labels.index(target_label)

use_sample = st.sidebar.checkbox("Use Sample Image")

# ====================== DASHBOARD ======================
if mode == "Dashboard":

    st.title("📊 Security Overview")

    if "orig_class" in st.session_state:
        attack_done = 1
        success = int(st.session_state["orig_class"] != st.session_state["adv_class"])
    else:
        attack_done = 0
        success = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f'<div class="card"><h2>{attack_done}</h2><p>Attacks Run</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><h2>{success}</h2><p>Successful Attacks</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><h2>{success*100}%</h2><p>Success Rate</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><h2>{"Active" if attack_done else "Idle"}</h2><p>System Status</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    if "orig_out" not in st.session_state:
        st.warning("⚠️ Run an attack first to see results.")
        st.stop()

    orig_out = st.session_state["orig_out"]
    adv_out = st.session_state["adv_out"]
    orig_class = st.session_state["orig_class"]
    adv_class = st.session_state["adv_class"]

    st.subheader("📊 Last Attack Result")

    st.markdown(f"""
    <div class="card">
    <p><b>Original:</b> {orig_class}</p>
    <p><b>Adversarial:</b> {adv_class}</p>
    </div>
    """, unsafe_allow_html=True)

    if orig_class != adv_class:
        st.success("Attack Successful ✅")
    else:
        st.warning("Attack Failed ⚠️")

# ====================== RUN ATTACK ======================
elif mode == "Run Attack":

    st.title("⚡ Run Adversarial Attack")

    image = None

    if use_sample:
        image = Image.open("sample.jpg").convert("RGB")
    else:
        uploaded = st.file_uploader("Upload Image")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    if image:
        st.image(image, caption="Input Image", width=250)

        if st.button("🚀 Launch Attack"):

            input_tensor = preprocess_image(image)

            # ORIGINAL
            with torch.no_grad():
                orig_out = model(input_tensor)
                orig_class = labels[orig_out.argmax().item()]

            # ATTACK
            adv_img = fgsm_attack(model, input_tensor, epsilon, target_class)

            # ADVERSARIAL
            with torch.no_grad():
                adv_out = model(adv_img)
                adv_class = labels[adv_out.argmax().item()]

            # SAVE
            st.session_state["orig_out"] = orig_out
            st.session_state["adv_out"] = adv_out
            st.session_state["orig_class"] = orig_class
            st.session_state["adv_class"] = adv_class

            # SHOW IMAGES
            col1, col2 = st.columns(2)

            col1.image(input_tensor.squeeze().permute(1,2,0).detach().numpy(), caption="Original")
            col2.image(adv_img.squeeze().permute(1,2,0).detach().numpy(), caption="Adversarial")

            st.markdown("---")

            # RESULT
            st.markdown(f"""
            <div class="card">
            <h3>Result Summary</h3>
            <p><b>Original:</b> {orig_class}</p>
            <p><b>Adversarial:</b> {adv_class}</p>
            </div>
            """, unsafe_allow_html=True)

            if orig_class != adv_class:
                st.success("Attack Successful ✅")
            else:
                st.warning("Attack Failed ⚠️")

            # GRAPH
            st.markdown("---")
            st.subheader("📊 Confidence Shift")

            orig_probs = torch.nn.functional.softmax(orig_out[0], dim=0)
            adv_probs = torch.nn.functional.softmax(adv_out[0], dim=0)

            top5_orig_prob, top5_orig_idx = torch.topk(orig_probs, 5)
            top5_adv_prob, top5_adv_idx = torch.topk(adv_probs, 5)

            combined = list(set(top5_orig_idx.tolist() + top5_adv_idx.tolist()))

            orig_vals = [float(orig_probs[i]) for i in combined]
            adv_vals = [float(adv_probs[i]) for i in combined]

            names = [labels[i] for i in combined]

            x = np.arange(len(names))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.bar(x - width/2, orig_vals, width, label='Original')
            ax.bar(x + width/2, adv_vals, width, label='Adversarial')

            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=25, ha='right')

            plt.tight_layout()

            ax.set_ylabel("Confidence")
            ax.set_title("Confidence Change After Attack")
            ax.legend()

            st.pyplot(fig)