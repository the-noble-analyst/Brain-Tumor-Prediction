# =============================================
# Brain MRI Clinical Assistant v3
# GitHub Version with Complete PDF Export
# =============================================
import streamlit as st
import os
import io
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from textwrap import wrap
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo # available from Python 3.9+

# Optional Gemini Import
try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

# ==============================
# 1️⃣ Streamlit Config
# ==============================
st.set_page_config(page_title="🧠 Brain MRI Clinical Assistant", layout="wide")

st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #4facfe, #00f2fe);
border-radius: 15px; color: white;">
    <h1>🧠 Brain MRI AI Assistant for Doctors</h1>
    <h4>AI-powered MRI analysis, Grad-CAM visualization, and Gemini AI conversation</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# 2️⃣ Gemini API Setup
# ==============================
if GEMINI_AVAILABLE:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = None
            st.warning("⚠️ Gemini API key not found in secrets. AI features will be limited.")
    except Exception as e:
        client = None
        st.warning(f"⚠️ Gemini initialization error: {e}")
else:
    client = None
    st.warning("⚠️ Gemini SDK not available. Install with: pip install google-genai")

def generate_gemini_text(prompt):
    """Generate text using Gemini API"""
    if client is None:
        return "Gemini AI is not configured. Please add your API key to enable AI summaries."
    try:
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = resp.text.strip()
    except Exception as e:
        text = f"Gemini Error: {e}"
    return clean_text(text)

def clean_text(s):
    """Clean and normalize text"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ==============================
# 3️⃣ Session State Initialization
# ==============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None
if "dr_name" not in st.session_state:
    st.session_state.dr_name = ""
if "hospital_name" not in st.session_state:
    st.session_state.hospital_name = ""
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_email" not in st.session_state:
    st.session_state.patient_email = ""
if "patient_address" not in st.session_state:
    st.session_state.patient_address = ""

# ==============================
# 4️⃣ Authentication Sidebar
# ==============================
if not st.session_state.authenticated:
    with st.sidebar:
        st.image("https://tse3.mm.bing.net/th/id/OIP.mkNQTA9e60kIima-KVR7PgHaFv?rs=1&pid=ImgDetMain", width=80)
        st.title("🔐 Doctor Authentication")
        
        with st.form("auth_form"):
            doctor_id = st.text_input("Doctor ID *", placeholder="(use doctorxyz)")
            dr_name = st.text_input("Doctor Name *", placeholder="Dr. Name")
            hospital = st.text_input("Hospital Name *", placeholder="City General Hospital")
            patient = st.text_input("Patient Name *", placeholder="Patient Name")
            patient_email = st.text_input("Patient Email *", placeholder="patient@example.com")
            patient_address = st.text_area("Patient Address *", placeholder="Enter patient address")
            
            submit_btn = st.form_submit_button("🔓 Authenticate & Continue", type="primary")
            
            if submit_btn:
                if doctor_id == "doctorxyz" and dr_name and hospital and patient and patient_email and patient_address:
                    st.session_state.authenticated = True
                    st.session_state.dr_name = dr_name
                    st.session_state.hospital_name = hospital
                    st.session_state.patient_name = patient
                    st.session_state.patient_email = patient_email
                    st.session_state.patient_address = patient_address
                    st.rerun()
                else:
                    st.error("❌ Please enter valid Doctor ID and fill all required fields.")
        
        st.info("💡 Use a valid Doctor ID")
    
    st.warning("👈 Please authenticate using the sidebar to access MRI analysis features.")
    st.stop()

# ==============================
# 5️⃣ Authenticated View - Sidebar
# ==============================
with st.sidebar:
    st.image("https://tse3.mm.bing.net/th/id/OIP.mkNQTA9e60kIima-KVR7PgHaFv?rs=1&pid=ImgDetMain", width=80)
    st.success("✅ Authenticated")
    st.markdown("---")
    st.subheader("📋 Session Details")
    st.write(f"**Doctor:** {st.session_state.dr_name}")
    st.write(f"**Hospital:** {st.session_state.hospital_name}")
    st.write(f"**Patient:** {st.session_state.patient_name}")
    st.write(f"**Email:** {st.session_state.patient_email}")
    st.write(f"**Address:** {st.session_state.patient_address}")
    
    st.markdown("---")
    if st.button("🚪 Logout", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==============================
# 6️⃣ Class Names + Model Load
# ==============================
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(class_names))
    )
    model_path = os.path.join("models", "model4.pth")
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Model loading error: {e}")
        st.info("Please ensure model4.pth is in the 'models' folder")
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==============================
# 7️⃣ Image Transformations
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 8️⃣ Grad-CAM Helper
# ==============================
def generate_gradcam(model, input_tensor, target_class=None):
    gradients, activations = [], []
    target_layer = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return np.zeros((224, 224))

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_full_backward_hook(save_gradient)

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    handle1.remove()
    handle2.remove()

    if not gradients or not activations:
        return np.zeros((224, 224))

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.maximum(np.sum(weights[:, None, None] * acts, axis=0), 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam

# ==============================
# 9️⃣ PDF GENERATION (MULTI-PAGE)
# ==============================
def generate_pdf_report(image, label, conf_pct, dr_name, hospital_name, 
                        patient_name, patient_email, patient_address, 
                        ai_summary, overlay, chat_history):
    """Generate comprehensive multi-page PDF report"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Header
    c.setFillColorRGB(0.13, 0.55, 0.71)
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 50, "Brain MRI Analysis Report")
    
    # Report details
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    y = height - 110
    datetime_india = datetime.now(tz=ZoneInfo('Asia/Kolkata'))
    formatted_time = datetime_india.strftime('%Y-%m-%d %H:%M')
    c.drawString(50, y, f"Date: {formatted_time}")
    y -= 18
    c.drawString(50, y, f"Doctor: {dr_name}")
    c.drawString(350, y, f"Hospital: {hospital_name}")
    y -= 18
    c.drawString(50, y, f"Patient: {patient_name}")
    y -= 18
    c.drawString(50, y, f"Email: {patient_email}")
    y -= 18
    
    # Address with wrapping
    c.drawString(50, y, "Address:")
    y -= 14
    address_lines = wrap(patient_address, width=70)
    for addr_line in address_lines[:3]:  # Max 3 lines for address
        c.drawString(70, y, addr_line)
        y -= 14
    y -= 10
    
    # Diagnosis section
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.4, 0.31, 0.64)
    c.drawString(50, y, "Diagnosis Result:")
    y -= 24
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(50, y, f"Prediction: {label.title()}")
    y -= 18
    c.drawString(50, y, f"Confidence: {conf_pct:.2f}%")
    y -= 35
    
    # AI Summary - COMPLETE
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.07, 0.6, 0.56)
    c.drawString(50, y, "AI Clinical Summary:")
    y -= 24
    c.setFillColorRGB(0, 0, 0)
    
    summary_paragraphs = [p.strip() for p in ai_summary.split('\n') if p.strip()]
    
    for paragraph in summary_paragraphs:
        if y < 80:
            c.showPage()
            y = height - 50
        
        wrapped_lines = wrap(paragraph, width=90)
        for line in wrapped_lines:
            if y < 80:
                c.showPage()
                y = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 14
        y -= 8
    
    y -= 20
    
    # Chat History - COMPLETE
    if chat_history and len(chat_history) > 0:
        if y < 150:
            c.showPage()
            y = height - 50
        
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.21, 0.58, 0.90)
        c.drawString(50, y, "Doctor-AI Discussion:")
        y -= 24
        c.setFillColorRGB(0, 0, 0)
        
        for msg in chat_history:
            if y < 100:
                c.showPage()
                y = height - 50
            
            role_prefix = f"[{msg['role']}]: "
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, role_prefix)
            
            msg_text = msg['text']
            wrapped_msg = wrap(msg_text, width=85)
            
            c.setFont("Helvetica", 10)
            role_width = c.stringWidth(role_prefix, "Helvetica-Bold", 10)
            
            for i, line in enumerate(wrapped_msg):
                if y < 80:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 10)
                
                if i == 0:
                    c.drawString(50 + role_width, y, line)
                else:
                    c.drawString(50, y, line)
                y -= 13
            
            y -= 10
    
    # Images on final page
    c.showPage()
    y = height - 60
    
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.4, 0.31, 0.64)
    c.drawCentredString(width / 2, y, "Visual Analysis")
    y -= 260
    
    try:
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        c.drawImage(ImageReader(img_buffer), 60, y, width=200, height=200)
        
        if overlay is not None:
            overlay_pil = Image.fromarray(overlay)
            overlay_buffer = io.BytesIO()
            overlay_pil.save(overlay_buffer, format='PNG')
            overlay_buffer.seek(0)
            c.drawImage(ImageReader(overlay_buffer), 340, y, width=200, height=200)
        
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(160, y - 25, "Original MRI")
        if overlay is not None:
            c.drawCentredString(440, y - 25, "Grad-CAM Analysis")
        
    except Exception as e:
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Error adding images: {e}")
    
    c.save()
    buffer.seek(0)
    return buffer

# ==============================
# 🔟 Upload & Predict
# ==============================
uploaded_file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="🩻 Uploaded MRI", use_container_width=True)
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    predicted_label = class_names[predicted_class.item()]
    conf_pct = confidence.item() * 100
    
    # Generate Grad-CAM
    overlay = None
    if predicted_label.lower() != "notumor":
        with col2:
            with st.spinner("🧩 Generating Grad-CAM visualization..."):
                try:
                    cam = generate_gradcam(model, input_tensor, predicted_class.item())
                    img_np = np.array(image.resize((224, 224)))
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    st.image(overlay, caption="🔥 Tumor Focus Region (Grad-CAM)", use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Grad-CAM generation error: {e}")
    else:
        with col2:
            st.info("✅ No tumor detected — Grad-CAM visualization is not required for normal scans.")

    st.markdown("---")
    
    # Diagnosis Result
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px;'>
        <h3 style='margin: 0;'>🎯 Diagnosis Result</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Predicted Tumor Type:** {predicted_label.title()}")
    st.markdown(f"**Confidence:** {conf_pct:.2f}%")

    st.markdown("#### 📊 Class Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls.title()}: {probs[0][i].item() * 100:.2f}%")
        st.progress(float(probs[0][i].item()))

    # ==============================
    # 1️⃣1️⃣ AI Summary (Generated Once)
    # ==============================
    st.markdown("---")
    st.subheader("🤖 AI Clinical Summary")

    if st.session_state.ai_summary is None:
        intro_prompt = f"""
You are a medical AI assistant providing a clinical summary for doctors.

Patient Name: {st.session_state.patient_name}
Doctor: {st.session_state.dr_name}
Hospital: {st.session_state.hospital_name}
Predicted Tumor Type: {predicted_label}
Confidence: {conf_pct:.2f}%

Provide a comprehensive clinical summary including:
- Medical significance of this diagnosis
- Typical characteristics and behavior of this tumor type
- Critical symptoms that require immediate attention
- Recommended follow-up tests and imaging
- Treatment considerations

Write in professional medical language suitable for healthcare providers.
Use 3-4 well-structured paragraphs.
"""
        
        with st.spinner("Generating AI clinical summary..."):
            st.session_state.ai_summary = generate_gemini_text(intro_prompt)
    
    st.write(st.session_state.ai_summary)

    # ==============================
    # 1️⃣2️⃣ Continue Chat
    # ==============================
    st.markdown("---")
    st.subheader("💬 Doctor-AI Discussion")

    if len(st.session_state.chat_history) == 0:
        st.info("💡 Ask questions about treatment options, prognosis, or request additional analysis.")
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["text"])
            else:
                st.chat_message("assistant").write(msg["text"])

    user_prompt = st.chat_input("Ask about treatment, progression, risk...")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "text": user_prompt})
        
        context = f"""
Patient: {st.session_state.patient_name}
Diagnosis: {predicted_label} with {conf_pct:.2f}% confidence

Conversation history:
"""
        context += "\n".join([f"{m['role']}: {m['text']}" for m in st.session_state.chat_history[-6:]])

        follow_prompt = f"""
{context}

Provide a professional medical response to the doctor's question.
Be concise, evidence-based, and clinically relevant.
"""
        
        with st.spinner("AI is thinking..."):
            reply_text = generate_gemini_text(follow_prompt)
        
        st.session_state.chat_history.append({"role": "assistant", "text": reply_text})
        st.rerun()

    # ==============================
    # 1️⃣3️⃣ PDF Generation
    # ==============================
    st.markdown("---")
    if st.button("📥 Generate Complete Clinical Report PDF", type="primary"):
        with st.spinner("Generating comprehensive PDF report..."):
            pdf_buffer = generate_pdf_report(
                image, 
                predicted_label, 
                conf_pct,
                st.session_state.dr_name, 
                st.session_state.hospital_name, 
                st.session_state.patient_name,
                st.session_state.patient_email,
                st.session_state.patient_address,
                st.session_state.ai_summary or "No AI summary available",
                overlay,
                st.session_state.chat_history
            )
            
            st.download_button(
                label="📄 Download Complete Report",
                data=pdf_buffer,
                file_name=f"brain_mri_report_{st.session_state.patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary"
            )
            st.success("✅ PDF report generated successfully!")

else:
    st.info("📎 Upload a Brain MRI image to start analysis.")

# ==============================
# 1️⃣4️⃣ Footer
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:14px;'>
<b>Brain MRI Clinical Assistant v3.0</b><br>
EfficientNet-B0 + Grad-CAM + Gemini AI + Multi-Page PDF Export<br>
Developed with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)




