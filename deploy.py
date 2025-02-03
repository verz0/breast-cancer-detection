import streamlit as st
from PIL import Image
import torch
from torchvision.models import densenet169, resnet50
from torchvision import transforms
import torch.nn.functional as F
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
@st.cache_resource
def load_densenet_model():
    model = densenet169(pretrained=False)
    model.classifier = torch.nn.Linear(1664, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

def load_resnet_model():
    model = resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("best_model-2.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model_choice = st.sidebar.selectbox("Select a model:", ["DenseNet-169", "ResNet-50"])
if model_choice == "DenseNet-169":
    model = load_densenet_model()
else:
    model = load_resnet_model()

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🩺 Breast Cancer Detection")
st.markdown("""
Welcome to the **Breast Cancer Detection** app. Upload an image, and the selected model will classify it into the appropriate category.  
This app offers two pre-trained models: **DenseNet-169** and **ResNet-50**, both fine-tuned on the **BreakHis dataset**.
""")

with st.sidebar:
    st.header("About the Models")
    st.markdown("""
    **DenseNet-169**
    - Dataset: BreakHis
    - Input Size: 128x128
    - Classes: [Benign, Malignant]

    **ResNet-50**
    - Dataset: BreakHis
    - Input Size: 128x128
    - Classes: [Benign, Malignant]
    """)

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Prediction Results")
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = output.argmax(dim=1).item()
            probabilities = F.softmax(output, dim=1).squeeze().tolist()

        class_labels = ["Benign", "Malignant"]
        st.write(f"**Predicted Class**: {class_labels[predicted_class]}")
        st.bar_chart(probabilities)

        for i, prob in enumerate(probabilities):
            st.write(f"{class_labels[i]}: **{prob * 100:.2f}%**")
else:
    st.info("Please upload an image to proceed.")
