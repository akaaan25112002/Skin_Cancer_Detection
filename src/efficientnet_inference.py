import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# ==========================
# CONFIG
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best threshold from notebook
BEST_THRESHOLD = 0.5299999999999998
# ==========================
# LOAD MODEL
# ==========================
num_classes = 2

model = timm.create_model("efficientnet_b2", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

model.load_state_dict(torch.load("models/best_efficientnet_b2.pt", map_location=device))
model.to(device)
model.eval()

# ==========================
# TRANSFORMS
# ==========================
infer_tf = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# ==========================
# INFERENCE FUNCTION
# ==========================
def predict_efficientnet(image, threshold=BEST_THRESHOLD):
    """
    Input: PIL image
    Output: (pred, prob)
    """
    img_tensor = infer_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        malignant_prob = probs[:, 1].item()

    pred = 1 if malignant_prob >= threshold else 0
    return pred, malignant_prob