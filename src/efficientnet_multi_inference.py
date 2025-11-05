import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ckpt
ckpt = torch.load("models/effb2_multi_best_stage2.pth", map_location=device)
CLASS_ORDER = ckpt["classes"]
num_classes = len(CLASS_ORDER)

# === build model EXACTLY like training ===
base = models.efficientnet_b2(weights=None)

in_feats = base.classifier[1].in_features
base.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_feats, num_classes)
)

model = base
model.load_state_dict(ckpt["state_dict"])
model.to(device)
model.eval()

infer_tf = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict_efficientnet_multi(image, topk=3):
    img = infer_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]

    cls_idx = probs.argmax().item()
    cls_name = CLASS_ORDER[cls_idx]
    confidence = probs[cls_idx].item()

    top_vals, top_idxs = torch.topk(probs, topk)
    top_list = [
        (CLASS_ORDER[i], float(top_vals[j]))
        for j, i in enumerate(top_idxs)
    ]

    return {
        "pred_class": cls_name,
        "confidence": float(confidence),
        "topk": top_list
    }