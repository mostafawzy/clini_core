import torch
import timm
import json
from PIL import Image
import io
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download


import torch
import torch.nn.functional as F
import timm
import json
import io

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


class VisionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = hf_hub_download(
            repo_id="menna143/skin-classifier-EfficientNet-B5",
            filename="efficientnet_b5_ft_best.pth"
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)

        #  LOAD CONFIG 
        self.class_names = checkpoint["class_names"]
        num_classes = len(self.class_names)
        img_size = checkpoint["input_size"]

        #  BUILD MODEL 
        self.model = timm.create_model(
            "efficientnet_b5",   
            pretrained=False,
            num_classes=num_classes
        )

        #  LOAD WEIGHTS 
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

        #  TRANSFORMS 
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    async def classify(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = self.transform(image)
        assert isinstance(x, torch.Tensor)
        x = x.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]

        top_idx = int(torch.argmax(probs))
        predicted_class = self.class_names[top_idx]
        confidence = float(probs[top_idx])

        topk = torch.topk(probs, k=min(5, len(self.class_names)))

        top_predictions = [
            {
                "label": self.class_names[int(i)],
                "score": float(probs[int(i)])
            }
            for i in topk.indices
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions
        }


_vision_service = VisionService()


def get_vision_service():
    return _vision_service