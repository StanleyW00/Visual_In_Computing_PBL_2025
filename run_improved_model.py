import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

IMG_SIZE = 224
BATCH_SIZE = 32

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

train_dataset = datasets.ImageFolder("ISIC_dataset/Train", transform=transform)
val_dataset = datasets.ImageFolder("ISIC_dataset/Test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# Load the pretrained MAE encoder (e.g., ViT base)
model = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
model.to(device)


# Add a classification head manually
class MAEClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


final_model = MAEClassifier(model, num_classes=len(train_dataset.classes))
final_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)

EPOCHS = 10

for epoch in range(EPOCHS):
    final_model.train()
    total_loss, total_correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = final_model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader.dataset):.4f} | Acc: {total_correct / len(train_loader.dataset):.4f}"
    )

final_model.eval()
correct = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = final_model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

print(f"Validation Accuracy: {correct / len(val_loader.dataset):.4f}")
