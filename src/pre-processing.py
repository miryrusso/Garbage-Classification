from torchvision import transforms
import torch, time

# Trasformazioni per il training e il test
# Le trasformazioni sono utili per aumentare la varietà dei dati di training
# e per normalizzare le immagini in modo che abbiano media 0 e deviazione standard
train_transform = transforms.Compose([
    transforms.Resize(400),
    transforms.RandomCrop(384),  #384 perché EfficientNetV2_s é stata allenata su immagini di questa dimensione
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#tolto transforms.RandomHorizontalFlip()
test_transform = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
])

