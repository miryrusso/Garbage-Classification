from torchvision import transforms
import torch, time
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

test_transform = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
])

def data_augumentation(set, class_to_aug, loader, times=0, batch_size=32):
  
  class_indices = [i for i, (_, label) in enumerate(set) if label == class_to_aug]

  class_augmented = Subset(set, class_indices)

  augmented_datasets = [class_augmented for _ in range(times)] 

  balanced_dataset = ConcatDataset([set] + augmented_datasets)

  loader = DataLoader(balanced_dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)

  return loader
  
