import torch
from torchvision.models import resnet50,vgg16
from PIL import Image
import json, os, time
from torchvision import transforms

class Classifier:
    def __init__(self, model_path, type, device, num_classes):
        self.model_path = model_path
        self.type = type # ResNet or VGG
        self.device = device
        self.num_classes = num_classes

        if type == 'resnet':
            #model_path = 'checkpoints/best_model_lr_0.01_weight_decay_0.0001.pth'
            self.model = resnet50(weights=None)
            self.model.fc = torch.nn.Linear(2048, num_classes)  # load pre-trained vgg16 model
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.model = self.model.to(device)

        elif type == 'vgg':
            self.model = vgg16(weights = None)
            self.model.classifier[6] = torch.nn.Linear(4096, num_classes)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.model = self.model.to(device)


    def run(self, image_path):
        """
        Args:
            image: Path to the image file

        Returns:
            Whether pollinator exists in the image or not
        """
        image = Image.open(image_path).convert('RGB')  # load the image using PIL
        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        image = transform(image)
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)

        with torch.no_grad():# no updation of gradient based on the validation data
            out = self.model(image)
        return torch.argmax(out)

if __name__ == "__main__":
    model_path = '../models/vgg_model.pt'
    num_classes = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    type = 'vgg'

    image_path = 'C:/Gaurav/Code/DL/pollinator-project/Pollinators-18_COCO_640x640_aug_null/test/Copy-of-motion_2022-06-12_17_29_49_91_mp4-32_jpg.rf.d271f0928f7284095faf4b9624181a86.jpg'
 # transform image
    classifier = Classifier(model_path, type, device, num_classes)
    print(classifier.run(image_path))
