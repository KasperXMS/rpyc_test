import rpyc
from rpyc.utils.server import ThreadedServer
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import os, time

# A ResNet-18 server that classify MNIST images

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResNetService(rpyc.Service):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.model.to("cuda")

    def on_connect(self, conn):
        # get the client IP address
        print("Connection established from", conn._channel.stream.sock.getpeername()[0])

    def on_disconnect(self, conn):
        print("Connection closed")

    def exposed_predict(self, img_tensor):
        with torch.no_grad():
            img_tensor = img_tensor.to("cuda")
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
        

if __name__ == "__main__":
    t = ThreadedServer(ResNetService, port = 18861)
    t.start()