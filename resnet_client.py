import rpyc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

# predict images from MNIST dataset

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.ToTensor(),
])
    dataset = torchvision.datasets.ImageNet(root = "data", split = "val", transform = transform, download = True)
    sampler = torch.utils.data.RandomSampler(dataset, num_samples = 1, replacement = True)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, sampler = sampler)

    c = rpyc.connect("192.168.1.133", 18861)
    # ramdomly sample 1 image from the dataset
    for i, (img, label) in enumerate(loader):
        result = c.root.predict(img)
        # plot the image and the prediction and the label
        plt.imshow(img[0][0], cmap = "gray")
        plt.title(f"Prediction: {result[0]}, Label: {label[0]}")
        plt.show()
        break

    c.close()
    