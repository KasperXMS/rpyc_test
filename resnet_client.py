import rpyc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
# predict images from MNIST dataset

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.ToTensor(),
])
    dataset = torchvision.datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)
    sampler = torch.utils.data.RandomSampler(dataset, num_samples = 4, replacement = True)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 4, sampler = sampler)

    c = rpyc.connect("192.168.1.133", 18861, config=rpyc.core.protocol.DEFAULT_CONFIG)
    # ramdomly sample 1 image from the dataset
    for i, (img, label) in enumerate(loader):
        result = c.root.predict(img.numpy())
        # plot the image and the prediction and the label
        plt.imshow(transforms.ToPILImage()(img[0]))
        plt.title(f"Prediction: {label_names[result[0]]}, Label: {label_names[label[0]]}")
        plt.show()
        break

    c.close()
    