import os
from PIL import Image

from torch.utils.data import Dataset
import torch

class CustomDataSet(Dataset):
    def __init__(self, dataDir, transform, imageTxt, splitTxt, boundingTxt, splitType):

        self.imageTxt = imageTxt
        self.splitTxt = splitTxt
        self.boundingTxt = boundingTxt
        self.splitType = splitType

        self.dataDir = dataDir
        self.transform = transform

        self.images = []
        self.labels = []
        self.bboxes = {}


        with open(imageTxt, "r") as f:
            with open(splitTxt, "r") as s:
                with open(boundingTxt, "r") as b:
                    while True:
                        imgPath = f.readline()
                        splitVal = s.readline()
                        box = b.readline()

                        if not imgPath:
                            break

                        imgPath = imgPath.strip("\n").split(" ")[1]
                        splitVal = splitVal.strip("\n").split(" ")[1]
                        box = box.strip("\n").split(" ", 1)[1]
                        bbox = tuple(map(float, box.split(" ")))  # Convert bbox values to integers
                        label = imgPath.split(".")[0]

                        
                        print(int(label)) if int(label) < 10 else None
                        if splitType == "train":
                            if splitVal == "0":
                                self.labels.append(int(label))
                                self.bboxes[imgPath] = bbox
                                self.images.append(imgPath)
                        elif splitType == "test":
                            if splitVal == "1":
                                self.labels.append(int(label))
                                self.images.append(imgPath)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        imgName = self.images[idx]
        imgPath = os.path.join(self.dataDir, imgName)

        image = Image.open(imgPath).convert("RGB")
        label = self.labels[idx]

        if imgName in self.bboxes:
            xMin, yMin, width, height = self.bboxes[imgName]
            image = image.crop((xMin, yMin, xMin + width, yMin + height))  # Crop image

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def getlabel(self, inStr):
        label = inStr.split(" ")[1].split("/")[0].split(".")[1]
        return label
        








