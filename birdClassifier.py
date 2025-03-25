import torch
from torch import nn
from torch import optim
from torchvision import models
from CreateTrainAndTest import CustomDataSet


# Function to load and split data
def load_split_train_test():
    from torchvision import transforms
 
    train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize first to keep aspect ratio
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_data = CustomDataSet("./CUB_200_2011/CUB_200_2011/images/", train_transforms, "./CUB_200_2011/CUB_200_2011/images.txt", 
                               "./CUB_200_2011/CUB_200_2011/train_test_split.txt", "./CUB_200_2011/CUB_200_2011/bounding_boxes.txt", "train")
    test_data = CustomDataSet("./CUB_200_2011/CUB_200_2011/images/", test_transforms, "./CUB_200_2011/CUB_200_2011/images.txt", 
                               "./CUB_200_2011/CUB_200_2011/train_test_split.txt", "./CUB_200_2011/CUB_200_2011/bounding_boxes.txt", "test")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
     

    return train_loader, test_loader


# Load data
trainloader, testloader = load_split_train_test()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained ResNet-50
#model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(1408, 200),
    nn.Dropout(0.5),
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam([
    {'params': model.classifier.parameters(), 'lr': 0.001},
    {'params': model.features[-4:].parameters(), 'lr': 1e-3},
    {'params': model.features[-6:-4].parameters(), 'lr': 1e-5}

])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  

model.to(device)

# Training
epochs = 20
best_accuracy = 0
print_every = 10
train_losses, test_losses, accuracies = [], [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model(inputs)
        loss = criterion(logps, labels)
       
        running_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    
    # Validation after each epoch
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.inference_mode():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            
            test_loss += criterion(logps, labels).item()
            #ps = torch.exp(logps)
            top_p, top_class = logps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    # Calculate average losses and accuracy per epoch
    avg_train_loss = running_loss / len(trainloader)
    avg_test_loss = test_loss / len(testloader)
    avg_accuracy = accuracy / len(testloader)
    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    accuracies.append(avg_accuracy)
    
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {avg_train_loss:.3f}.. "
          f"Validation loss: {avg_test_loss:.3f}.. "
          f"Validation accuracy: {avg_accuracy:.3f}")
        
    # Save the best model
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        torch.save(model.state_dict(), 'birdmodelbest.pth')
# Save model
torch.save(model.state_dict(), 'birdmodel.pth')

