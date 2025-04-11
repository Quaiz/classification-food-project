from setup import device, torch, nn, optim, lr_scheduler, models, summary, tqdm
from data_preprocess import train_loader, valid_loader, test_loader, root_train, Name_food
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torch.amp import GradScaler, autocast

# Xây dựng mô hình
def build_model(num_classes=5):
    model = models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(256, num_classes)
    )
    return model.to(device)

# Class weights
def get_class_weights(root_train):
    categories = [d for d in os.listdir(root_train) if os.path.isdir(os.path.join(root_train, d))]
    counts = [len(os.listdir(os.path.join(root_train, c))) for c in categories]
    n_samples = sum(counts)
    n_classes = len(counts)
    weight = [n_samples / (n_classes * count) for count in counts]
    weight[categories.index('BanhChung')] *= 2.5
    weight[categories.index('BanhXeo')] *= 1.5
    return torch.FloatTensor(weight).to(device)

# Loss function
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Checkpoint
def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, train_accs, val_accs, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['train_losses'], checkpoint['val_losses'], checkpoint['train_accs'], checkpoint['val_accs']

# Train mô hình
def train_model(model, train_loader, valid_loader, num_epochs=100, patience=20, start_epoch=0, checkpoint_path=None):
    class_weights = get_class_weights(root_train)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 0.0005},
        {'params': model.layer4.parameters(), 'lr': 0.0005},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ], weight_decay=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler('cuda')
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    if checkpoint_path and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, train_losses, val_losses, train_accs, val_accs = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resumed from epoch {start_epoch+1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):  # Sửa: dùng torch.amp.autocast
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({'loss': loss.item(), 'VRAM': f'{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB'})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(epoch_loss)
        train_accs.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valid_loader.dataset)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%")
        scheduler.step()

        checkpoint_path = f"D:\\ThirdYearsInHell\\deeplearning\\Project\\checkpoint\\checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch+1, train_losses, val_losses, train_accs, val_accs, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'D:\\ThirdYearsInHell\\deeplearning\\Project\\checkpoint\\best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping!")
                break

    return model, train_losses, val_losses, train_accs, val_accs

# Evaluate
def evaluate_model(model, data_loader, dataset_name="Dataset"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Classification Report for {dataset_name}:")
    print(classification_report(all_labels, all_preds, target_names=list(Name_food.values()), zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(Name_food.values()), yticklabels=list(Name_food.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.show()

# Chạy
if not os.path.exists('D:\\ThirdYearsInHell\\deeplearning\\Project\\checkpoint'):
    os.makedirs('D:\\ThirdYearsInHell\\deeplearning\\Project\\checkpoint')

model = build_model(num_classes=5)
summary(model, input_size=(3, 224, 224))

checkpoint_dir = 'D:\\ThirdYearsInHell\\deeplearning\\Project\\checkpoint'
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
checkpoint_path = os.path.join(checkpoint_dir, max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))) if checkpoint_files else None

model, train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, valid_loader, num_epochs=100, patience=20, checkpoint_path=checkpoint_path)

evaluate_model(model, valid_loader, "Validation")
evaluate_model(model, test_loader, "Test")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy')
plt.show()