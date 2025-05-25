import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision.transforms.v2 as tf
from tqdm import tqdm
from nets import make_cnn
from dataset import DepthImageDataModule
from paths import *
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import Utils
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {pu}')

class SnakeImit(Utils):
    def __init__(self,params):
        self.params = params
        with open(f'{MISC_DIR}/misc.yaml') as config_file:
            self.configs = yaml.safe_load(config_file)
        self.model_file = f'{MISC_DIR}/{self.params.name}_model.pth'
        self.plot_file = f'{MISC_DIR}/{self.params.name}_plot.txt'
        self.data_module = DepthImageDataModule(train_trans = self.params.train_trans,
                                                val_trans = self.params.val_trans,
                                                test_trans = self.params.test_trans,
                                                train_batch_size = self.params.train_batch_size,
                                                test_batch_size = self.params.test_batch_size,
                                                val_batch_size= self.params.val_batch_size)

        self.criterion = CrossEntropyLoss()
        self.check_status()
        self.acc_param = False
        if self.params.metric_param in ['train_acc', 'val_acc']:
            self.acc_param = True
    
    def to_one_hot(self, x, length):
        batch_size = x.size(0)
        x_one_hot = torch.zeros(batch_size, length)
        for i in range(batch_size):
            x_one_hot[i, x[i]] = 1.0
        return x_one_hot
    
    def create_model(self):
        if hasattr(self.params, 'custom_model'):
            pass
        else:
            self.model = make_cnn(dataset=self.data_module.train_ds, hid_layers=self.params.hid_layers, act_fn=self.params.act_fn,
                                max_pool=self.params.max_pool, pooling_after_layers=self.params.pool_after_layers,
                                batch_norm=self.params.batch_norm, conv_layers=self.params.conv_layers, dropout=self.params.dropout).to(pu)
        
        self.optim = Adam(self.model.parameters(), lr=self.params.lr)

        print(f'Model: {self.model}')
        print(f'Number of classes: {len(self.data_module.train_ds.classes)}')
        print(f'Input image size: {self.data_module.train_ds.__getitem__(0)[0].shape}')
        print(f'total number of parameters: {sum([p.numel() for p in self.model.parameters()])}')
        

    def train(self):
        print("Training...")
        epochs = self.configs['epochs']

        for epoch in range(epochs, self.params.epochs):
            print(f"\nEpoch {epoch+1}/{self.params.epochs}")

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")\
            
            self.write_plot_data(list(map(lambda y: round(y, 3), [train_loss, train_acc, val_loss, val_acc])))
            self.save_check_interval(epoch=epoch+1, interval=1, queue_size=30)
            self.save_best_model(round(val_loss, 4))


        print("Evaluating on test set...")
        self._test_and_plot()

    def _train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in self.data_module.train_loader:
            images, labels = images.to(pu), labels.to(pu)

            self.optim.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.params.clip_grad)
            self.optim.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

    def _validate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.data_module.val_loader:
                images, labels = images.to(pu), labels.to(pu)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return running_loss / total, correct / total

    def _test_and_plot(self):
        print("Testing the best model...")
        self.load_model()

        all_preds = []
        all_labels = []

        # Run inference
        with torch.no_grad():
            for images, labels in self.data_module.test_loader:
                images, labels = images.to(pu), labels.to(pu)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Print classification report
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.data_module.train_ds.classes))

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Plot and save confusion matrix
        plt.figure(figsize=(11, 9))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=self.data_module.train_ds.classes, 
                    yticklabels=self.data_module.train_ds.classes, cmap="viridis")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        # Ensure the folder exists
        os.makedirs(MISC_DIR, exist_ok=True)
        plt.savefig(f"{MISC_DIR}/confusion_matrix.png", dpi=300)
        plt.close('all')


