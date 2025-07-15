import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from nets import make_cnn, make_cnn_lstm
from dataset import DepthImageDataModule
from paths import *
import matplotlib.pyplot as plt
from utils import Utils
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import datetime as dt
import time
from drive import GoogleDrive

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
                                                val_batch_size= self.params.val_batch_size,
                                                sequence_len= self.params.seq_len,
                                                sequence_step= self.params.seq_step)

        self.criterion = CrossEntropyLoss()
        self.acc_param = False
        if self.params.metric_param in ['train_acc', 'val_acc']:
            self.acc_param = True

        self.drive = GoogleDrive(MISC_DIR)


        self.check_status()
        self.create_model()
    
    def create_model(self):
        if hasattr(self.params, 'custom_model'):
            pass
        else:
            if hasattr(self.params, 'model_type') and self.params.model_type == 'cnn_lstm':
                self.model = make_cnn_lstm(
                    dataset=self.data_module.train_ds,
                    lstm_hidden=self.params.hid_layers[0],
                    lstm_layers=self.params.lstm_layers,
                    hid_layers=self.params.hid_layers,
                    act_fn=self.params.act_fn,
                    max_pool=self.params.max_pool,
                    avg_pool=self.params.avg_pool,
                    pooling_after_layers=self.params.pool_after_layers,
                    batch_norm=self.params.batch_norm,
                    conv_layers=self.params.conv_layers,
                    dropout=self.params.dropout
                ).to(pu)
            else:
                self.model = make_cnn(
                    dataset=self.data_module.train_ds,
                    hid_layers=self.params.hid_layers,
                    act_fn=self.params.act_fn,
                    max_pool=self.params.max_pool,
                    avg_pool=self.params.avg_pool,
                    pooling_after_layers=self.params.pool_after_layers,
                    batch_norm=self.params.batch_norm,
                    conv_layers=self.params.conv_layers,
                    dropout=self.params.dropout
                ).to(pu)
        self.optim = Adam(self.model.parameters(), lr=self.params.lr)
        if self.configs['status'] == 'not_started':
            self.params_to_text()
            text = f'''\n{30*'=='}\nTraining Dataset: {self.data_module.train_ds.__report__()}
            \n{30*'=='}\nValidation Dataset: {self.data_module.val_ds.__report__()}
            \n{30*'=='}\nTesting Dataset: {self.data_module.test_ds.__report__()}"
            \n{30*'=='}\nModel: {self.model}
            \n{30*'=='}\nTotal number of parameters: {sum([p.numel() for p in self.model.parameters()])}
            \nNumber of classes: {len(self.data_module.train_ds.classes)}
            \nClasses: {self.data_module.train_ds.classes}
            \nInput image size (height, width): {self.data_module.train_ds.__getrawimage__().shape}
            \nTransformed image size (channels, height, width): {self.data_module.train_ds.__getitem__(0)[0].shape}\n'''

            print(text)

            with open(f'{MISC_DIR}/hyperparams.txt', '+a') as file:
                file.write(text)

        

    def train(self):
        print("Training...")
        epochs = self.configs['epochs']

        for epoch in range(epochs, self.params.epochs):
            self.prev_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.params.epochs}")

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()
            self.configs['total_elapsed_seconds'] += int(time.time() - self.prev_time)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")\
            
            self.write_plot_data(list(map(lambda y: round(y, 3), [train_loss, train_acc, val_loss, val_acc])))
            self.save_check_interval(epoch=epoch+1, interval=1, queue_size=30)
            self.save_best_model(round(val_loss, 4))
        

        with open(f'{MISC_DIR}/hyperparams.txt', '+a') as file:
            file.write(f'''\nTotal training time: {time.strftime("%H:%M:%S", time.gmtime(self.configs['total_elapsed_seconds']))}
                       \n{40*'=='}\nTraining stopped on {dt.datetime.now().strftime("Date: %d/%m/%Y, %a, at time: %H:%M")}\n{40*'=='}''')


        print("Evaluating on test set...")
        self._test_and_plot()
        self.save_report_pdf()

        if self.configs['status'] == 'finished':
            self.drive.upload_folder()


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
        text = f'''\nFinale classification report based on testing on best saved model:"
        {classification_report(all_labels, all_preds, target_names=self.data_module.train_ds.classes)}'''

        with open(f'{MISC_DIR}/hyperparams.txt', '+a') as file:
            file.write(text)

        print(text)

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


