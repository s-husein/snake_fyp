from paths import CHECKPOINT_DIR, MISC_DIR, create_dirs
import torch
import os
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys
import shutil
import datetime as dt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Utils:

    def read_file(self, path):
        file = open(path, 'r')
        file.seek(0)
        info = file.readline()
        file.close()
        return info

    def write_file(self, path, content):
        mode = 'w'
        if path == self.plot_file:
            mode = '+a'
        file = open(path, mode=mode)
        file.write(content)
        file.close()

    def create_file(self, path):
        file = open(path, 'w')
        file.close()

    def create_checkpoint_file(self, num):
        path = f'{CHECKPOINT_DIR}/checkpoint_{num}.pth'
        file = open(path, 'w')
        file.close()
        return path
    
    def save_config(self, args: dict):
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(args, file)

    def check_status(self):
        status = self.configs['status']
        if status == 'finished':
            comm = input("Training already finished, do you want to start a new training session (y/n): ")
            if comm == 'y':
                n_comm = input("\nThis is a destructive option, starting new training will delete the checkpoint and misc folder causing data loss make sure you have the data backed up...\nDo you still want to continue (y/n): ")
                if n_comm == 'y':
                    self.rem_and_create()
                elif n_comm == 'n':
                    print("Exiting...")
                    sys.exit()
            elif comm == 'n':
                print("Exiting...")
                sys.exit()
            else:
                print("Enter (y) or (n)...")
        elif status == 'not_started':
            file = open(self.plot_file, 'w')
            file.close()
            self.write_file(self.plot_file, ','.join(self.params.plot_params)+'\n')
            self.configs['status'] = 'in_progress'
            with open(f'{MISC_DIR}/hyperparams.txt', 'w') as file:
                file.write(f'Training started on {dt.datetime.now().strftime("Date: %d/%m/%Y, %a, at time: %H:%M")}\n')
            return
        elif status == 'in_progress':
            q = str(input('\nDo you want to continue the training (y/n)\nCaution: starting new training will remove all previous checkpoints and plot data..: '))
            if q == 'n':
                self.rem_and_create()
            elif q == 'y':
                self.create_model()
                epoch = self.configs['epochs']
                if epoch+1 >= self.params.epochs:
                    self.configs['status'] = 'finished'
                    with open(f'{MISC_DIR}/misc.yaml', 'w') as conf_file:
                        yaml.safe_dump(self.configs, conf_file)
                    return 
                checkpath = f'{CHECKPOINT_DIR}/checkpoint_{epoch}.pth'
                if os.path.exists(checkpath):
                    self.load_checkpoint(checkpath)
                    file = open(self.plot_file, 'r')
                    lines = file.readlines()
                    file = open(self.plot_file, 'w')
                    file.writelines(lines[:epoch+1])
                    file.close()
        return
        

    def write_plot_data(self, data):
        str_data = ','.join(map(str, data))
        self.write_file(self.plot_file, f'{str_data}\n')

    def save_plot(self):
        data = pd.read_csv(self.plot_file)

        with plt.style.context('default'):
            fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

            axs[0].plot(data['Training Loss'], label='Train Loss', color='blue', linewidth=2)
            axs[0].plot(data['Validation Loss'], label='Val Loss', color='orange', linestyle='--', linewidth=2)
            axs[0].set_ylabel('Loss', fontsize=13)
            axs[0].grid(True, linestyle='--')
            axs[0].legend(fontsize=11)

            # Plot Accuracy
            axs[1].plot(data['Training Accuracy'], label='Train Accuracy', color='green', linewidth=2)
            axs[1].plot(data['Validation Accuracy'], label='Val Accuracy', color='red', linestyle='--', linewidth=2)
            axs[1].set_xlabel('Epochs', fontsize=13)
            axs[1].set_ylabel('Accuracy', fontsize=13)
            axs[1].grid(True, linestyle='--')
            axs[1].legend(fontsize=11)

            plt.xticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{MISC_DIR}/plot.png', dpi=300)
            plt.close('all')

        del data

    def save_checkpoint(self, epoch, checkpath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': epoch
        }
        self.configs['epochs'] = epoch
        with open(f'{MISC_DIR}/misc.yaml', 'w') as conf_file:
            yaml.safe_dump(self.configs, conf_file)
        torch.save(checkpoint, checkpath)
        print('checkpoint saved..')
    
    def load_checkpoint(self, checkpath):
        print('loading checkpoint..')
        checkpoint = torch.load(checkpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        self.model.train()
        print('checkpoint loaded...')
    
    def save_check_interval(self, epoch, interval=50, queue_size=50):
        if not(epoch % interval) and epoch > 0:
            if epoch >= self.params.epochs-1:
                self.configs['status'] = 'finished'
            checkpath = self.create_checkpoint_file(epoch)
            self.save_checkpoint(epoch, checkpath)
            self.save_plot()
            del_checkpoint = f'{CHECKPOINT_DIR}/checkpoint_{epoch - (queue_size*interval)}.pth'
            if os.path.exists(del_checkpoint):
                os.remove(del_checkpoint)

    
    def load_model(self):
        print('loading model...')
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()
        print('model loaded...')

    def save_model(self):
        print("Saving Model...")    
        torch.save(self.model.state_dict(), self.model_file)
        print('model saved...')

    def save_best_model(self, val_loss):
        if val_loss < self.configs['best_val_loss']:
            self.configs['best_val_loss'] = val_loss
            self.save_model()
            with open(f'{MISC_DIR}/misc.yaml', 'w') as conf_file:
                yaml.safe_dump(self.configs, conf_file)

    def rem_and_create(self):
        dirs_to_delete = [CHECKPOINT_DIR, MISC_DIR]
        for dir_path in dirs_to_delete:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                print(f"Deleting: {dir_path}")
                shutil.rmtree(dir_path)
        create_dirs()
        with open(f'{MISC_DIR}/misc.yaml') as config_file:
            self.configs = yaml.safe_load(config_file)
        self.check_status()