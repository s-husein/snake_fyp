from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import torchvision.transforms.v2 as tf
from agent import SnakeImit
from paths import MISC_DIR
from dataset import DepthImageDataModule
import matplotlib.pyplot as plt

class Params:
    def __init__(self):
        self.name = 'snake_imitation'
        self.model_type = 'cnn_lstm'
        self.schedular = False
        self.conv_layers = [[4, 5, 5]]
        self.avg_pool = [8, 8]
        self.lstm_layers = 2
        self.seq_len = 10
        self.seq_step = 5
        self.max_pool = None
        self.pool_after_layers = 1
        self.act_fn = 'relu'
        self.batch_norm = True
        self.dropout = 0.3
        self.hid_layers = [256]
        self.lr = 1e-4
        self.epochs = 200
        self.clip_grad = 0.5
        self.metric_param = 'val_acc'
        self.train_batch_size = 16
        self.test_batch_size = 32
        self.val_batch_size = 32
        self.test_trans = tf.Compose([
            tf.Resize((90, 160)),
            tf.ToTensor(),
            tf.Normalize([0.5], [0.5])
        ])
        self.train_trans = tf.Compose([
            tf.Resize((90, 160)),
            tf.ToTensor(),
            tf.Normalize([0.5], [0.5]),
        ])
        self.val_trans = tf.Compose([
            tf.Resize((90, 160)),
            tf.ToTensor(),
            tf.Normalize([0.5], [0.5]),
        ])
        self.plot_params = ["Training Loss", "Training Accuracy", "Validation Loss", 'Validation Accuracy']


params = Params()

data_module = DepthImageDataModule(train_trans = params.train_trans,
                                                val_trans = params.val_trans,
                                                test_trans = params.test_trans,
                                                train_batch_size = params.train_batch_size,
                                                test_batch_size = params.test_batch_size,
                                                val_batch_size= params.val_batch_size,
                                                sequence_len=params.seq_len,
                                                sequence_step=params.seq_step)



params = Params()

def params_to_text(params):
    lines = ["Training Configuration:\n"]
    for key, value in vars(params).items():
        if isinstance(value, tf.Compose):
            transform_list = [t.__class__.__name__ for t in value.transforms]
            lines.append(f"{key}: {transform_list}")
        elif key == "custom_model" and value is not None:
            lines.append(f"{key}: {value.__class__.__name__}")
        else:
            lines.append(f"{key}: {value}")
    text = "\n\n".join(lines)
    with open(f'{MISC_DIR}/hyperparams.txt', "+a") as f:
        f.write(text)

params_to_text(params)


agent = SnakeImit(params)

agent.train()




