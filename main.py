from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import torchvision.transforms.v2 as tf
from agent import SnakeImit
from paths import MISC_DIR

class Params:
    def __init__(self):
        self.name = 'snake_imitation'
        self.schedular = False
        self.conv_layers = [[4, 5, 5]]
        self.max_pool = [2, 2]
        self.pool_after_layers = 1
        self.act_fn = 'relu'
        self.batch_norm = False
        self.dropout = None
        self.hid_layers = [256, 256]
        self.lr = 1e-3
        self.epochs = 10
        self.clip_grad = 0.5
        self.metric_param = 'val_acc'
        self.train_batch_size = 16
        self.test_batch_size = 32
        self.val_batch_size = 32
        self.test_trans = tf.Compose([
            tf.Resize((18, 36)),
            tf.ToTensor(),
            tf.Normalize([0.5], [0.5])
        ])
        self.train_trans = tf.Compose([
            tf.Resize((18, 36)),
            tf.ToTensor(),
            tf.Normalize([0.5], [0.5]),
        ])
        self.val_trans = tf.Compose([
            tf.Resize((18, 36)),
            tf.ToTensor(),
            tf.Normalize([0.5], [0.5]),
        ])
        self.plot_params = ["Training Loss", "Training Accuracy", "Validation Loss", 'Validation Accuracy']

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
    text = "\n".join(lines)
    with open(f'{MISC_DIR}/hyperparams.txt', "w") as f:
        f.write(text)

params_to_text(params)


agent = SnakeImit(params)

# print(f'training dataset size: {len(agent.dataset.train_ds)}')
# print(f'validation dataset size: {len(agent.dataset.val_ds)}')

agent.train()



