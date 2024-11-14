import os
import torch
from torch import optim
from network import U_Net
import csv
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_JS(SR_pred, GT, class_label):
    SR_class = SR_pred == class_label
    GT_class = GT == class_label
    Inter = torch.sum(SR_class & GT_class).item()
    Union = torch.sum(SR_class | GT_class).item()
    JS = float(Inter) / (float(Union) + 1e-6)
    return JS

def show_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        if isinstance(img, Image.Image):
            img = np.array(img)
        axs[i].imshow(img)
        axs[i].set_title(title)
        axs[i].axis('off')
    plt.show()

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.criterion = torch.nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model_type = config.model_type
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        # Simplified to use basic U-Net
        self.unet = U_Net(img_ch=3, output_ch=4)
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f.pkl' % (
        #     self.model_type, self.num_epochs, self.lr, self.augmentation_prob))

        # U-Net Train
        # if os.path.isfile(unet_path):
        #     # Load the pretrained Encoder
        #     self.unet.load_state_dict(torch.load(unet_path, weights_only=True))
        #     print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        # else:
            # Train for Encoder
        lr = self.lr

        for epoch in range(self.num_epochs):
            unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f.pkl' % (
                self.model_type, epoch, self.lr, self.augmentation_prob))
            self.unet.train(True)
            epoch_loss = 0

            JS_flippers = 0
            JS_carapace = 0
            JS_head = 0
            JS = 0.
            length = 0
            print('Starting epoch:', epoch + 1)

            for i, (images, GT, image_path, GT_path) in enumerate(self.train_loader):

                images = images.to(self.device)
                GT = GT.to(self.device).squeeze(1).long()

                SR = self.unet(images)
                SR_pred = SR.argmax(dim=1)

                print(f'Batch {i}:')

                loss = self.criterion(SR, GT)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                JS_flippers += get_JS(SR_pred, GT, class_label=1)
                JS_carapace += get_JS(SR_pred, GT, class_label=2)
                JS_head += get_JS(SR_pred, GT, class_label=3)
                JS = (JS_flippers + JS_carapace + JS_head) /3

                length += 1

            JS_flippers /= length
            JS_carapace /= length
            JS_head /= length
            JS /= length

            print(
                '[train] JS_flipper: %.4f, JS_carapace: %.4f, JS_head: %.4f, JS: %.4f' % (
                     JS_flippers, JS_carapace, JS_head, JS))

            torch.cuda.empty_cache()

            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))

        # ===================================== Validation ====================================#
            # Models are validated, not used for training
            self.unet.train(False)
            self.unet.eval()

            JS_flippers = 0
            JS_carapace = 0
            JS_head = 0
            JS = 0.
            length = 0

            with torch.no_grad():
                for i, (images, GT, image_path, GT_path) in enumerate(self.valid_loader):
                    images = images.to(self.device)
                    GT = GT.to(self.device).squeeze(1).long()

                    SR = self.unet(images)
                    SR_pred = SR.argmax(dim=1)

                    JS_flippers += get_JS(SR_pred, GT, class_label=1)
                    JS_carapace += get_JS(SR_pred, GT, class_label=2)
                    JS_head += get_JS(SR_pred, GT, class_label=3)
                    JS = (JS_flippers + JS_carapace + JS_head) /3

                    length += 1

            JS_flippers /= length
            JS_carapace /= length
            JS_head /= length
            JS /= length

            print('[Validation] JS_flipper: %.4f, JS_carapace: %.4f, JS_head: %.4f, JS: %.4f' % (
                JS_flippers, JS_carapace, JS_head, JS ))

            torch.save(self.unet.state_dict(), unet_path)

        # ===================================== Test ====================================#
        self.test()

    def test(self):
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f.pkl' % (
        self.model_type, self.num_epochs-1, self.lr, self.augmentation_prob))
        del self.unet
        self.build_model()
        if os.path.isfile(unet_path):
            self.unet.load_state_dict(torch.load(unet_path, weights_only=True))
        else:
            print(f"Model file not found: {unet_path}")

        self.unet.eval()

        JS_flippers = 0
        JS_carapace = 0
        JS_head = 0
        JS = 0.
        length = 0
        max_JS = -1
        min_JS = 100

        with torch.no_grad():
            for i, (images, GT, image_path, GT_path) in enumerate(self.valid_loader):
                images = images.to(self.device)
                GT = GT.to(self.device).squeeze(1).long()

                SR = self.unet(images)
                SR_pred = SR.argmax(dim=1)

                JS_flippers += get_JS(SR_pred, GT, class_label=1)
                JS_carapace += get_JS(SR_pred, GT, class_label=2)
                JS_head += get_JS(SR_pred, GT, class_label=3)
                JS = (JS_flippers + JS_carapace + JS_head) / 3

                if JS / (length + 1) > max_JS:
                    max_JS = JS / (length + 1)
                    best_img_pil = Image.open(image_path[0])
                    best_gt_pil = Image.open(GT_path[0])

                    inverse_mapping = {0: 0, 1: 44, 2: 100, 3: 144}

                    converted_pred = torch.full_like(SR_pred, 0)

                    for pred_label, original_label in inverse_mapping.items():
                        converted_pred[SR_pred == pred_label] = original_label

                    best_pred_pil = TF.to_pil_image(converted_pred.float().cpu())

                if JS / (length + 1) < min_JS:
                    min_JS = JS / (length + 1)
                    worst_img_pil = Image.open(image_path[0])
                    worst_gt_pil = Image.open(GT_path[0])

                    inverse_mapping = {0: 0, 1: 44, 2: 100, 3: 144}
                    converted_pred = torch.full_like(SR_pred, 0)
                    for pred_label, original_label in inverse_mapping.items():
                        converted_pred[SR_pred == pred_label] = original_label

                    worst_pred_pil = TF.to_pil_image(converted_pred.float().cpu())

                length += 1

        if best_img_pil and best_gt_pil and best_pred_pil:
            # show_images([best_img_pil, best_gt_pil, best_pred_pil], ['Input Image', 'Ground Truth', 'Best Prediction'])
            show_images([best_img_pil, best_pred_pil],
                        ['Input Image', 'Best Prediction'])
            print(f'Max JS: {max_JS:.4f}')

        if worst_img_pil and worst_gt_pil and worst_pred_pil:
            # show_images([worst_img_pil, worst_gt_pil, worst_pred_pil], ['Input Image', 'Ground Truth', 'Worst Prediction'])
            show_images([worst_img_pil, worst_pred_pil],
                        ['Input Image', 'Worst Prediction'])
            print(f'Min JS: {min_JS:.4f}')

        JS_flippers /= length
        JS_carapace /= length
        JS_head /= length
        JS /= length

        print(
            '[test] JS_flipper: %.4f, JS_carapace: %.4f, JS_head: %.4f, JS: %.4f' % (
                JS_flippers, JS_carapace, JS_head, JS))


