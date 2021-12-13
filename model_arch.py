import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from function import coral
import numpy as np
import math

# Control VGG-11 Model
class vgg11(nn.Module):
    def __init__(self, num_classes=20):
        super(vgg11, self).__init__()
        # List of potentially different rate of growth of receptive fields
        # assuming a center fixation.
        self.num_classes = num_classes
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3))
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3))
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3))
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3))
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3))
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3))
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3))
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3))
        self.fc1 = nn.Linear(51200, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

    def forward(self, vector):
        vector = self.pad(self.pool(F.relu(self.conv1_1(vector))))
        vector = self.pad(self.pool(F.relu(self.conv2_1(vector))))
        vector = self.pad(F.relu(self.conv3_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv3_2(vector))))
        vector = self.pad(F.relu(self.conv4_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv4_2(vector))))
        vector = self.pad(F.relu(self.conv5_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv5_2(vector))))
        vector = torch.flatten(vector, 1)
        vector = F.relu(self.fc1(vector))
        vector = F.relu(self.fc2(vector))
        vector = self.fc3(vector)
        return vector


# Foveatex Texture Transform Class
class vgg11_tex_fov(nn.Module):
    def __init__(self, scale, image_size, num_classes=20, permutation=None):
        super(vgg11_tex_fov, self).__init__()
        self.scale = scale
        self.perm = permutation
        self.num_classes = num_classes
        self.conv_size = 34
        self.image_size = image_size
        # define recpetive field scale parameters
        self.scale_in = ['0.25', '0.3', '0.4', '0.5', '0.6', '0.7']
        self.scale_out = [377, 301, 187, 126, 103, 91]
        self.Pooling_Region_Map = dict(zip(self.scale_in, self.scale_out))
        # load receptive fields
        self.mask_total, self.alpha_matrix = self.load_receptive_fields(self.scale, self.image_size, self.conv_size)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3))
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3))
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3))
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3))
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3))
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3))
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3))
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3))
        self.fc1 = nn.Linear(51200, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

    # function that loads receptive fields:
    def load_receptive_fields(self, scale, image_size, conv_layer_size):
        def mask_transform(size):
            transform = transforms.Compose([transforms.Resize(size), transforms.Grayscale(1), transforms.ToTensor()])
            return transform
        mask_tf = mask_transform(conv_layer_size)
        mask_regular_tf = mask_transform(image_size)
        d = 1.281  # a value that was fitted via psychophysical experiments assuming 26 deg of visual angle maps to 512 pixels on a screen.
        mask_total = torch.zeros(self.Pooling_Region_Map[scale], conv_layer_size, conv_layer_size)
        alpha_matrix = torch.zeros(self.Pooling_Region_Map[scale])
        for i in range(self.Pooling_Region_Map[scale]):
            i_str = str(i)
            # mask_str = './Receptive_Fields/MetaWindows_clean_s0.4/' + i_str + '.png'
            mask_str = 'Receptive_Fields/MetaWindows_clean_s' + scale + '/' + i_str + '.png'
            mask_temp = mask_tf(Image.open(str(mask_str)))
            mask_total[i, :, :] = mask_temp
            mask_regular = mask_regular_tf(Image.open(str(mask_str)))
            mask_size = torch.sum(torch.sum(mask_regular > 0.5))
            recep_size = np.sqrt(mask_size / 3.14) * 26.0 / 512.0
            if i == 0:
                alpha_matrix[i] = 0
            else:
                alpha_matrix[i] = -1 + 2.0 / (1.0 + math.exp(-recep_size * d))
        return mask_total, alpha_matrix

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def encoder(self, vector):
        vector = self.pad(self.pool(F.relu(self.conv1_1(vector))))
        vector = self.pad(self.pool(F.relu(self.conv2_1(vector))))
        vector = self.pad(F.relu(self.conv3_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv3_2(vector))))
        vector = self.pad(F.relu(self.conv4_1(vector)))
        return vector
    
    # during forward pass, the foveated texture transform (AdaIN) is applied to feature vectors after convolutional layer 4_1 in foveated receptive fields.
    def forward(self, content):
        content_f = self.encoder(content)
        noise = torch.randn(len(content), 3, self.image_size, self.image_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        noise = noise.to(device)
        for j in range(len(content)):
            noise[j, :, :, :] = coral(noise[j, :, :, :], content[j, :, :, :])
        noise_f = self.encoder(noise)
        # initialize empty foveated feature vector
        foveated_f = torch.zeros(np.shape(content_f)[0], np.shape(content_f)[1], self.conv_size, self.conv_size).to(device)
        # receptive field loop
        for i in range(self.Pooling_Region_Map[self.scale]):  # Loop over all the receptive fields (pooling regions)
            alpha_i = self.alpha_matrix[i]
            mask = self.mask_total[i, :, :]
            mask_binary = mask > 0.001
            content_f_mask = content_f[:, :, mask_binary[:, :]]  # 0 was 0th prefix before
            noise_f_mask = noise_f[:, :, mask_binary[:, :]]
            content_f_mask = content_f_mask.unsqueeze(3)
            noise_f_mask = noise_f_mask.unsqueeze(3)
            # Perform the Crowding Operation and Localized Auto Style-Transfer
            texture_f_mask = self.adaptive_instance_normalization(noise_f_mask, content_f_mask)
            if self.perm == 'random':
                diff_vec = texture_f_mask - content_f_mask
                idx = torch.randperm(diff_vec.nelement())
                diff_vec = diff_vec.view(-1)[idx].view(diff_vec.size())
                alpha_mixture = content_f_mask + alpha_i * diff_vec
            elif self.perm == 'fixed':
                diff_vec = texture_f_mask - content_f_mask
                torch.manual_seed(42)
                idx = torch.randperm(diff_vec.nelement())
                diff_vec = diff_vec.view(-1)[idx].view(diff_vec.size())
                alpha_mixture = content_f_mask + alpha_i * diff_vec
            else:
                alpha_mixture = (1 - alpha_i) * content_f_mask + alpha_i * texture_f_mask
            foveated_f[:, :, mask_binary[:, :]] = alpha_mixture.squeeze(3)
        vector = self.pad(self.pool(F.relu(self.conv4_2(foveated_f))))
        vector = self.pad(F.relu(self.conv5_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv5_2(vector))))
        vector = torch.flatten(vector, 1)
        vector = F.relu(self.fc1(vector))
        vector = F.relu(self.fc2(vector))
        vector = self.fc3(vector)
        return vector

# modified version of the control VGG-11 which returns the flattened feature vector after convolutional layer 4_1
class vgg11_modified(nn.Module):
    def __init__(self, num_classes=20):
        super(vgg11_modified, self).__init__()
        # List of potentially different rate of growth of receptive fields
        # assuming a center fixation.
        self.num_classes = num_classes
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3))
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3))
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3))
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3))
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3))

    def forward(self, vector):
        vector = self.pad(self.pool(F.relu(self.conv1_1(vector))))
        vector = self.pad(self.pool(F.relu(self.conv2_1(vector))))
        vector = self.pad(F.relu(self.conv3_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv3_2(vector))))
        vector = self.pad(F.relu(self.conv4_1(vector)))
        vector = torch.flatten(vector, 1)
        return vector

# modified version of the Foveated Texture Transform which returns the flattened feature vector after the transform is applied to the output of convolutional layer 4_1
class vgg11_tex_fov_modified(nn.Module):
    def __init__(self, scale, image_size, num_classes=20, permutation=None):
        super(vgg11_tex_fov_modified, self).__init__()
        # List of potentially different rate of growth of receptive fields
        # assuming a center fixation.
        self.scale = scale
        self.perm = permutation
        self.num_classes = num_classes
        self.conv_size = 34
        self.image_size = image_size
        # define recpetive field scale parameters
        self.scale_in = ['0.25', '0.3', '0.4', '0.5', '0.6', '0.7']
        self.scale_out = [377, 301, 187, 126, 103, 91]
        self.Pooling_Region_Map = dict(zip(self.scale_in, self.scale_out))
        # load receptive fields
        self.mask_total, self.alpha_matrix = self.load_receptive_fields(self.scale, self.image_size, self.conv_size)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3))
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3))
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3))
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3))
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3))

    # function that loads receptive fields:
    def load_receptive_fields(self, scale, image_size, conv_layer_size):
        def mask_transform(size):
            transform = transforms.Compose([transforms.Resize(size), transforms.Grayscale(1), transforms.ToTensor()])
            return transform
        mask_tf = mask_transform(conv_layer_size)
        mask_regular_tf = mask_transform(image_size)
        d = 1.281  # a value that was fitted via psychophysical experiments assuming 26 deg of visual angle maps to 512 pixels on a screen.
        mask_total = torch.zeros(self.Pooling_Region_Map[scale], conv_layer_size, conv_layer_size)
        alpha_matrix = torch.zeros(self.Pooling_Region_Map[scale])
        for i in range(self.Pooling_Region_Map[scale]):
            i_str = str(i)
            # mask_str = './Receptive_Fields/MetaWindows_clean_s0.4/' + i_str + '.png'
            mask_str = 'Receptive_Fields/MetaWindows_clean_s' + scale + '/' + i_str + '.png'
            mask_temp = mask_tf(Image.open(str(mask_str)))
            mask_total[i, :, :] = mask_temp
            mask_regular = mask_regular_tf(Image.open(str(mask_str)))
            mask_size = torch.sum(torch.sum(mask_regular > 0.5))
            recep_size = np.sqrt(mask_size / 3.14) * 26.0 / 512.0
            if i == 0:
                alpha_matrix[i] = 0
            else:
                alpha_matrix[i] = -1 + 2.0 / (1.0 + math.exp(-recep_size * d))
        return mask_total, alpha_matrix

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def encoder(self, vector):
        vector = self.pad(self.pool(F.relu(self.conv1_1(vector))))
        vector = self.pad(self.pool(F.relu(self.conv2_1(vector))))
        vector = self.pad(F.relu(self.conv3_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv3_2(vector))))
        vector = self.pad(F.relu(self.conv4_1(vector)))
        return vector

    def forward(self, content):
        content_f = self.encoder(content)
        noise = torch.randn(len(content), 3, self.image_size, self.image_size)
        device = content_f.get_device()
        noise = noise.to(device)
        for j in range(len(content)):
            noise[j, :, :, :] = coral(noise[j, :, :, :], content[j, :, :, :])
        noise_f = self.encoder(noise)
        # initialize empty foveated feature vector
        foveated_f = torch.zeros(np.shape(content_f)[0], np.shape(content_f)[1], self.conv_size, self.conv_size).to(device)
        # receptive field loop
        for i in range(self.Pooling_Region_Map[self.scale]):  # Loop over all the receptive fields (pooling regions)
            alpha_i = self.alpha_matrix[i]
            mask = self.mask_total[i, :, :]
            mask_binary = mask > 0.001
            content_f_mask = content_f[:, :, mask_binary[:, :]]  # 0 was 0th prefix before
            noise_f_mask = noise_f[:, :, mask_binary[:, :]]
            content_f_mask = content_f_mask.unsqueeze(3)
            noise_f_mask = noise_f_mask.unsqueeze(3)
            # Perform the Crowding Operation and Localized Auto Style-Transfer
            texture_f_mask = self.adaptive_instance_normalization(noise_f_mask, content_f_mask)
            if self.perm == 'random':
                diff_vec = texture_f_mask - content_f_mask
                idx = torch.randperm(diff_vec.nelement())
                diff_vec = diff_vec.view(-1)[idx].view(diff_vec.size())
                alpha_mixture = content_f_mask + alpha_i * diff_vec
            elif self.perm == 'fixed':
                diff_vec = texture_f_mask - content_f_mask
                torch.manual_seed(42)
                idx = torch.randperm(diff_vec.nelement())
                diff_vec = diff_vec.view(-1)[idx].view(diff_vec.size())
                alpha_mixture = content_f_mask + alpha_i * diff_vec
            else:
                alpha_mixture = (1 - alpha_i) * content_f_mask + alpha_i * texture_f_mask
            foveated_f[:, :, mask_binary[:, :]] = alpha_mixture.squeeze(3)
            vector = torch.flatten(foveated_f, 1)
        return vector
