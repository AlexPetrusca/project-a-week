import os

import torch
from torchvision import transforms, models

import numpy as np
import cv2 as cv

DEVICE = torch.device("cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN / IMAGENET_STD).reshape(1, -1, 1, 1)).to(DEVICE)
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, -1, 1, 1)).to(DEVICE)

def read_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img

def read_image_tensor(img_path, target_shape=None):
    img = read_image(img_path, target_shape=target_shape)  # load numpy, [0, 1] image
    # Normalize image - VGG 16 and in general Pytorch (torchvision) models were trained like this,
    # so they learned to work with this particular distribution
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # Transform into PyTorch tensor, send to GPU and add dummy batch dimension. Models are expecting it, GPUs are
    # highly parallel computing machines so in general we'd like to process multiple images all at once
    # shape = (1, 3, H, W)
    img_tensor = transforms.ToTensor()(img).to(DEVICE).unsqueeze(0)
    img_tensor.requires_grad = True  # set this to true so that PyTorch will start calculating gradients for img_tensor
    return img_tensor

def write_image_tensor(img_path, img_tensor):
    # Send the PyTorch tensor back to CPU, detach it from the computational graph, convert to numpy
    # and make it channel last format again (calling ToTensor converted it to channel-first format)
    img = np.moveaxis(img_tensor.to('cpu').detach().numpy()[0], 0, 2)
    img = (img * IMAGENET_STD) + IMAGENET_MEAN  # de-normalize
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)

    cv.imwrite(img_path, img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...

def load_model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(DEVICE)  # Instantiate VGG 16 and send it to GPU
    for param in model.parameters():
        param.requires_grad = False

    print(dict(model.named_modules()))
    print()

    return model

def get_layer_activation(model, input, *layer_names):
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    layers = dict(model.named_modules())
    for layer_name in layer_names:
        layers[layer_name].register_forward_hook(get_activation(layer_name))

    model(input)

    return activations

def deep_dream_simple(img_path, dump_path):
    img_tensor = read_image_tensor(img_path, target_shape=500)
    model = load_model()

    # hyperparameters
    layers = ['features.29', 'features.11']
    n_iterations = 20
    learning_rate = 0.3

    for iter in range(n_iterations):
        # 1. grab layer activations
        layer_activations = get_layer_activation(model, img_tensor, *layers)

        # 2. define loss
        loss = 0
        for activation in layer_activations.values():
            loss += activation.mean()
        loss.backward()

        # 3. normalize the gradients
        img_tensor_grad = img_tensor.grad.data
        smooth_grads = img_tensor_grad / torch.std(img_tensor_grad)

        # 3. gradient ascent
        img_tensor.data += learning_rate * smooth_grads

        img_tensor.grad.data.zero_()  # clear the gradients otherwise they would get accumulated
        if iter % 10 == 0:
            print(f'Iteration {iter}, loss: {loss:.4f}')

    write_image_tensor(dump_path, img_tensor)
    print(f'Saved naive deep dream image to {os.path.relpath(dump_path)}')

def deep_dream_jitter(img_path, dump_path):
    img_tensor = read_image_tensor(img_path, target_shape=500)
    model = load_model()

    # hyperparameters
    layers = ['features.29', 'features.11']
    n_iterations = 20
    learning_rate = 0.3

    for iter in range(n_iterations):
        # 1. grab layer activations
        layer_activations = get_layer_activation(model, img_tensor, *layers)

        # 2. define loss
        loss = 0
        for activation in layer_activations.values():
            loss += activation.mean()
        loss.backward()

        # 3. normalize the gradients
        img_tensor_grad = img_tensor.grad.data
        smooth_grads = img_tensor_grad / torch.std(img_tensor_grad)

        # 4. gradient ascent
        img_tensor.data += learning_rate * smooth_grads  # gradient ascent

        img_tensor.grad.data.zero_()  # clear the gradients otherwise they would get accumulated
        if iter % 10 == 0:
            print(f'Iteration {iter}, loss: {loss:.4f}')

    write_image_tensor(dump_path, img_tensor)
    print(f'Saved naive deep dream image to {os.path.relpath(dump_path)}')


deep_dream_jitter("input_image.png", "output_image.png")