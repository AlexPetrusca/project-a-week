import os

import torch
from torch import nn
from torch.nn.functional import threshold
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
    mean = IMAGENET_MEAN.reshape(1, 1, -1)
    std = IMAGENET_STD.reshape(1, 1, -1)
    img = (img * std) + mean  # de-normalize
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)

    cv.imwrite(img_path, img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...

def load_model(model_name="vgg16"):
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(DEVICE)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
    elif model_name == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2).to(DEVICE)
    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
    elif model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1).to(DEVICE)
    else:
        raise ValueError('Invalid model.')

    for param in model.parameters():
        param.requires_grad = False

    print([module for module in model.named_modules()])
    print()

    return model

def get_layer_activation(model, input, *layer_names):
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    layers = dict(model.named_modules())

    hooks = {}
    for layer_name in layer_names:
        # NOTE: these hook needs to be unregistered or else they stick around and run in the background
        hooks[layer_name] = layers[layer_name].register_forward_hook(get_activation(layer_name))

    model(input)

    for layer_name in layer_names:
        hooks[layer_name].remove()

    return activations

#-----------------------------------------------------------------------------------------------------------------------

def deep_dream(img_path, dump_path, model_name="vgg16", octaves=4, octave_scale=1.4, n_iterations=3,
               learning_rate=0.1, jitter=32, layers=None):
    img_tensor = read_image_tensor(img_path, target_shape=800)
    base_shape = img_tensor.shape[2:]  # save initial height and width
    model = load_model(model_name)

    # hyperparameters
    if layers is None:
        layers = ['features.26']

    # Adds stochasticity to the algorithm and makes the results more diverse
    def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
        if should_undo:
            h_shift = -h_shift
            w_shift = -w_shift
        with torch.no_grad():
            rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
            rolled.requires_grad = True
            return rolled

    # Get octave
    def get_new_shape(base_shape, octave_level):
        exponent = octave_level - octaves + 1
        new_shape = np.round(np.float32(base_shape) * (octave_scale ** exponent)).astype(np.int32)

        SHAPE_MARGIN = 10
        if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
            print(
                f'Pyramid size {octaves} with pyramid ratio {octave_scale} gives too small pyramid levels with size={new_shape}')
            print(f'Please change parameters.')
            exit(0)

        return new_shape

    # One step of deep dream
    def deep_dream_step(img_tensor, iter, layers, learning_rate, model):
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
        # smooth_grads = img_tensor_grad / img_tensor_grad.abs().mean()

        blur_fn = transforms.GaussianBlur(61)
        smooth_grads = blur_fn(smooth_grads)

        # 4. gradient ascent
        img_tensor.data += learning_rate * smooth_grads  # gradient ascent

        # 5. clamp the range
        img_tensor.data = torch.max(torch.min(img_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

        img_tensor.grad.data.zero_()  # clear the gradients otherwise they would get accumulated
        if iter % 10 == 0:
            print(f'Iteration {iter}, loss: {loss:.4f}')

    # Going from smaller to bigger resolution
    for octave in range(octaves):
        new_shape = get_new_shape(base_shape, octave)
        resize_transform = transforms.Resize(tuple(new_shape))
        img_tensor = resize_transform(img_tensor)

        for iter in range(n_iterations):
            h_shift, w_shift = np.random.randint(-jitter, jitter + 1, 2)
            img_tensor = random_circular_spatial_shift(img_tensor, h_shift, w_shift)

            deep_dream_step(img_tensor, iter, layers, learning_rate, model)

            img_tensor = random_circular_spatial_shift(img_tensor, h_shift, w_shift, should_undo=True)

    write_image_tensor(dump_path, img_tensor)
    print(f'Saved deep dream image to {os.path.relpath(dump_path)}')

def deep_dream_guided(img_path, guide_img_path, dump_path, model_name="vgg16", octaves=4, octave_scale=1.4,
                      n_iterations=3, learning_rate=0.1, jitter=32, layers=None, target_shape=800):
    img_tensor = read_image_tensor(img_path, target_shape=target_shape)
    base_shape = img_tensor.shape[2:]  # save initial height and width
    model = load_model(model_name)

    guide_img_tensor = read_image_tensor(guide_img_path, target_shape=target_shape // 4)
    guide_activations = get_layer_activation(model, guide_img_tensor, *layers)

    # Adds stochasticity to the algorithm and makes the results more diverse
    def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
        if should_undo:
            h_shift = -h_shift
            w_shift = -w_shift
        with torch.no_grad():
            rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
            rolled.requires_grad = True
            return rolled

    # Get octave
    def get_new_shape(base_shape, octave_level):
        exponent = octave_level - octaves + 1
        new_shape = np.round(np.float32(base_shape) * (octave_scale ** exponent)).astype(np.int32)

        SHAPE_MARGIN = 10
        if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
            print(
                f'Pyramid size {octaves} with pyramid ratio {octave_scale} gives too small pyramid levels with size={new_shape}')
            print(f'Please change parameters.')
            exit(0)

        return new_shape

    # One step of deep dream
    def deep_dream_step(img_tensor, iter, layers, learning_rate, model):
        # 1. grab layer activations
        layer_activations = get_layer_activation(model, img_tensor, *layers)

        # # 2. define loss
        # def _guided_loss(features, guide_features):
        #     """maximize top k feature maps"""
        #     b, ch, h, w = features.shape
        #     target_features = features.view(ch, -1) # (ch, h*w)
        #     guide_features = guide_features.view(ch, -1) # (ch, h*w)
        #
        #     channel_norms = guide_features.abs().sum(dim=1)
        #     top_k_values, top_k_indices = torch.topk(channel_norms, 10)
        #
        #     print(top_k_indices)
        #     activation = target_features[top_k_indices]
        #     # activation = target_features[top_k_indices].t() @ guide_features[top_k_indices]
        #     return activation.mean()

        # def _guided_loss(features, guide_features):
        #     """maximize dot product of guide_features and features"""
        #     b, ch, h, w = features.shape
        #     target_features = features.view(ch, -1) # (ch, h*w)
        #     guide_features = guide_features.view(ch, -1) # (ch, h*w)
        #
        #     affinities = torch.matmul(target_features.t(), guide_features)
        #     return affinities.mean()

        def _guided_loss(features, guide_features):
            """maximize dot product of masked guide_features and features"""
            b, ch, h, w = features.shape
            target_features = features.view(ch, -1) # (ch, h*w)
            guide_features = guide_features.view(ch, -1) # (ch, h*w)

            # unrelated features have no contribution
            mean = guide_features.mean()
            std = guide_features.std()
            threshold = mean + 1 * std
            guide_features = torch.where(guide_features > threshold, guide_features, 0)

            affinities = torch.matmul(target_features.t(), guide_features)
            return affinities.mean()

        def _guided_loss(features, guide_features):
            """maximize dot product of shifted guide_features and features"""
            b, ch, h, w = features.shape
            target_features = features.view(ch, -1) # (ch, h*w)
            guide_features = guide_features.view(ch, -1) # (ch, h*w)

            # unrelated features are penalized
            mean = guide_features.mean()
            std = guide_features.std()
            guide_features = guide_features - (mean + 1 * std)

            affinities = torch.matmul(target_features.t(), guide_features)
            return affinities.mean()

        loss = 0
        for name, activation in layer_activations.items():
            cur_loss = _guided_loss(activation, guide_activations[name].detach())
            cur_loss.backward()
            loss += cur_loss

        # 3. normalize the gradients
        img_tensor_grad = img_tensor.grad.data
        smooth_grads = img_tensor_grad / img_tensor_grad.abs().mean()

        blur_fn = transforms.GaussianBlur(31)
        smooth_grads = blur_fn(smooth_grads)

        # 4. gradient ascent
        img_tensor.data += learning_rate * smooth_grads  # gradient ascent

        # 5. clamp the range
        img_tensor.data = torch.max(torch.min(img_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

        img_tensor.grad.data.zero_()  # clear the gradients otherwise they would get accumulated
        if iter % 10 == 0:
            print(f'Iteration {iter}, loss: {loss:.4f}')

    # Going from smaller to bigger resolution
    for octave in range(octaves):
        new_shape = get_new_shape(base_shape, octave)
        resize_transform = transforms.Resize(tuple(new_shape))
        img_tensor = resize_transform(img_tensor)

        for iter in range(n_iterations):
            h_shift, w_shift = np.random.randint(-jitter, jitter + 1, 2)
            img_tensor = random_circular_spatial_shift(img_tensor, h_shift, w_shift)

            deep_dream_step(img_tensor, iter, layers, learning_rate, model)

            img_tensor = random_circular_spatial_shift(img_tensor, h_shift, w_shift, should_undo=True)

    write_image_tensor(dump_path, img_tensor)
    print(f'Saved deep dream image to {os.path.relpath(dump_path)}')

#-----------------------------------------------------------------------------------------------------------------------

# deep_dream("in/starry_night.png", "out/test.png",
#            model_name="vgg16", octaves=4, octave_scale=1.8, n_iterations=15,
#            learning_rate=0.05, layers=['features.26'])

deep_dream_guided("in/starry_night.png", "in/grass.png", "out/test.png",
                  model_name="vgg16", octaves=4, octave_scale=1.4, n_iterations=10,
                  learning_rate=0.1, layers=['features.26'], target_shape=800)
