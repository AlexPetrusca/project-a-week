## Weekly Projects

linear boundary
logical and
logical xor
quadratic boundary


1. **FNN Backpropagation:** 
Conducted experiments with simple feedforward neural networks, training them using 
backpropagation to learn both linear and non-linear decision boundaries. Tasks included 
solving problems such as logical AND, logical XOR, and identifying quadratic boundaries.
2. **MNIST:**
Built and trained a 5-layer feedforward neural network to classify handwritten digits 
from the MNIST dataset. Training was performed over 100 epochs with a learning rate of 7
and a batch size of 64. Achieved 96.7% accuracy.
3. **ML Framework - The Basics:** 
Started work on a machine learning framework with the goal of learning more about the
fundamental mechanisms behind backpropagation and deep learning. Implemented the linear 
(feedforward) layer and activation layer. Supported the sigmoid activation function and
Mean Square Error (MSE) loss function. Tested the framework on MNIST.
4. **ML Framework - Expanding on the Basics:** 
Added various activation functions, including Tanh, Relu, and Gelu, and various loss
functions, including Huber loss, Binary Cross Entropy loss, and Categorical Cross Entropy
loss. Added support for SGD with momentum and weight decay. Used the framework to train a
feedforward network on MNIST and FashionMNIST achieving 97% and 87% accuracy respectively.
5. **ML Framework - Convolutions:**
Implemented the MaxPool2D layer and Conv2D layer to support building convolutional neural
networks. Added "utility" layers to facilitate working with multidimensional tensors, 
such as the flatten layer, reshape layer, and transpose layer. Added "shim" layers to 
interop with MLX and PyTorch. Used the framework to train CNNs for MNIST, FashionMNIST,
and CIFAR10 achieving 99.4%, 92.7%, and 85.6% accuracy respectively.
6. **Micrograd:** 
Followed Andrej Karpathy's tutorial to build micrograd, a tiny, educational deep learning 
framework that closely resembles PyTorch.
7. **Makemore:** 
Followed Andrej Karpathy's makemore series to build a character‑level autoregressive
language model from scratch - going from simple bigrams to a feedforward neural language
model and culminating in implementing wavenet. The series also covered
topics like weight initialization, batch normalization, and best practices.
8. **GPT and TikToken:** 
Followed several of Andrej Karpathy's tutorials to build 1) a character-level transformer
language model based on GPT-1 and 2) a BPE tokenizer and token-level transformer language
model based on GPT-2. Reimplemented the PyTorch implementation of GPT-2 in MLX for 
performance testing. Briefly introduced to OpenAI's TikToken and Google's SentencePiece.
9. **CNN Feature Visualization:**
Experimented with visualizing classes learned by CNNs trained on CIFAR10 using gradient 
ascent. Using standard CNNs resulted in visualizations that were incoherent except for 
specific classes. ResNet visualizations captured the "outline" of the classes (e.g. 
outline of a bird). Added support for model checkpoints to my machine learning framework.
10. **DeepDream:**
Recreated and built my own versions of DeepDream. Experimented with extensions to gradient
ascent, like jitter, octaves, gradient normalization, and gaussian blurred gradients. 
Experimented with "controlled" dreams, a variant of DeepDream that maximizes features 
from an input image in the target image. Models used include VGG16, Resnet50, GoogLeNet,
and DenseNet.
11. **Principal Component Analysis:**
Experimented with applying Principal Component Analysis to 2D and 3D random data. Used PCA
to reduce the dimensionality of 3D datapoints to 2 dimensions.
12. **Vector Database:**

## Misc Projects

1. **Mephisto V2.0:** 
Added continuous analysis, multiple lines, support for all chess variants on Lichess, and 
support for the latest and greatest stockfish engines as well as an emscripten build of 
Leela Chess Zero and remote chess engines via a python script. 
2. **Cooking on the Fly:** 
Built a cooking simulation game in Unreal Engine for the 2025 thatgamecompany × COREBLAZER
Game Jam.
