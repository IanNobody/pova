"""
Before AlexNet, SIFT(scale-invariant feature transform), SURF or HOG were some of the hand tuned feature extractors for Computer Vision.

In AlexNet, Interestingly in the lowest layers of the network, the model learned feature extractors that resembled some traditional filters.
Higher layers in the network might build upon these representations to represent larger structures, like eyes, noses, blades of grass, and so on.
Even higher layers might represent whole objects like people, airplanes, dogs, or frisbees. Ultimately, the final hidden state learns a compact
representation of the image that summarizes its contents such that data belonging to different categories can be easily separated.

Challenges perceived before AlexNet:

Computational Power:

Due to the limited memory in early GPUs, the original AlexNet used a dual data stream design, so that each of their two GPUs could be responsible
for storing and computing only its half of the model. Fortunately, GPU memory is comparatively abundant now, so we rarely need to break up models
across GPUs these days.

Data Availability:

ImageNet was released during this period by researchers under Fei-Fei Li with 1 million images, 1000 images per class with total of 1000 class.

Note:

Instead of using ImageNet, I am using MNIST and resizing the image to 224 x 224 dimension to make it justify with the AlexNet architecture.
"""
import torch.nn as nn
from copy import deepcopy

class ModifiedAlexNet(nn.Module):
	def __init__(self, input_channel, alexnet, fc):
		super().__init__()
		self.conv1_p = nn.Sequential(
			# transforming (bsize x 1 x 224 x 224) to (bsize x 96 x 54 x 54)
			#From floor((n_h - k_s + p + s)/s), floor((224 - 11 + 3 + 4) / 4) => floor(219/4) => floor(55.5) => 55
			nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=3), #(batch_size * 96 * 55 * 55)
			nn.ReLU(inplace=True), #(batch_size * 96 * 55 * 55)
			#nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
			nn.MaxPool2d(kernel_size=3, stride=2)) #(batch_size * 96 * 27 * 27)

		self.conv1 = deepcopy(alexnet.conv1)
		self.conv2 = deepcopy(alexnet.conv2)
		self.conv3 = deepcopy(alexnet.conv3)
		self.fc = deepcopy(fc)

		self.conv1_p.apply(self.init_weights)
		self.fc.apply(self.init_weights)

	def init_weights(self, layer):
		if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
			nn.init.xavier_uniform_(layer.weight)

	def forward(self, x, bg):
		out = self.conv1(x)

		if bg is not None:
			mask = self.conv1_p(bg)
			out = out - mask

		out = self.conv2(out)
		out = self.conv3(out)
		out = self.fc(out)

		return out

# #GPU Usage for NVIDIA: nvidia-smi --loop=1