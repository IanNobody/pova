#
#   POVa Project
#	Modified AlexNet Architecture
#	Author: Šimon Strýček <xstryc06@vutbr.cz>
#

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