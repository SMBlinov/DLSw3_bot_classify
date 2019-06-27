from PIL import Image as PIL_Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from fastai.vision import load_learner, Image

# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
class ClassPredictor:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = load_learner('../model/','densenet161.pkl')
		self.to_tensor = transforms.ToTensor()

	def predict(self, img_stream):
		return self.model.predict(self.process_image(img_stream))

	def process_image(self, img_stream):
		# используем PIL, чтобы получить картинку из потока и изменить размер
		image = PIL_Image.open(img_stream).resize((300, 300))
		# переводим картинку в тензор и оборачиваем в объект Image, который использует fastai у себя внутри
		image = Image(self.to_tensor(image))
		return image
