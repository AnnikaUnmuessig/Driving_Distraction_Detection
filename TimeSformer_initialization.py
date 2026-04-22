import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from PIL import Image
import numpy as np


model_name = "facebook/timesformer-base-finetuned-k400"
processor = AutoImageProcessor.from_pretrained(model_name)
model = TimesformerForVideoClassification.from_pretrained(model_name)