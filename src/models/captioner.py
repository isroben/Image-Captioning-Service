import numpy as np
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.data.preprocessing import decodePreprocessImg

class ImageCaptioner:
    def __init__(self, model_name:str, device:str = 'cpu', max_new_tokens : int = 30):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = max_new_tokens

        self.processor = BlipProcessor.from_pretrained(model_name) # Load processor (img norm + text decoding)

        self.model = BlipForConditionalGeneration.from_pretrained(model_name) # Load pretrained weights and move to target dev in evel mode
        self.model.to(self.device)
        self.model.eval()

    def predict(self, rgb_image : np.ndarray) -> str:
        """
        Generates an English caption from a preprocessed RGB numpy img.
        """

        # Convert Numpy arr -> normalized pytorch tensor on target dev
        inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)

        with torch.no_grad(): # run inference
            output_ids = self.model.generate(**inputs, max_new_tokens= self.max_new_tokens)

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True) # decode token id's back into clean string

        return caption.strip()


# file = open("testImg.png", "rb").read()

# image = decodePreprocessImg(file)

# captioner = ImageCaptioner(
#     model_name="Salesforce/blip-image-captioning-base",
#     device="cpu",
# )

# caption = captioner.predict(image)
# print("Generated Caption: ", caption)