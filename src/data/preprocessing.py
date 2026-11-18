import cv2
import numpy as np

def decodePreprocessImg(image_bytes : bytes, max_size : int = 384) -> np.ndarray:
    # Decodes raw image bytes into a validated RGB NumPy array.
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8) # Converts raw bytes into 1D numpy arr

    bgr_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR) # decodes into a bgr image arr
    if bgr_image is None:
        raise ValueError("Invalid image bytes: failed to decode image.")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Resize while preserving aspect ratio if larger than max_size
    height, width = rgb_image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width*scale), int(height*scale))
        rgb_image = cv2.resize(rgb_image, new_size, interpolation=cv2.INTER_AREA)

    return rgb_image


if "__name__" == "__main__":
    file = open("testImg.png", "rb").read()

    print(type(file))

    image = decodePreprocessImg(file)
    print(image.shape)