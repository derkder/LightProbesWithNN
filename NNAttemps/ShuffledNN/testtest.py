import OpenEXR
import Imath
import numpy as np
import ctypes

def should_return_true(x, y, seed):
    total_calls = 1920 * 1080
    max_true_count = 10000

    h = (ctypes.c_int32(x * 374761393).value +
         ctypes.c_int32(y * 668265263).value +
         ctypes.c_int32(seed * 1013904223).value)
    h = ctypes.c_int32(h).value
    h = ctypes.c_int32((h ^ (h >> 13)) * 1274126177).value
    h = ctypes.c_int32(h).value
    h = ctypes.c_int32(h ^ (h >> 16)).value
    random_value = abs(h) % total_calls

    return random_value < max_true_count

def read_exr(filename):
    file = OpenEXR.InputFile(filename)
    dw = file.header()['dataWindow']
    size = (1920, 1080)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb_channels = ['R', 'G', 'B']
    channels = [np.frombuffer(file.channel(c, pt), dtype=np.float32) for c in rgb_channels]
    image = np.zeros((size[1], size[0], 3), dtype=np.float32)
    print(image.shape)
    for i, channel in enumerate(channels):
        image[:, :, i] = np.reshape(channel, (size[1], size[0]))

    return image

def process_image(image):
    height, width, _ = image.shape
    seed = 1577140874

    for y in range(height):
        for x in range(width):
            if should_return_true(x, y, seed):
                # print(f"pixel at ({x}, {y})")
                pixel = image[y, x]
                if not np.allclose(pixel, [1.0, 1.0, 1.0], atol=1e-5):  # Check if the pixel is not white
                    print(f"Non-white pixel")

# Example usage
image = read_exr("C:/Files/CGProject/NNLightProbes/dumped_data/temptemp/frame_0001/Mogwai.AccumulatePass.output.1000.exr")
# print(should_return_true(15, 4, 0))
# print(image[4, 15])
process_image(image)
