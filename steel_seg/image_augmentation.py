import random

import numpy as np
import matplotlib


def uint8_to_float32(img):
    assert img.dtype == np.uint8
    cast = img.astype(np.float32)
    scale = 1.0 / np.iinfo(img.dtype).max
    return cast * scale

def float32_to_uint8(img):
    assert img.dtype == np.float32
    scale = np.iinfo(np.uint8).max + 0.5  # avoid rounding problems in the cast
    scaled = img * scale
    return scaled.astype(np.uint8)

def adjust_brightness_and_contrast(img, brightness_delta, contrast_factor):
    dtype = img.dtype
    if dtype == np.uint8:
        img_flt = uint8_to_float32(img)
    elif dtype == np.float32:
        img_flt = img.copy()
    else:
        raise TypeError('Unsupported data type to adjust_brightness.')

    img_flt *= contrast_factor
    img_flt += brightness_delta
    img_flt[img_flt > 1.0] = 1.0
    img_flt[img_flt < 0.0] = 0.0

    if dtype == np.uint8:
        return float32_to_uint8(img_flt)
    return img_flt


# def augment_rgb_image(img, hue_delta, saturation_delta, brightness_delta, contrast_factor):
#     # RGB uint8 image -> HSV float32 image
#     rgb_flt_img = uint8_to_float32(img)
#     hsv_flt_img = matplotlib.colors.rgb_to_hsv(rgb_flt_img)

#     # Adjust hue:
#     hsv_flt_img += [hue_delta, 0.0, 0.0]

#     # Adjust saturation:
#     hsv_flt_img += [0.0, saturation_delta, 0.0]

#     hsv_flt_img[hsv_flt_img > 1.0] = 1.0
#     hsv_flt_img[hsv_flt_img < 0.0] = 0.0

#     # HSV float32 image -> RGB float32 image
#     rgb_flt_img = matplotlib.colors.hsv_to_rgb(hsv_flt_img)

#     # Adjust contrast
#     rgb_flt_img *= contrast_factor

#     # Adjust brightness
#     rgb_flt_img += [brightness_delta, brightness_delta, brightness_delta]
#     rgb_flt_img[rgb_flt_img > 1.0] = 1.0
#     rgb_flt_img[rgb_flt_img < 0.0] = 0.0

#     # RGB float32 image -> RGB uint8 image
#     rgb_int_img = float32_to_uint8(rgb_flt_img)
#     return rgb_int_img

# def random_augment(img):
#     return augment_rgb_image(img,
#                          random.uniform(-0.5, 0.5),   # hue_delta
#                          random.uniform(-0.03, 0.03), # saturation_delta
#                          random.uniform(-0.2, 0.2),   # brightness_delta
#                          random.uniform(0.5, 1.5))    # contrast_factor