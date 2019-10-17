import random
import numpy as np

def rotate90(image_array: np.ndarray):
    return np.rot90(image_array, random.randint(0,3))

available_transforms = {
    'fliplr': np.fliplr,
    'flipud': np.flipud,
    'rotate90': rotate90,
}

no_transforms_to_applied = random.randint(0, len(available_transforms))

def transform(img):
    applied_transforms = 0
    while applied_transforms <= no_transforms_to_applied:
        key = random.choice(list(available_transforms))
        img = available_transforms[key](img)
        applied_transforms += 1
    return img