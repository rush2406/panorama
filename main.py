import cv2
import numpy as np
from images import Image
from typing import List
import os
from matching import MultiImageMatches, PairMatch, build_homographies, find_connected_components
from rendering import multi_band_blending, set_gain_compensations, simple_blending

# read images and transform them to grayscale
image1 = Image('/home/rushali/CV702_Assignment/Project/image pairs_03_02.jpg')
image2 = Image('/home/rushali/CV702_Assignment/Project/image pairs_03_01.jpg')

images = [image1,image2]

print('("Computing features with SIFT...")')

for image in images:
	image.compute_features()

matcher = MultiImageMatches(images)
pair_matches = matcher.get_pair_matches()

build_homographies([images], pair_matches)

print('("Computing gain compensations...")')

set_gain_compensations(
        images,
        pair_matches,
        sigma_n=10,
        sigma_g=0.1,
    )

for image in images:
    image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

result = simple_blending(images)

os.makedirs(os.path.join("./", "results"), exist_ok=True)
cv2.imwrite(os.path.join("./", "results", f"pano_{i}.jpg"), result)