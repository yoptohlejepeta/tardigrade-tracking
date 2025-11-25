import imageio.v3 as iio
import matplotlib.pyplot as plt


first_image = iio.imread("data/FC1.0.mkv", index=0)
# crop sides

plt.axis('off')
plt.imshow(first_image)
