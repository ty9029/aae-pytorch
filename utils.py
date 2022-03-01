import numpy as np
import cv2
import matplotlib.pyplot as plt


def concat_image(images):
    b, h, w, c = images.shape
    num_side = int(np.sqrt(b))
    image = np.vstack([np.hstack(images[i * num_side:(i + 1) * num_side]) for i in range(num_side)])
    return image


def save_latent_variable(file_name, latent_values, labels):
    plt.figure(figsize=(8, 8))
    for i in set(labels):
        plt.scatter(x=latent_values[labels == i][:, 0], y=latent_values[labels == i][:, 1], label=str(i))
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend(loc="upper right")
    plt.savefig(file_name)
    plt.close()


def save_image(file_name, image):
    image = np.squeeze(image)
    image = np.array(image * 255.0, dtype="uint8")
    cv2.imwrite(file_name, image)


def save_images(dir_name, images):
    images = np.array(images * 255.0, dtype="uint8")
    for i, image in enumerate(images):
        cv2.imwrite("{}/{}.jpg".format(dir_name, i), image)
