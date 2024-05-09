import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import transform

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = Path(file_path)
        self.image_files = list(self.file_path.glob('*.npy'))
        self.labels = json.load(open(label_path))
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.current_index = 0
        self.epoch = 0

    def next(self):
        if self.current_index + self.batch_size > len(self.image_files):
            self.current_index = 0
            self.epoch += 1
            if self.shuffle:
                np.random.shuffle(self.image_files)

        batch_indices = range(self.current_index, min(self.current_index + self.batch_size, len(self.image_files)))
        self.current_index += self.batch_size

        batch_files = [self.image_files[i] for i in batch_indices]
        batch_images = np.array([np.load(f) for f in batch_files])
        batch_labels = np.array([int(self.labels[str(f.stem)]) for f in batch_files])

        if self.rotation:
            batch_images = np.array([self.rotate_image(img, np.random.choice([90, 180, 270])) for img in batch_images])

        if self.mirroring:
            batch_images = np.array(
                [self.mirror_image(img) if np.random.choice([True, False]) else img for img in batch_images])

        batch_images = np.array([self.resize_image(img, self.image_size) for img in batch_images])

        return batch_images, batch_labels


    def rotate_image(self, img, angle):
        return transform.rotate(img, angle)

    def mirror_image(self, img):
        return np.fliplr(img)

    def resize_image(self, img, size):
        return transform.resize(img, size, mode='reflect')

    # def augment(self,img):
    #     if self.rotation:
    #         angle = np.random.choice([90, 180, 270])
    #         img = skimage.transform.rotate(img, angle, mode='reflect')
    #
    #     if self.mirroring:
    #         if np.random.rand() > 0.5:
    #             img = np.fliplr(img)
    #     return img

    def current_epoch(self):
        return self.epoch

    def class_name(self, label):
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return classes[label]

    def show(self):
        batch_images, batch_labels = self.next()
        num_images = len(batch_images)
        num_cols = 2
        num_rows = num_images // num_cols + (num_images % num_cols > 0)
        for i, img in enumerate(batch_images):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.title(self.class_name(batch_labels[i]))
            plt.imshow(img)
        plt.savefig('batch_images.png')  # Save the figure before showing it
        plt.show()

# Define paths and parameters
file_path = "/Users/prashanthgadwala/Documents/Study material/Semester2/Deep learning/Exercise/exercise0_material/src_to_implement/exercise_data/"
label_path = "/Users/prashanthgadwala/Documents/Study material/Semester2/Deep learning/Exercise/exercise0_material/src_to_implement/Labels.json"
batch_size = 6
image_size = (36, 36)  # Specify the desired image size

# Create an instance of ImageGenerator
generator = ImageGenerator(file_path, label_path, batch_size, image_size, rotation=True, mirroring=True)

# Call the show() method to display images
generator.show()
