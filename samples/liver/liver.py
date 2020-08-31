"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os, glob
import sys
import datetime
import numpy as np
import skimage.draw
import scipy.ndimage as ni

# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/cglab/Desktop/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn_.config import Config
import mrcnn_.model as modellib
import mrcnn_.utils as utils

import samples.liver.image_processing as iu
import samples.liver.matrix_processing as mu

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class LiverConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    pass


############################################################
#  Dataset
############################################################

class LiverDataset(utils.Dataset):

    def load_liver(self, root_dir_image, root_dir_label, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("liver", 1, "liver")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]

        self.width = 128
        self.height = 128
        self.depth = 64

        # configurations of data set.
        self.image_paths = glob.glob(root_dir_image + '/*')
        self.label_paths = glob.glob(root_dir_label + '/*')

        # Add images
        for image_path, label_path in zip(self.image_paths, self.label_paths):

            if subset is "train":
                aug = (iu.get_random() > 0.5)
                mat1 = mu.generate_random_rotation_around_axis((1, 0, 0), 20)
                mat2 = mu.generate_random_shear(5)
                mat = mat1 * mat2
            else:
                aug = None
                mat = None

            image = iu.load_raw_image(image_path)

            self.add_image(
                "liver",
                image_id=image_path,  # use file name as a unique image id
                path=image_path,
                width=image.shape[2], height=image.shape[1],
                depth=image.shape[0],
                aug=aug, mat=mat,
                label_path=label_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "liver":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count] => [height, width, depth]
        label = iu.load_raw_image(info["label_path"])
        label = label.astype(np.uint8)

        label = iu.resize_image(label, True, [self.depth, self.height, self.width], binary_threshold=0.5)

        if info["aug"] is not None:
            label = ni.affine_transform(label, matrix=info["mat"]) > 0.5
            label = label.astype(np.uint8)

        label = np.expand_dims(label, axis=3)

        class_ids = np.ones(shape=(label.shape[0], 1)).astype(np.int32)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return label.astype(np.bool), class_ids # [64, 128, 128, 1], [64, 1]

    def load_image(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "liver":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, 3] => [height, width, depth]
        image = iu.load_raw_image(info["path"])

        image = iu.resize_image(image, False, [self.depth, self.height, self.width])

        if info["aug"] is not None:
            image = ni.affine_transform(image, matrix=info["mat"], cval=-1024.0)

        image = iu.normalize(image, -340, 360)
        image = image.transpose(1, 2, 0)

        image = np.expand_dims(image, axis=0)
        input2d = image[:, :, :, 0:2]
        single = image[:, :, :, 0:1]
        input2d = np.concatenate([input2d, single], axis=3)
        for i in range(self.depth - 2):
            input2d_tmp = image[:, :, :, i:i + 3]
            input2d = np.concatenate([input2d, input2d_tmp], axis=0)
            if i == self.depth - 3:
                final1 = image[:, :, :, self.depth - 2:self.depth]
                final2 = image[:, :, :, self.depth - 1:self.depth]
                final = np.concatenate([final1, final2], axis=3)
                input2d = np.concatenate([input2d, final], axis=0)
        image = input2d[:, :, :, :]

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return image # [64, 128, 128, 3]

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "liver":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LiverDataset()
    dataset_train.load_liver(os.path.join(args.dataset, "train", "image"),
                             os.path.join(args.dataset, "train", "label"),
                             "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LiverDataset()
    dataset_val.load_liver(os.path.join(args.dataset, "val", "image"),
                           os.path.join(args.dataset, "val", "label"),
                           "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='all')


def test(model, dst_path):
    dataset_test = LiverDataset()
    dataset_test.load_liver(os.path.join(args.testdataset, "image"),
                            os.path.join(args.testdataset, "label"),
                            "test")
    dataset_test.prepare()

    image_ids = dataset_test.image_ids

    for image_id in image_ids:
        info = dataset_test.image_info[image_id]
        print("Processing {} images".format(info["path"].split("\\")[-1]))
        print((info["depth"], info["height"], info["width"]))

        image = dataset_test.load_image(image_id)
        out_label = model.detect(image, verbose=0)

        out_label = iu.resize_image(out_label, True, (info["depth"], info["height"], info["width"]))
        iu.write_raw(out_label.astype(np.uint8), os.path.join(dst_path, '%s' % info["label_path"].split("\\")[-1]))

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset',
                        default="D:/chungmy/PyTorch/CENet-SR-AutoNet/database/nfold_90") # nfold_50
    parser.add_argument('--testdataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset',
                        default="D:/chungmy/PyTorch/CENet-SR-AutoNet/database/nfold_test_80")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default="C:/Users/cglab/Desktop/Mask_RCNN/results",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--results', required=False,
                        default="C:/Users/cglab/Desktop/Mask_RCNN/results",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=results/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.testdataset, "Argument --testdataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LiverConfig()
    else:
        class InferenceConfig(LiverConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 64
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        test(model, dst_path=args.results)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
