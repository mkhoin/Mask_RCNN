import numpy as np
from scipy import ndimage
from skimage.transform import resize

import re
import os
from random import SystemRandom

def parse_properties_from_path(path):
    # get base name without extension
    base_name = os.path.splitext(os.path.basename(path))[0]
    # parse properties
    property_array = re.split('-|_', base_name)
    properties = {
        'name' : property_array[0],
        'number' : property_array[1],
        'dimension' : re.split('x',re.search('[0-9]+[x][0-9]+[x][0-9]+',base_name).group(0)),
        'data_type' : property_array[-1] }
    return properties


def load_raw_image(path, endian_convert=False):
    properties = parse_properties_from_path(path)
    # read data
    data = np.fromfile(path, dtype=properties['data_type'])

    if endian_convert:
        data = data.byteswap()

    return data.reshape([int(i) for i in properties['dimension']])


def print_raw(image, data_type, path):
    with open(path, "wb") as file:
        file.write(image.astype(data_type).tobytes('Any'))


def write_raw(image, path, endian_convert=False):
    outFile = open(path, 'wb')
    if endian_convert:
        image = image.byteswap()
    outFile.write(image.tobytes())
    outFile.close()


def normalize(data, min, max):
    ptp = max - min
    nimage = (data - min) / ptp
    nimage = np.clip(nimage, 0, 1)

    return nimage.astype(np.float32)


def add_gaussian_noise(img):
    mean = 0
    var = 0.01

    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2]))
    noise_img = img + gaussian

    return noise_img


def get_gradient_image(data):
    sx = ndimage.filters.prewitt(data.astype(float), axis=0)
    sy = ndimage.filters.prewitt(data.astype(float), axis=1)
    sz = ndimage.filters.prewitt(data.astype(float), axis=2)
    return np.sqrt(sx**2 + sy**2 + sz**2).astype(data.dtype)


def get_edge_image(data):
    inside = np.empty_like(data)
    for z in range(data.shape[0]):
        inside[z] = ndimage.binary_erosion(data[z]).astype(data.dtype)
    return data - inside


def resize_image(data, is_binary, shape, binary_threshold=0.25):

    if not is_binary:
        data_type = data.dtype
        data = resize(data.astype(float), shape)
        return data.astype(data_type)
    else:
        data_type = data.dtype
        data = resize(data.astype(float), shape) >= binary_threshold
        return data.astype(data_type)


def get_random():
    crypto = SystemRandom()
    return crypto.random()


def cutout(data):
    data_type = data.dtype

    mask = np.ones((data.shape[0], data.shape[1], data.shape[2]), np.float32)

    n_holes = 1
    # if get_random() > 0.5:
    #     n_holes = 2

    # set range to width/5 ~ width/3
    len_plane = int(data.shape[2]/5) + int(get_random() * (data.shape[2]/4 - data.shape[2]/5))
    # set range to depth/5 ~ depth/3
    len_depth = int(data.shape[0]/5) + int(get_random() * (data.shape[0]/4 - data.shape[0]/5))

    for n in range(n_holes):
        # x = np.random.randint(data.shape[2])
        # y = np.random.randint(data.shape[1])
        # z = np.random.randint(data.shape[0])
        x = int(get_random() * data.shape[2])
        y = int(get_random() * data.shape[1])
        z = int(get_random() * data.shape[0])

        x1 = np.clip(x-len_plane//2, 0, data.shape[2])
        x2 = np.clip(x+len_plane//2, 0, data.shape[2])
        y1 = np.clip(y-len_plane//2, 0, data.shape[1])
        y2 = np.clip(y+len_plane//2, 0, data.shape[1])
        z1 = np.clip(z-len_depth//2, 0, data.shape[0])
        z2 = np.clip(z+len_depth//2, 0, data.shape[0])

        mask[z1:z2, y1:y2, x1:x2] = 0.

    data = data * mask

    return data.astype(data_type)

def get_class_ids(data):
    ids = []
    for z in range(data.shape[0]):
        if np.sum(data[z]) > 0:
            ids.append(1)
        else:
            ids.append(0)
    return np.array(ids)

def get_bbox(data):
    '''
    :param data:
    :return: gt_boxes: [depth, (y1, x1, y2, x2)]
    '''
    bbox = []
    for z in range(data.shape[0]):
        ind = np.where(data[z] != 0)
        if ind[0].size != 0:
            bbox_coord = [np.min(ind[0]), np.min(ind[1]), np.max(ind[0])+1, np.max(ind[1])+1]
        else:
            bbox_coord = [0, 0, 0, 0]
        bbox.append(bbox_coord)
    return np.array(bbox)

def build_rpn_targets(gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    config = Config()
    anchors = mask_utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                  config.RPN_ANCHOR_RATIOS,
                                                  config.BACKBONE_SHAPES,
                                                  config.BACKBONE_STRIDES,
                                                  config.RPN_ANCHOR_STRIDE)

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = mask_utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]

    rpn_match[(anchor_iou_max < 0.3)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    #print(np.sort(anchor_iou_max)[-4:])
    #print(sum(anchor_iou_max >= 0.7))

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h) if gt_h / a_h > 0 else 0,
            np.log(gt_w / a_w) if gt_w / a_w > 0 else 0,
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox