import os
import json
import cv2
#from keras.models import load_model
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from scipy.special import expit

#########################################################
from OcrImages import ocr_images

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


#from app import infer_model_gas,infer_model_iesco,infer_model_ptcl

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
#infer_model_gas = load_model("modelgas1.h5")
#infer_model_iesco = load_model("iescomodel.h5")
#infer_model_ptcl = load_model("ptclmodel.h5")


from PreProcessImage import split

def get_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    code originally from https://github.com/fizyr/keras-retinanet/
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
    """
    if label < len(colors):
        return colors[label]
    else:
        print('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)

colors = [
    [31  , 0   , 255] ,
    [0   , 159 , 255] ,
    [255 , 95  , 0]   ,
    [255 , 19  , 0]   ,
    [255 , 0   , 0]   ,
    [255 , 38  , 0]   ,
    [0   , 255 , 25]  ,
    [255 , 0   , 133] ,
    [255 , 172 , 0]   ,
    [108 , 0   , 255] ,
    [0   , 82  , 255] ,
    [0   , 255 , 6]   ,
    [255 , 0   , 152] ,
    [223 , 0   , 255] ,
    [12  , 0   , 255] ,
    [0   , 255 , 178] ,
    [108 , 255 , 0]   ,
    [184 , 0   , 255] ,
    [255 , 0   , 76]  ,
    [146 , 255 , 0]   ,
    [51  , 0   , 255] ,
    [0   , 197 , 255] ,
    [255 , 248 , 0]   ,
    [255 , 0   , 19]  ,
    [255 , 0   , 38]  ,
    [89  , 255 , 0]   ,
    [127 , 255 , 0]   ,
    [255 , 153 , 0]   ,
    [0   , 255 , 255] ,
    [0   , 255 , 216] ,
    [0   , 255 , 121] ,
    [255 , 0   , 248] ,
    [70  , 0   , 255] ,
    [0   , 255 , 159] ,
    [0   , 216 , 255] ,
    [0   , 6   , 255] ,
    [0   , 63  , 255] ,
    [31  , 255 , 0]   ,
    [255 , 57  , 0]   ,
    [255 , 0   , 210] ,
    [0   , 255 , 102] ,
    [242 , 255 , 0]   ,
    [255 , 191 , 0]   ,
    [0   , 255 , 63]  ,
    [255 , 0   , 95]  ,
    [146 , 0   , 255] ,
    [184 , 255 , 0]   ,
    [255 , 114 , 0]   ,
    [0   , 255 , 235] ,
    [255 , 229 , 0]   ,
    [0   , 178 , 255] ,
    [255 , 0   , 114] ,
    [255 , 0   , 57]  ,
    [0   , 140 , 255] ,
    [0   , 121 , 255] ,
    [12  , 255 , 0]   ,
    [255 , 210 , 0]   ,
    [0   , 255 , 44]  ,
    [165 , 255 , 0]   ,
    [0   , 25  , 255] ,
    [0   , 255 , 140] ,
    [0   , 101 , 255] ,
    [0   , 255 , 82]  ,
    [223 , 255 , 0]   ,
    [242 , 0   , 255] ,
    [89  , 0   , 255] ,
    [165 , 0   , 255] ,
    [70  , 255 , 0]   ,
    [255 , 0   , 172] ,
    [255 , 76  , 0]   ,
    [203 , 255 , 0]   ,
    [204 , 0   , 255] ,
    [255 , 0   , 229] ,
    [255 , 133 , 0]   ,
    [127 , 0   , 255] ,
    [0   , 235 , 255] ,
    [0   , 255 , 197] ,
    [255 , 0   , 191] ,
    [0   , 44  , 255] ,
    [50  , 255 , 0]
]


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

###########################################################################
def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (int(new_w), int(new_h)))
    np.reshape(resized, (resized.shape[0], resized.shape[1], resized.shape[2]))
    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    print(new_image.shape)
    sizer = new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2),
            int((net_w - new_w) // 2):int((net_w + new_w) // 2), :]
    if sizer.shape[1] > resized.shape[1] and sizer.shape[0] == resized.shape[0]:
        new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2),
        int((net_w - new_w) // 2):int((net_w + new_w) // 2) - 1, :] = resized
    elif sizer.shape[1] == resized.shape[1] and sizer.shape[0] == resized.shape[0]:
        new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2),
        int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized

    elif sizer.shape[0] > resized.shape[0] and sizer.shape[1] == resized.shape[1]:
        new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2) - 1,
        int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized
    elif sizer.shape[1] > resized.shape[1] and sizer.shape[0] > resized.shape[0]:
        new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2) - 1,
        int((net_w - new_w) // 2):int((net_w + new_w) // 2) - 1, :] = resized

    new_image = np.expand_dims(new_image, 0)

    return new_image

####################################################################################
def decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            # objectness = netout[..., :4]

            if (objectness.all() <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            # box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)
            box.xmin = (x - (w / 2))
            box.ymin = (y - (h / 2))
            box.xmax = (x + (w / 2))
            box.ymax = (y + (h / 2))
            boxes.append(box)

    return boxes

########################################################################
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

#################################################################################
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

##################################################################################
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def draw_boxes(output_path, image, boxes, labels, obj_thresh, quiet=True):
    st = 0
    imgCount=0
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin - 3, box.ymin],
                               [box.xmin - 3, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin]], dtype='int32')

            ####temp
            if ("ptcl_grandtotal" in label_str ):
                cropped = image[box.ymin-20:box.ymax+3, box.xmin-30:box.xmax+3]
            elif ("ptcl_phonenumber" in label_str):
                cropped = image[box.ymin-50:box.ymax+3, box.xmin-30:box.xmax+3]
            elif ("ptcl_address" in label_str):
                cropped = image[box.ymin:box.ymax, box.xmin-30:box.xmax ]
            else:
                if box.xmin < 0:
                    box.xmin=0
                cropped = image[box.ymin:box.ymax, box.xmin:box.xmax]
            ####
            #cropped = image[box.ymin:box.ymax, box.xmin:box.xmax]
            print(st)
            if(cropped.size>0):
                cv2.imwrite(output_path + "cropped_" + label_str + ".png", np.uint8(cropped))
                if ("gData" in label_str and imgCount==0):
                    split(output_path + "cropped_" + label_str + ".png")
                    imgCount += 1

            cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(label),
                          thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin + 13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(0, 0, 0),
                        thickness=2)

    return image

#######################################################################################
def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = images[0].shape
    nb_images = len(images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

        # run the prediction
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        batch_output = model.predict_on_batch(batch_input)

    batch_boxes = [None] * nb_images

    for i in range(nb_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2 - j) * 6:(3 - j) * 6]  # config['model']['anchors']
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, 0.45, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        batch_boxes[i] = boxes

    return batch_boxes


##########################################################################
def executePredictYolo(model):

    #print("executeEntering")
    if model =="0":
        print("DETECTING GAS")
        config_path = "gasconfig1.json"
        #infer_model = infer_model_gas

    elif model == "1":
        config_path = "iescoconfig.json"
        #infer_model = infer_model_iesco

    else:
        config_path = "ptcldata.json"
        #infer_model = load_model("ptclmodel.h5")
        #infer_model = infer_model_ptcl
    input_path = "ptcl_test_image/"
    #input_path = testImage
    output_path = "Outputs/"

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']


    ###############################
    #   Predict bounding boxes
    ###############################
    if 'webcam' in input_path:  # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)

        # the main loop
        batch_size = 1
        images = []
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]

            if (len(images) == batch_size) or (ret_val == False and len(images) > 0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh,
                                             nms_thresh)

                for i in range(len(images)):
                    draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)
                    cv2.imshow('video with bboxes', images[i])
                images = []
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            cv2.destroyAllWindows()
    elif input_path[-4:] == '.mp4':  # do detection on a video
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       50.0,
                                       (frame_w, frame_h))
        # the main loop
        batch_size = 1
        images = []
        start_point = 0  # %
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i + 1) / nb_frames) > start_point / 100.:
                images += [image]

                if (i % batch_size == 0) or (i == (nb_frames - 1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh,
                                                 nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)

                        # show the video with detection bounding boxes
                        if show_window: cv2.imshow('video with bboxes', images[i])

                        # write result to the output video
                        video_writer.write(images[i])
                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()
    else:  # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path):
            for inp_file in os.listdir(input_path):
                print(1)
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.JPG', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[
                0]

            # draw bounding boxes on the image using labels
            draw_boxes(output_path, image, boxes, config['model']['labels'], obj_thresh)

            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))

    textOcr=ocr_images()
    return textOcr

###############################################################################
def executePredictYoloSingleImage():
    config_path = "ptclconf.json"
    input_image = cv2.imread('androidFlask.jpg',0)
    cv2.imshow("input",input_image)
    cv2.waitKey()
    output_path = "Outputs/"

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################
     # do detection on an image or a set of images

    # the main loop

    # predict the bounding boxes
    boxes = get_yolo_boxes(infer_model, [input_image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[
        0]

    # draw bounding boxes on the image using labels
    draw_boxes(output_path, input_image, boxes, config['model']['labels'], obj_thresh)

    # write the image with bounding boxes to file
    cv2.imwrite(output_path + input_image.split('/')[-1], np.uint8(input_image))

    ocr_images()
