import numpy as np
import json
import math
import os
import cv2
import torch
import torchvision.transforms as transforms

from typing import List

from triton_python_backend_utils import get_output_config_by_name, triton_string_to_numpy, get_input_config_by_name, get_input_tensor_by_name
from c_python_backend_utils import Tensor, InferenceResponse, InferenceRequest


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def normalize(self, img):
        img = self.toTensor(img) # not working in triton server
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'mask': 'mask',
            'image': 'image',
            'scale': 'scale',
        }
        self.output_names = {
            'bbox': 'bbox',
            'mbbox': 'mbbox',
            'textline': 'textline'
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')

        input_configs = {k: get_input_config_by_name(
            model_config, name) for k, name in self.input_names.items()}
        for k, cfg in input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        responses = []

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found ' f'in request {request.request_id()}')
                batch_in[k] = tensor.as_numpy()  # shape (batch_size, ...)

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            img_idx = 0
            map_image_boxes = {}
            for image, mask, scale_img in zip(batch_in['image'], batch_in['mask'], batch_in['scale']): # img is shape (1,)
                img_idx += 1
                image_h, image_w = image.shape[:2]
                org_img = cv2.resize(np.array(image), (int(image_w*scale_img[0]) ,int(image_h*scale_img[1])), interpolation = cv2.INTER_AREA)
                org_image_h, org_image_w = org_img.shape[:2]
                mask_h, mask_w = mask.shape[1:]
                scale_net = image_w/mask_w, image_h/mask_h
                img = mask[0]
                img = (img > 0.3) * 255
                kernel = np.ones((5, 5), np.uint8)
                img = cv2.dilate(np.array(img).astype("uint8"), kernel, iterations=1)
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                map_image_boxes[img_idx] = {"image": org_img, "bbox": []}
                n_boxes = 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x = int(x*scale_net[0]*scale_img[0])
                    y = int(y*scale_net[1]*scale_img[1])
                    w = int(w*scale_net[0]*scale_img[0])
                    h = int(h*scale_net[1]*scale_img[1])
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if w < 20 or h < 10: continue
                    x2 = x + w
                    y2 = y + h
                    if x2 > org_image_w: x2 = org_image_w
                    if y2 > org_image_h: y2 = org_image_h
                    n_boxes += 1
                    map_image_boxes[img_idx]["bbox"].append([x, y, x2-x, y2-y])
                    batch_out['bbox'].append([x, y, x2 - x, y2 - y])
                    batch_out['mbbox'].append(img_idx)
                if n_boxes == 0: # add dummy boxes for image which have no textline
                    batch_out['mbbox'].append(img_idx)
                    batch_out['bbox'].append([-1, -1, -1, -1])

            if len(batch_out['bbox']) > 0:
                max_w = max([math.ceil(bb[2]/bb[3])*64 for bb in batch_out['bbox']])
                normalizePAD = NormalizePAD((1, 64, max_w))
                for image_idx, image_box in map_image_boxes.items():
                    org_img = image_box["image"]
                    for bb in image_box["bbox"]:  # img is shape (1,)
                        x, y, w, h = bb
                        roi = org_img[y:y+h, x:x+w]
                        roi = cv2.resize(np.array(roi), (int(w*64/h) ,64), interpolation = cv2.INTER_AREA)
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) / 255.0
                        roi = normalizePAD.normalize(roi)
                        roi = np.array(roi).astype("float32")
                        batch_out['textline'].append(roi)
                    if len(image_box["bbox"]) == 0: # add dummy textline for image which have no textline
                        batch_out['textline'].append(np.zeros((1, 64, max_w)).astype("float32"))

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses