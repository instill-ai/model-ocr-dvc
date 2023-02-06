from operator import le
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F

from typing import List

from triton_python_backend_utils import get_output_config_by_name, triton_string_to_numpy, get_input_config_by_name
from c_python_backend_utils import Tensor, InferenceResponse, InferenceRequest


def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

    
class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, separator_list = {}, dict_pathlist = {}):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

        self.separator_list = separator_list
        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep
        self.ignore_idx = [0] + [i+1 for i,item in enumerate(separator_char)]

        ####### latin dict
        if len(separator_list) == 0:
            dict_list = []
            for lang, dict_path in dict_pathlist.items():
                try:
                    with open(dict_path, "r", encoding = "utf-8-sig") as input_file:
                        word_count =  input_file.read().splitlines()
                    dict_list += word_count
                except:
                    pass
        else:
            dict_list = {}
            for lang, dict_path in dict_pathlist.items():
                with open(dict_path, "r", encoding = "utf-8-sig") as input_file:
                    word_count =  input_file.read().splitlines()
                dict_list[lang] = word_count

        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            # Returns a boolean array where true is when the value is not repeated
            a = np.insert(~((t[1:]==t[:-1])),0,True)
            # Returns a boolean array where true is when the value is not in the ignore_idx list
            b = ~np.isin(t,np.array(self.ignore_idx))
            # Combine the two boolean array
            c = a & b
            # Gets the corresponding character according to the saved indexes
            text = ''.join(np.array(self.character)[t[c.nonzero()]])
            texts.append(text)
            index += l
        return texts

class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'box': 'box',
            'mbox': 'mbox',
            'ocr': 'ocr',
        }
        self.output_names = {
            'box': 'box',
            'text': 'text',
            'score': 'score'
        }

        self.converter = CTCLabelConverter("0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", 
                                            {}, 
                                            {'en': './en.txt'})

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found ' f'in request {request.request_id()}')
                batch_in[k] = tensor.as_numpy()  # shape (batch_size, ...)

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            box = batch_in['box']
            mbox = batch_in['mbox']
            ocr = batch_in['ocr']

            # dummy zero box return in case of no text bounding box
            if len(ocr) > 0:
                preds_size = torch.IntTensor([ocr.shape[1]] * ocr.shape[0])
                preds_prob = F.softmax(torch.from_numpy(np.array(ocr)).float().to(device), dim=2)
                preds_prob = preds_prob.cpu().detach().numpy()
                pred_norm = preds_prob.sum(axis=2)
                preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)
                preds_prob = torch.from_numpy(preds_prob).float().to(device)
                _, preds_index = preds_prob.max(2)
                preds_index = preds_index.view(-1)
                preds_str = self.converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)

                preds_prob = preds_prob.cpu().detach().numpy()
                confidence_scores = []
                values = preds_prob.max(axis=2)
                indices = preds_prob.argmax(axis=2)
                preds_max_prob = []
                for v, i in zip(values, indices):
                    max_probs = v[i != 0]
                    if len(max_probs) > 0:
                        preds_max_prob.append(max_probs)
                    else:
                        preds_max_prob.append(np.array([0]))
                for pred_max_prob in preds_max_prob:
                    confidence_score = custom_mean(pred_max_prob)
                    confidence_scores.append(confidence_score)

                img_idx_prev = 0
                for i, img_idx in enumerate(mbox): # image idex start from 1
                    if img_idx != img_idx_prev:
                        batch_out['box'].append([box[i]])
                        batch_out['text'].append([preds_str[i]])
                        batch_out['score'].append([confidence_scores[i]])
                        img_idx_prev = img_idx
                        continue
                    batch_out['box'][-1].append(box[i])
                    batch_out['text'][-1].append(preds_str[i])
                    batch_out['score'][-1].append(confidence_scores[i])

                max_obj = max([len(b)  for b in batch_out['box']])
                # The output of all imgs must have the same size for Triton to be able to output a Tensor of type self.output_dtypes
                # Non-meaningful bounding boxes have coords [-1, -1, -1, -1] and text '' will be added and remove in model-backend
                for i, b in enumerate(batch_out['box']):
                    for _ in range(max_obj - len(b)):
                        batch_out['box'][i].append([-1, -1, -1, -1])
                        batch_out['text'][i].append("")
                        batch_out['score'][i].append(0)

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses