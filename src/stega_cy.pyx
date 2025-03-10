# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
from cython.operator cimport dereference as deref
from libc.math cimport log2
from libcpp cimport nullptr
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.memory cimport shared_ptr, make_shared

import time
import random
import torch
from scipy.stats import entropy
from tqdm import tqdm

from config import text_default_settings
from utils import get_probs_indices_past, set_seed, SingleExampleOutput

cdef bint msg_exhausted_flag = False

## Classes & Structures
# Nodes of Huffman tree 
cdef struct Node:
    double prob
    shared_ptr[Node] left
    shared_ptr[Node] right
    int index
    # >=0 - index
    # -1 - None
    int search_path
# 0  - this node
# -1 - in left subtree
# 1  - in right subtree
# 9  - unknown


cdef inline bint is_leaf(shared_ptr[Node] node_ptr):
    return deref(node_ptr).index != -1

# Sampling (Encoding) results and statistics for single time step
cdef struct CySingleEncodeStepOutput:
    int sampled_index
    int n_bits
    double entropy_t
    double kld
    double minimum_entropy_t

cdef class SingleEncodeStepOutput:
    cdef public:
        int sampled_index, n_bits
        double entropy_t, kld, minimum_entropy_t
    def __init__(self,
                 int sampled_index,
                 int n_bits,
                 double entropy_t,
                 double kld,
                 double minimum_entropy_t):
        self.sampled_index = sampled_index
        self.n_bits = n_bits
        self.entropy_t = entropy_t
        self.kld = kld
        self.minimum_entropy_t = minimum_entropy_t

    def __call__(self):
        return self.sampled_index, self.n_bits, self.entropy_t, self.kld, self.minimum_entropy_t

    def __str__(self):
        d = {
            'sampled_index': self.sampled_index,
            'n_bits': self.n_bits,
            'entropy_t': self.entropy_t,
            'kld': self.kld,
            'minimum_entropy_t': self.minimum_entropy_t
        }
        return '\n'.join('{} = {}'.format(key, value) for (key, value) in d.items())


# Decoding results for single time step
cdef struct CySingleDecodeStepOutput:
    string message_decoded_t

cdef class SingleDecodeStepOutput:
    cdef public:
        string message_decoded_t

    def __init__(self, string message_decoded_t) -> None:
        self.message_decoded_t = message_decoded_t

    def __call__(self):
        return self.message_decoded_t

## Utils
# Building a Huffman tree
cdef shared_ptr[Node] create_huffman_tree(list indices, list probs, int search_for):
    # Returns a pointer to the root node of the Huffman tree
    # if `search_for == -1`, we don't need to initialize the `search_path` of any Node object
    cdef:
        int sz = len(indices)
        int i, search_path
        double prob
        shared_ptr[Node] node_ptr, first, second, ans
        queue[shared_ptr[Node]] q1, q2

    for i in range(sz - 1, -1, -1):
        # search_path = 0 if search_for == indices[i] else 9
        if search_for == indices[i]:
            search_path = 0
        else:
            search_path = 9
        node_ptr = make_shared[Node](
            Node(probs[i], shared_ptr[Node](nullptr), shared_ptr[Node](nullptr), indices[i], search_path))
        q1.push(node_ptr)

    while q1.size() + q2.size() > 1:
        # first
        if not q1.empty() and not q2.empty() and deref(q1.front()).prob < deref(q2.front()).prob:
            first = q1.front()
            q1.pop()
        elif q1.empty():
            first = q2.front()
            q2.pop()
        elif q2.empty():
            first = q1.front()
            q1.pop()
        else:
            first = q2.front()
            q2.pop()

        # second
        if not q1.empty() and not q2.empty() and deref(q1.front()).prob < deref(q2.front()).prob:
            second = q1.front()
            q1.pop()
        elif q1.empty():
            second = q2.front()
            q2.pop()
        elif q2.empty():
            second = q1.front()
            q1.pop()
        else:
            second = q2.front()
            q2.pop()

        # merge
        prob = deref(first).prob + deref(second).prob
        search_path = 9
        if deref(first).search_path != 9:
            search_path = -1
        elif deref(second).search_path != 9:
            search_path = 1
        q2.push(make_shared[Node](Node(prob, first, second, -1, search_path)))

    if not q2.empty():
        ans = q2.front()
    else:
        ans = q1.front()
    return ans

## Steganography process - single time step
# Sampling (Encoding) - single time step
cdef CySingleEncodeStepOutput cy_encode_step(list indices, list probs, string message_bits):
    # Encode step
    global msg_exhausted_flag
    cdef:
        int sampled_index, n_bits = 0
        double entropy_t = 0.0, kld = 0.0, minimum_entropy_t = 0.0, prob_sum, ptr, ptr_0, ptr_1, partition
        shared_ptr[Node] node_ptr = create_huffman_tree(indices, probs, -1)
        vector[int] path_table = [-1, 1]
        int len_message_bits = len(message_bits)

    # if len_message_bits > 0:
    #     print('len(message_bits) = {}'.format(len_message_bits))
    while not is_leaf(node_ptr):  # non-leaf node
        prob_sum = deref(node_ptr).prob
        ptr = random.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum

        partition = deref(deref(node_ptr).left).prob

        # path_table[0] = -1 if (ptr_0 < partition) else 1
        if ptr_0 < partition:
            path_table[0] = -1
        else:
            path_table[0] = 1
        # path_table[1] = -1 if (ptr_1 < partition) else 1
        if ptr_1 < partition:
            path_table[1] = -1
        else:
            path_table[1] = 1

        # node_ptr = deref(node_ptr).right if path_table[message_bits[n_bits] - 48] == 1 else deref(node_ptr).left
        if not msg_exhausted_flag and (len_message_bits <= n_bits):
            print('[*] The message is exhausted and will be padded with all zeros!')
            msg_exhausted_flag = True
        # print(n_bits)
        if msg_exhausted_flag:
            if path_table[0] == 1:
                node_ptr = deref(node_ptr).right
            else:
                node_ptr = deref(node_ptr).left
        else:
            if path_table[message_bits[n_bits] - 48] == 1:
                node_ptr = deref(node_ptr).right
            else:
                node_ptr = deref(node_ptr).left

        if path_table[0] != path_table[1]:
            n_bits += 1
    # print(deref(node_ptr).index)
    sampled_index = deref(node_ptr).index
    minimum_entropy_t = -log2(probs[0])
    entropy_t = entropy(probs, base=2)
    return CySingleEncodeStepOutput(sampled_index, n_bits, entropy_t, kld, minimum_entropy_t)


# Decoding - single time step
cdef CySingleDecodeStepOutput cy_decode_step(list indices, list probs, int stego_t):
    # Decode step
    cdef:
        string message_decoded_t
        double prob_sum, ptr, ptr_0, ptr_1, partition
        shared_ptr[Node] node_ptr = create_huffman_tree(indices, probs, stego_t)
        vector[int] path_table = vector[int](2)
        map[int, string] path_table_swap

    while not is_leaf(node_ptr):  # non-leaf node
        prob_sum = deref(node_ptr).prob
        ptr = random.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum

        partition = deref(deref(node_ptr).left).prob

        # path_table[0] = -1 if (ptr_0 < partition) else 1
        if ptr_0 < partition:
            path_table[0] = -1
        else:
            path_table[0] = 1
        # path_table[1] = -1 if (ptr_1 < partition) else 1
        if ptr_1 < partition:
            path_table[1] = -1
        else:
            path_table[1] = 1

        if path_table[0] != path_table[1]:  # can embed 1 bit
            if deref(node_ptr).search_path == 9:  # fail to decode
                message_decoded_t = b'x'
                break

            if path_table[0] == -1:
                path_table_swap[-1] = b'0'
                path_table_swap[1] = b'1'
            else:
                path_table_swap[-1] = b'1'
                path_table_swap[1] = b'0'
            message_decoded_t += path_table_swap[deref(node_ptr).search_path]

            # walk
            if deref(node_ptr).search_path == -1:
                node_ptr = deref(node_ptr).left
            else:
                node_ptr = deref(node_ptr).right
        else:
            if path_table[0] == -1:
                node_ptr = deref(node_ptr).left
            else:
                node_ptr = deref(node_ptr).right

    if deref(node_ptr).search_path != 0:  # cannot reach a leaf node
        message_decoded_t = b'x'
    return CySingleDecodeStepOutput(message_decoded_t)


def encode(model, context, message_bits, settings, bint verbose = False, string tqdm_desc = b'Enc '):
    # Steganography Encoding (message_bits -> English text)
    cdef:
        int t = 0, length = settings.length, indices_idx
        string stego_object
        list generated_ids = []

        # CySingleEncodeStepOutput
        CySingleEncodeStepOutput single_encode_step_output
        int sampled_index
        int capacity_t
        double entropy_t
        double kld_step
        double minimum_entropy_t

        # statistics
        int total_capacity = 0
        double total_entropy = 0.0
        double total_minimum_entropy = 0.0
        double total_log_probs = 0.0  # for perplexity
        double total_kld = 0.0
        double max_kld = 0.0
        double perplexity, ave_kld

    set_seed(settings.seed)

    past = None  # pass into the `past_keys_values` for speed-up
    prev = context  # indices that were never passed to the model before

    start = time.time()
    for t in tqdm(range(length), desc=tqdm_desc, ncols=70):
        probs, indices, past = get_probs_indices_past(model, prev, past, settings)
        probs = probs.tolist()
        indices = indices.tolist()

        single_encode_step_output = cy_encode_step(indices, probs, message_bits)
        sampled_index = single_encode_step_output.sampled_index
        capacity_t = single_encode_step_output.n_bits
        entropy_t = single_encode_step_output.entropy_t
        kld_step = single_encode_step_output.kld
        minimum_entropy_t = single_encode_step_output.minimum_entropy_t

        indices_idx = indices.index(sampled_index)

        # update statistics
        total_entropy += entropy_t
        total_minimum_entropy += minimum_entropy_t
        total_log_probs += log2(probs[indices_idx])
        total_kld += kld_step
        if kld_step > max_kld:
            max_kld = kld_step

        # when `capacity_t == 0`, cannot embed message, but still needs to return a token_index
        if capacity_t > 0:
            total_capacity += capacity_t
            message_bits = message_bits[capacity_t:]  # remove the encoded part of `message_bits`
        generated_ids.append(sampled_index)
        prev = torch.tensor([sampled_index], device=settings.device).unsqueeze(0)
    end = time.time()

    perplexity = 2 ** (-1 / length * total_log_probs)
    ave_kld = total_kld / length
    return SingleExampleOutput(generated_ids, None, total_capacity, total_entropy, ave_kld, max_kld, perplexity,
                               end - start, settings,
                               total_minimum_entropy)

def decode(model, context, list stego, settings, bint verbose = False, string tqdm_desc = b'Dec '):
    # Steganography Decoding (English text -> message_bits)
    cdef:
        int t = 0, length = len(stego), indices_idx
        string message_decoded

        # CySingleEncodeStepOutput
        CySingleDecodeStepOutput single_decode_step_output
        int sampled_index
        int capacity_t
        double entropy_t
        double kld_step
        double minimum_entropy_t

    set_seed(settings.seed)
    past = None  # pass into the `past_keys_values` for speed-up
    prev = context  # indices that were never passed to the model before

    start = time.time()
    for t in tqdm(range(length), desc=tqdm_desc, ncols=70):
        probs, indices, past = get_probs_indices_past(model, prev, past, settings)
        probs = probs.tolist()
        indices = indices.tolist()

        single_decode_step_output = cy_decode_step(indices, probs, stego[t])
        message_decoded_t = single_decode_step_output.message_decoded_t

        if message_decoded_t == b'x':
            print('Fail to decode!')
            break
        message_decoded += message_decoded_t
        prev = torch.tensor([stego[t]], device=settings.device).unsqueeze(0)
    end = time.time()
    print('Decode time = {}s'.format(end - start))
    return message_decoded

def encode_text(model, tokenizer, message_bits, context, settings = text_default_settings):
    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(settings.device)

    single_encode_step_output = encode(model, context, message_bits, settings)
    single_encode_step_output.stego_object = tokenizer.decode(single_encode_step_output.generated_ids)

    return single_encode_step_output

def decode_text(model, tokenizer, string stego_object, context, settings = text_default_settings):
    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(settings.device)

    stego = tokenizer.encode(stego_object, return_tensors='pt').squeeze(0).tolist()
    message_decoded = decode(model, context, stego, settings)

    return message_decoded
