# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
from cython.operator cimport dereference as deref
from libcpp cimport nullptr, bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.memory cimport shared_ptr, make_shared

import random
import torch
from scipy.stats import entropy
from tqdm import tqdm
from utils import get_logits, get_probs_indices, set_seed, SingleExampleOutput

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
    bool msg_exhausted_flag

cdef class SingleEncodeStepOutput:
    cdef public:
        int sampled_index, n_bits
        double entropy_t
    def __init__(self,
                 int sampled_index,
                 int n_bits,
                 double entropy_t):
        self.sampled_index = sampled_index
        self.n_bits = n_bits
        self.entropy_t = entropy_t

    def __call__(self):
        return self.sampled_index, self.n_bits, self.entropy_t

    def __str__(self):
        d = {
            'sampled_index': self.sampled_index,
            'n_bits': self.n_bits,
            'entropy_t': self.entropy_t,
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
    cdef:
        int sampled_index, n_bits = 0
        double entropy_t = 0.0, prob_sum, ptr, ptr_0, ptr_1, partition
        shared_ptr[Node] node_ptr = create_huffman_tree(indices, probs, -1)
        vector[int] path_table = [-1, 1]
        int len_message_bits = len(message_bits)
        bool msg_exhausted_flag = False

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
    entropy_t = entropy(probs, base=2)
    return CySingleEncodeStepOutput(sampled_index, n_bits, entropy_t, msg_exhausted_flag)


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
        string stego_object
        list generated_ids = []
        int estimated_msg_length = (len(message_bits) // 2)  # Estimate message length

        # CySingleEncodeStepOutput
        CySingleEncodeStepOutput single_encode_step_output
        int sampled_index
        int capacity_t
        double entropy_t

        list forbidden_tokens_ids = [settings.eos_token_id]

        # statistics
        int total_capacity = 0
        double total_entropy = 0.0

    set_seed(settings.seed)

    # Create progress bar with estimated message length instead of token positions
    progress_bar = tqdm(total=estimated_msg_length, desc=tqdm_desc, ncols=70, unit='bits')

    # empty list
    while True:
        logits = get_logits(model, context, settings)

        second_last_char = context[:, -1]
        if second_last_char in [settings.eos_token_id, settings.dot_token_id, settings.dash_token_id]:
            forbidden_tokens_ids = [settings.eos_token_id, settings.dot_token_id, settings.dash_token_id]
        else:
            forbidden_tokens_ids = [settings.eos_token_id]


        probs, indices = get_probs_indices(logits, settings, forbidden_tokens_ids)

        probs = probs.tolist()
        indices = indices.tolist()

        single_encode_step_output = cy_encode_step(indices, probs, message_bits)
        sampled_index = single_encode_step_output.sampled_index
        capacity_t = single_encode_step_output.n_bits
        entropy_t = single_encode_step_output.entropy_t

        # update statistics
        total_entropy += entropy_t

        # when `capacity_t == 0`, cannot embed message, but still needs to return a token_index
        if capacity_t > 0:
            total_capacity += capacity_t
            message_bits = message_bits[capacity_t:]  # remove the encoded part of `message_bits`
        generated_ids.append(sampled_index)

        # Update progress bar
        progress_bar.update(capacity_t)

        if single_encode_step_output.msg_exhausted_flag and not sampled_index == settings.dot_token_id:
            break

        context = torch.cat((context, torch.tensor([[sampled_index]], device=settings.device)), dim=1)

    # Close progress bar
    progress_bar.close()

    return SingleExampleOutput(generated_ids, total_capacity, total_entropy, settings)

def decode(model, context, int stego_start_index, int stego_end_index, settings, bint verbose = False, string tqdm_desc = b'Dec '):
    # Steganography Decoding (English text -> message_bits)
    cdef:
        int t = 0, indices_idx
        string message_decoded = b''  # Initialize message_decoded
        int estimated_msg_length = (stego_end_index - stego_start_index + 1) * 3  # Estimate message length

        # CySingleEncodeStepOutput
        CySingleDecodeStepOutput single_decode_step_output
        
        list forbidden_tokens_ids = [settings.eos_token_id]

    set_seed(settings.seed)
    
    # Create progress bar with estimated message length instead of token positions
    progress_bar = tqdm(total=estimated_msg_length, desc=tqdm_desc, ncols=70, unit='bits')
    
    for t in range(stego_start_index, stego_end_index+1):
        current_char = context[0, t]
        current_context = context[:, :t]
        logits = get_logits(model, current_context, settings)

        last_char = context[:, t-1]
        if last_char in [settings.eos_token_id, settings.dot_token_id, settings.dash_token_id]:
            forbidden_tokens_ids = [settings.eos_token_id, settings.dot_token_id, settings.dash_token_id]
        else:
            forbidden_tokens_ids = [settings.eos_token_id]

        probs, indices = get_probs_indices(logits, settings, forbidden_tokens_ids)
        probs = probs.tolist()
        indices = indices.tolist()

        single_decode_step_output = cy_decode_step(indices, probs, current_char)
        message_decoded_t = single_decode_step_output.message_decoded_t

        if message_decoded_t == b'x':
            print('Fail to decode!')
            break
            
        message_decoded = message_decoded.append(message_decoded_t)

        # Update progress bar
        progress_bar.update(message_decoded_t.length())

    # Close progress bar
    progress_bar.close()

    # remove padding
    message_decoded.resize(message_decoded.length() - message_decoded.length() % 8)

    return message_decoded
