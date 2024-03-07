from GTG.utils.parse_output import get_nodes_set_from_nodes_str
from GTG.utils.parse_output import remove_all_node_id, parse_bool, parse_first_integer
from GTG.utils.parse_output import parse_first_float, parse_node_list
from GTG.utils.parse_output import parse_wrapped_answer, parse_first_node_id, parse_list, parse_dict
import networkx as nx
import numpy as np


class BaseEvaluator:

    def __init__(self):
        pass
    
    def parse_output_ans_str(self, sample, output_ans_str):
        # success: return int/float/list/...
        # fail: return None
        raise NotImplementError
    
    def check_correctness(self, sample, output_ans):
        # correct: return 1, wrong: return 0
        raise NotImplementError

    def eval_a_sample(self, sample):
        '''
            1. parse_wrapped -> ouput_ans_str
            2. parse_output_ans_str -> int/float/list/... output_ans
            3. check correctness -> 1/0
        '''
        output_ans_str = parse_wrapped_answer(str(sample['output']))
        if output_ans_str is None:
            sample['output_ans_str'] = None
            sample['output_ans'] = None
            sample['correct'] = 0
            return sample
        else:
            sample['output_ans_str'] = output_ans_str
            output_ans = self.parse_output_ans_str(
                sample, output_ans_str
            )
            if output_ans is None:
                sample['output_ans'] = None
                sample['correct'] = 0
                return sample
            else:
                sample['output_ans'] = output_ans
                correct = self.check_correctness(
                    sample, output_ans
                )
                if correct:
                    sample['correct'] = 1
                else:
                    sample['correct'] = 0
                return sample


class BoolEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        output_ans = parse_bool(output_ans_str)
        return output_ans
    
    def check_correctness(self, sample, output_ans):
        return output_ans == sample['answer']


class IntEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        output_ans = parse_first_integer(output_ans_str)
        return output_ans
    
    def check_correctness(self, sample, output_ans):
        return output_ans == sample['answer']

    
class FloatEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        output_ans = parse_first_float(output_ans_str.replace(' ', ''))
        return output_ans
    
    def check_correctness(self, sample, output_ans):
        ans = sample['answer']
        error_ratio = 0.03
        return abs(output_ans - ans) / (ans + 1e-8) < error_ratio


class NodeEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        nodes_set = get_nodes_set(sample)
        output_ans = output_ans_str.strip()
        if output_ans in nodes_set:
            return output_ans
        else:
            return None
    
    def check_correctness(self, sample, output_ans):
        return output_ans == str(sample['answer'])


class NodeListEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        output_ans = parse_node_list(output_ans_str, sample)
        # ensure valid node id in output_ans
        return output_ans

    def check_correctness(self, sample, output_ans):
        raise NotImplementError


class NodeSetEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        output_ans = parse_node_list(output_ans_str, sample)
        # ensure valid node id in output_ans
        return output_ans

    def check_correctness(self, sample, output_ans):
        answer = parse_node_list(str(sample['answer']))
        return set(output_ans) == set(answer)


def eval_bool(sample):
    s = sample['output']
    has_yes = ('yes' in s) | ('Yes' in s) | ('YES' in s)
    if has_yes:
        sample['parsed_output'] = 'Yes'
    else:
        sample['parsed_output'] = 'No'
    
    if sample['answer'] == sample['parsed_output']:
        sample['correct'] = 1
    else:
        sample['correct'] = 0
    
    return sample


def eval_integer(sample):
    parse_success = False
    s = remove_all_node_id(sample['output'])
    x = parse_first_integer(s)
    if x is not None:
        parse_success = True
        output_ans = x

    if parse_success:
        sample['parsed_output'] = str(output_ans)
        if output_ans == sample['answer']:
            sample['correct'] = 1
        else:
            sample['correct'] = 0
    else:
        sample['parsed_output'] = 'None'
        sample['correct'] = 0

    return sample


def eval_float(sample):
    parse_success = False
    s = remove_all_node_id(sample['output'])
    x = parse_first_float(s)
    if x is not None:
        parse_success = True
        output_ans = x

    if parse_success:
        sample['parsed_output'] = str(output_ans)
        ans = sample['answer']
        if abs(output_ans - ans) / (ans + 1e-6) < 0.05:
            sample['correct'] = 1
        else:
            sample['correct'] = 0
    else:
        sample['parsed_output'] = 'None'
        sample['correct'] = 0

    return sample


def eval_node_in_list(sample):
    parse_success = False
    x = parse_first_node_id(sample['output'])
    if x is not None:
        parse_success = True
        output_ans = x

    if parse_success:
        sample['parsed_output'] = output_ans
        ans = sample['answer']
        if output_ans in ans:
            sample['correct'] = 1
        else:
            sample['correct'] = 0
    else:
        sample['parsed_output'] = 'None'
        sample['correct'] = 0
    # print(sample['correct'])

    return sample


def eval_a_node(sample):
    parse_success = False
    x = parse_first_node_id(sample['output'])
    if x is not None:
        parse_success = True
        output_ans = x

    if parse_success:
        sample['parsed_output'] = output_ans
        ans = sample['answer']
        if output_ans == ans:
            sample['correct'] = 1
        else:
            sample['correct'] = 0
    else:
        sample['parsed_output'] = 'None'
        sample['correct'] = 0

    return sample


def eval_ad_location(sample):
    parse_success = False
    x = parse_first_node_id(sample['output'])
    if x is not None:
        parse_success = True
        output_ans = x

    if parse_success:
        sample['parsed_output'] = output_ans
        ans = sample['answer']
        if output_ans in ans:
            sample['correct'] = 1
        else:
            sample['correct'] = 0
    else:
        sample['parsed_output'] = 'None'
        sample['correct'] = 0


def eval_node_set(sample):
    parse_success = False
    x = parse_list(sample['output'])
    if x is not None:
        parse_success = True
        output_ans = x

    if parse_success:
        sample['parsed_output'] = str(output_ans)
        ans = parse_list(sample['answer'])
        if set(output_ans) == set(ans):
            sample['correct'] = 1
        else:
            sample['correct'] = 0
    else:
        sample['parsed_output'] = 'None'
        sample['correct'] = 0

    return sample
