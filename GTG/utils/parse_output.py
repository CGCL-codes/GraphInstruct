import re


LEFT_MARK = '<<<'
RIGHT_MARK = '>>>'


def parse_wrapped_answer(s):
    left_mark = LEFT_MARK
    right_mark = RIGHT_MARK

    # s = s[s.find('>>') + 2:]

    left_start = s.find(left_mark)
    p_left = left_start + len(left_mark)

    right_start = s.find(right_mark)
    p_right = right_start

    if p_right > p_left:
        target = s[p_left:p_right]
    else:
        target = None
    return target


# def parse_wrapped_answer(s):
#     left_mark = LEFT_MARK
#     right_mark = RIGHT_MARK

#     left_start = s.find(left_mark)
#     p_left = left_start + len(left_mark)

#     right_start = s.find(right_mark)
#     p_right = right_start

#     if p_right > p_left:
#         target = s[p_left:p_right]
#     else:
#         target = None
#     return target


def parse_bool(s):
    s = s.split(' ')
    has_yes = ('yes' in s) | ('Yes' in s) | ('YES' in s) | ('1' in s) | ('True' in s) | ('true' in s) | ('TRUE' in s)
    has_no = ('no' in s) | ('No' in s) | ('NO' in s) | ('0' in s) | ('False' in s) | ('false' in s) | ('FALSE' in s)
    if has_yes and not has_no:
        return 'Yes'
    elif not has_yes and has_no:
        return 'No'
    else:
        return None


def parse_first_node_id(s):
    start_index = s.find("<")
    end_index = s.find(">")
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return s[start_index:end_index+1]
    else:
        return None


def remove_all_node_id(s):
    pattern = r"<[^>]*>"
    result = re.sub(pattern, " ", s)
    return result


def parse_first_integer(s):
    matches = re.findall(r'\d+', s)
    if matches:
        return int(matches[0])
    else:
        return None


def parse_first_float(string):
    pattern = r'(\d*\.\d+|\d+\/\d+)'
    match = re.search(pattern, string)
    if match:
        number = match.group(0)
        if '/' in number:
            numerator, denominator = number.split('/')
            return float(numerator) / float(denominator)
        else:
            return float(number)
    else:
        return None


def get_nodes_set(sample):
    if 'nodes' in sample:
        return get_nodes_set_from_nodes_str(sample['nodes'])
    else:
        num_nodes = int(sample['num_nodes'])
        return set([str(u) for u in range(num_nodes)])


def get_nodes_set_from_nodes_str(s):
    for m in '[],':
        s = s.replace(m, ' ')
    return set([u for u in s.split(' ') if len(u) > 0])


def parse_node_list(s, sample=None):
    for m in '\'\"()[]{},->':
        s = s.replace(m, ' ')
    L = [u for u in s.split(' ') if len(u) > 0]
    if sample is not None:
        # check if node id is valid
        nodes_set = get_nodes_set(sample)
        for u in L:
            if u not in nodes_set:
                return None
    return L


def parse_list(s):
    start = s.find('[')
    end = s.find(']')
    if start == -1 or end == -1:
        return None
    list_str = s[start+1:end]
    elements = list_str.split(',')
    parsed_list = [element.strip(" '") for element in elements if element.strip() != '']
    return parsed_list


def parse_dict(s):
    start = s.find('{')
    end = s.find('}')
    if start == -1 or end == -1:
        return None
    
    parsed_dict={}
    
    end1=s.find('[')
    while end1 != -1:
        _key=s[start+1:end1-2].find('<')+start+1
        key=s[_key:end1-2]
        end2=s[start+1:end].find(']')+start+1
        a=s[end2]
        value=parse_list(s[end1:end2+1])
        parsed_dict[key]=value
        start=end2+1
        end1=s[start:end+1].find('[')
        if end1 != -1:
            end1+=start
   
    return parsed_dict


def parse_list_with_weight(s):
    start = s.find('[')
    end = s.find(']')
    if start == -1 or end == -1:
        return None
    parsed_list=[]
    
    end1=s.find(')')
    while end1!=-1:
        key1=s[start:end1].find('<')+start
        key2=s[start:end1].find('>')+start
        key=s[key1:key2+1]
        weight1=s[start:end1].find(':')+start
        weight=int(s[weight1+1:end1])
        parsed_list.append((key,weight))
        start=end1+1
        end1=s[start:end].find(')')
        if end1!=-1:
            end1=s[start:end].find(')')+start
        
    return parsed_list


def parse_dict_with_weight(s):
    start = s.find('{')
    end = s.find('}')
    if start == -1 or end == -1:
        return None
    
    parsed_dict={}
    
    end1=s.find('[')
    while end1 != -1:
        _key=s[start+1:end1-2].find('<')+start+1
        key=s[_key:end1-2]
        end2=s[start+1:end].find(']')+start+1
        a=s[end2]
        value=parse_list_with_weight(s[end1:end2+1])
        parsed_dict[key]=value
        start=end2+1
        end1=s[start:end+1].find('[')
        if end1 != -1:
            end1+=start
    return parsed_dict


def add_quotation_mark(s):
    return s.replace('<', "'<").replace('>', ">'")


def parse_edge_list(s, sample=None):
    L = parse_node_list(s, sample)
    if L is None:
        return None
    edge_list = []
    for i in range(len(L) // 2):
        edge_list.append((L[2 * i], L[2 * i + 1]))
    return edge_list
