import numpy as np
from tatqa_utils import to_number, is_number
import re
from enum import IntEnum

# "SPAN-TABLE-TEXT": 2 has no data
OPERATOR_CLASSES_ = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "CHANGE_RATIO": 3,
                    "AVERAGE": 4, "COUNT": 5, "SUM": 6, "DIFF": 7, "TIMES": 8, "DIVIDE": 9}

def get_op_1(op_mode):
    if op_mode==1:
        return {"SPAN-TEXT": 0}
    elif op_mode==2:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1}
    elif op_mode==3:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2}
    elif op_mode==4:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "SUM": 3}
    elif op_mode==5:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "SUM": 3, "COUNT": 4}
    elif op_mode==6:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3, "SUM": 4,
                "AVERAGE": 5}
    elif op_mode==7:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3, "SUM": 4,
                "AVERAGE": 5, "TIMES": 6}
    elif op_mode==8:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3, "SUM": 4,
                "AVERAGE": 5, "TIMES": 6, "DIVIDE": 7}
    else:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3, "SUM": 4,
                "AVERAGE": 5, "TIMES": 6, "DIVIDE": 7, "DIFF": 8}

def get_op_2(op_mode):
    if op_mode==1:
        return {"SPAN-TABLE": 0, "MULTI_SPAN": 1, "COUNT": 2, "SUM": 3,
                "AVERAGE": 4, "TIMES": 5, "DIVIDE": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==2:
        return {"SPAN-TEXT": 0, "MULTI_SPAN": 1, "COUNT": 2, "SUM": 3,
                "AVERAGE": 4, "TIMES": 5, "DIVIDE": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==3:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "COUNT": 2, "SUM": 3,
                "AVERAGE": 4, "TIMES": 5, "DIVIDE": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==4:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "SUM": 3,
                "AVERAGE": 4, "TIMES": 5, "DIVIDE": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==5:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3,
                "AVERAGE": 4, "TIMES": 5, "DIVIDE": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==6:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3,
                "SUM": 4, "TIMES": 5, "DIVIDE": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==7:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3,
                "SUM": 4, "AVERAGE": 5, "DIVIDE": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==8:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3,
                "SUM": 4, "AVERAGE": 5, "TIMES": 6, "DIFF": 7, "CHANGE_RATIO": 8}
    elif op_mode==9:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3,
                "SUM": 4, "AVERAGE": 5, "TIMES": 6, "DIVIDE": 7, "CHANGE_RATIO": 8}
    else:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3,
                "SUM": 4, "AVERAGE": 5, "TIMES": 6, "DIFF": 7, "DIVIDE": 8}

def get_op_3(op_mode):
    if op_mode == 1:
        return {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3}
    else:
        return {"SUM": 0, "AVERAGE": 1, "TIMES": 2, "DIVIDE": 3, "DIFF": 4, "CHANGE_RATIO": 5}

def get_arithmetic_op_index_1(op_mode):
    if op_mode==1:
        return []
    elif op_mode==2:
        return []
    elif op_mode==3:
        return []
    elif op_mode==4:
        return []
    elif op_mode==5:
        return [4]
    elif op_mode==6:
        return[4, 5]
    elif op_mode==7:
        return [4, 5, 6]
    elif op_mode==8:
        return [4, 5, 6, 7]
    else:
        return [4, 5, 6, 7, 8]

def get_arithmetic_op_index_2(op_mode):
    if op_mode==1:
        return [3, 4, 5, 6, 7, 8]
    elif op_mode==2:
        return [3, 4, 5, 6, 7, 8]
    elif op_mode==3:
        return [3, 4, 5, 6, 7, 8]
    elif op_mode==4:
        return [3, 4, 5, 6, 7, 8]
    elif op_mode==5:
        return [4, 5, 6, 7, 8]
    elif op_mode==6:
        return [4, 5, 6, 7, 8]
    elif op_mode==7:
        return [4, 5, 6, 7, 8]
    elif op_mode==8:
        return [4, 5, 6, 7, 8]
    elif op_mode==9:
        return [4, 5, 6, 7, 8]
    else:
        return [4, 5, 6, 7, 8]

def get_arithmetic_op_index_3(op_mode):
    if op_mode == 1:
        return []
    else:
        return [0, 1, 2, 3, 4, 5]

OPERATOR = ['+', '-', '*', '/']

SCALE = ["", "thousand", "million", "billion", "percent"]

def get_operators(derivation:str):
    res = []
    for c in derivation:
        if c in OPERATOR:
            res.append(c)
    return res

def get_operands(derivation):
    num_strs = re.split('\+|-|\*|/', derivation)
    result = []
    for it in num_strs:
        one = to_number(it)
        if one is not None:
            result.append(one)
    return result

def facts_to_nums(facts):
    return [to_number(f) for f in facts]

def _is_average(num_facts:list, answer):
    return round(np.average(num_facts), 2) == round(answer, 2)

def _is_change_ratio(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        ori_percent = round(100 * (num_facts[0] - num_facts[1]) / num_facts[1], 2)
        cands.append(ori_percent)
    if num_facts[0] != 0:
        ori_percent = round(100 * (num_facts[1] - num_facts[0]) / num_facts[0], 2)
        cands.append(ori_percent)
    return round(answer, 2) in cands

def _is_division(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        cands.append(round(num_facts[0]/num_facts[1], 2))
        cands.append(100 * round(num_facts[0]/num_facts[1], 2))
    if num_facts[0] != 0:
        cands.append(round(num_facts[1]/num_facts[0], 2))
        cands.append(100 * round(num_facts[1]/num_facts[0], 2))
    return round(answer, 2) in cands

def _is_diff(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    ans_1 = round(num_facts[0] - num_facts[1], 2)
    ans_2 = round(num_facts[1] - num_facts[0], 2)
    return round(answer, 2) in (ans_1, ans_2)

def _is_sum(num_facts:list, answer):
    return round(np.sum(num_facts), 2) == round(answer, 2)

def _is_times(num_facts:list, answer):
    return round(np.prod(num_facts), 2) == round(answer, 2)

def get_operator_class(derivation:str, answer_type:str, facts:list, answer, mapping:dict, scale, OPERATOR_CLASSES):
    operator_class = None
    try:
        if answer_type == "span":
            if "table" in mapping:
                operator_class = OPERATOR_CLASSES["SPAN-TABLE"]
            else:
                operator_class = OPERATOR_CLASSES["SPAN-TEXT"]
        elif answer_type == "multi-span":
            operator_class = OPERATOR_CLASSES["MULTI_SPAN"]
        elif answer_type == "count":
            operator_class = OPERATOR_CLASSES["COUNT"]
        elif answer_type == "arithmetic":
            num_facts = facts_to_nums(facts)
            if not is_number(str(answer)):
                return None  # not support date
            if _is_change_ratio(num_facts, answer):
                operator_class = OPERATOR_CLASSES["CHANGE_RATIO"]
            elif _is_average(num_facts, answer):
                operator_class = OPERATOR_CLASSES["AVERAGE"]
            elif _is_sum(num_facts, answer):
                operator_class = OPERATOR_CLASSES["SUM"]
            elif _is_times(num_facts, answer):
                operator_class = OPERATOR_CLASSES["TIMES"]
            elif _is_diff(num_facts, answer):
                operator_class = OPERATOR_CLASSES["DIFF"]
            elif _is_division(num_facts, answer):
                operator_class = OPERATOR_CLASSES["DIVIDE"]

            operators = get_operators(derivation)
            if len(operators) == 1: # if it is detected that only have one operator, use the one in the derivation
                if operators[0] == "/":
                    return OPERATOR_CLASSES["DIVIDE"]
                elif operators[0] == "-":
                    operator_class = OPERATOR_CLASSES["DIFF"]
                elif operators[0] == "*":
                    operator_class = OPERATOR_CLASSES["TIMES"]
                elif operators[0] == "+":
                    operator_class = OPERATOR_CLASSES["SUM"]
    except KeyError:
        operator_class = None
    return operator_class