import numpy as np
import datetime
import json
from math import log10
from scipy.optimize import minimize
from random import random
import sys

from jax.lax import fori_loop, scan
from jax import jit
from time import time
import jax.numpy as np
import jax

sys.float_info.max

kInspectThreshold = 5
kMinValue = -999999999
kMaxValue = sys.float_info.max


def loadFilteredData(file_path):
    dict = {}
    resturant_count = 0
    food_inspection_data = open(file_path, "r")
    food_inspection_data = json.load(food_inspection_data)
    for license in food_inspection_data.keys():
        if (len(food_inspection_data[license]) > kInspectThreshold):
            resturant_count += 1
            dict[license] = food_inspection_data[license]
    return dict


def test(act_seq, rel_seq, x, init_state):
    curr_state = init_state
    x = jax.nn.sigmoid(x)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    noaction_matrix = np.array([[x1, 1 - x1], [x2, 1 - x2]])
    action_matrix = np.array([[x3, 1 - x3], [x4, 1 - x4]])
    for i in range(len(act_seq)):
        if (act_seq[i] == 0):
            curr_state = np.dot(curr_state, noaction_matrix)
        else:
            print("curr_b_state ", end=" ")
            print(curr_state, end=" ")
            if (rel_seq[i] == 0):
                curr_state = np.array([1.0, 0.0])
            else:
                curr_state = np.array([0.0, 1.0])
            print("curr_state ", end=" ")
            print(curr_state, end=" ")
            print()
            curr_state = np.dot(curr_state, action_matrix)


def gridSearch(inspec_seq):
    if (inspec_seq[0][1] == 0):
        init_state = np.array([1.0, 0.0])
    else:
        init_state = np.array([0.0, 1.0])
    act_seq, rel_seq = insToActionSeq(inspec_seq)
    x = [.9, .1, .9, 0.9]
    bnds = ((0, 1), (0, 1), (0, 1), (0, 1))
    res = minimize(loss, x,args=(act_seq, rel_seq, init_state), method='Nelder-Mead', bounds=bnds)
    return res.x

def gradStep(gout, x, lr):
    return x - lr * gout[0]


def update(x, act_seq, rel_seq, init_state, epoches):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    noActionsCount = 0
    actionsCount = 0
    prev_state = init_state
    for iter in range(epoches):
        for i in range(len(act_seq)):
            if (act_seq[i] == 0):
                noActionsCount += 1
            else:
                # print(prev_state,end="  ")
                # print(actionsCount,end=" ")
                # print(noActionsCount,end=" ")
                # print(rel_seq[i])
                grad = jax.grad(stepFun, argnums=[1, 2, 3, 4])
                gout = grad(prev_state, x1, x2, x3, x4, actionsCount, noActionsCount, rel_seq[i])
                x1, x2, x3, x4 = gradStep(gout, x1, x2, x3, x4, lr=0.01)
                # print(gout)
                actionsCount = 1
                noActionsCount = 0
                if (rel_seq[i] == 0):
                    prev_state = np.array([1.0, 0.0])
                else:
                    prev_state = np.array([0.0, 1.0])
    print(x1, x2, x3, x4)
    return [x1, x2, x3, x4]


def insToActionSeq(inspec_seq):
    timeStrSeq = []
    rel_dict = {}
    print(inspec_seq)
    for i in range(1, len(inspec_seq)):
        inspec_date = datetime.datetime.strptime(inspec_seq[i][0], "%m/%d/%Y")
        key = str(inspec_date.year) + str(inspec_date.month)
        timeStrSeq.append(key)
        if (not (key in rel_dict.keys())):
            if (inspec_seq[i][1] == 1):
                rel_dict[key] = 1
            else:
                rel_dict[key] = 0
        else:
            rel_dict[key] = max(rel_dict[key], inspec_seq[i][1])
    start_date = datetime.datetime.strptime(inspec_seq[0][0], "%m/%d/%Y")
    end_date = datetime.datetime.strptime(inspec_seq[-1][0], "%m/%d/%Y")
    interval = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
    act_sqe = []
    rel_seq = []
    start_from_year = start_date.year
    start_from_month = start_date.month
    for months in range(1, interval + 1):
        month = start_from_month + months
        year_tmp, month_tmp = start_from_year + month // 12, month % 12
        if (month_tmp == 0):
            month_tmp = 12
            year_tmp -= 1
        if ((str(year_tmp) + str(month_tmp)) in timeStrSeq):
            act_sqe.append(1)
            rel_seq.append(rel_dict[str(year_tmp) + str(month_tmp)])
        else:
            act_sqe.append(0)
            rel_seq.append(-1)
    return act_sqe, rel_seq


def findTransitionMatrix(inspects):
    tmp = inspects
    tmp.reverse()
    min_error = kMaxValue
    trans_probality = None
    for i in range(20):
        init_probality = [random(), random(), random(), random()]
        error, probality = gridSearch(tmp, init_probality)
        if (error < min_error):
            min_error = error
            trans_probality = probality
    return trans_probality


def stepFun(state, x1, x2, x3, x4, init, iters, label):
    P1 = np.array([[x1, 1 - x1], [x2, 1 - x2]])
    P2 = np.array([[x3, 1 - x3], [x4, 1 - x4]])
    if (init):
        state = np.dot(state, P2)

    def step(t, carry):
        last_s = carry
        next_x = np.dot(last_s, P1)
        return next_x

    outs = fori_loop(0, iters, step, (state))
    if (label == 0):
        return -np.log(outs[0])
    else:
        return -np.log(outs[1])


def loss(x, act_seq, rel_seq, init_state):
    x = jax.nn.sigmoid(x)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    out = 0
    noaction_t = np.array([[x1, 1 - x1], [x2, 1 - x2]])
    action_t = np.array([[x3, 1 - x3], [x4, 1 - x4]])
    prev_state = init_state
    for i in range(len(act_seq)):
        if (act_seq[i] == 0):
            prev_state = np.dot(prev_state, noaction_t)
        else:
            y_hat = prev_state
            if (rel_seq[i] == 0):
                label = np.array([1.0, 0.0])
                out -= np.log(prev_state[0])
                prev_state = np.array([1.0, 0.0])
            else:
                label = np.array([0.0, 1.0])
                out -= np.log(prev_state[1])
                prev_state = np.array([0.0, 1.0])
            prev_state = np.dot(prev_state, action_t)

    return out


# grad = jax.grad(test, argnums=[0])
# gout = grad(1.0)
# print(gout)

inspect_data = loadFilteredData("inspection_samples.json")
inspec_seq = inspect_data["31035"]
inspec_seq.reverse()
if inspec_seq[0][1] == 0:
    init_state = np.array([1.0, 0.0])
else:
    init_state = np.array([0.0, 1.0])
act_seq, rel_seq = insToActionSeq(inspec_seq)

epoches = 500
x = np.array([.9, .1, .9, 0.1])
print("Intital Loss:" ,end=" ")
print(loss(x, act_seq,rel_seq, init_state))
grads = []
for i in range(epoches):
    grad = jax.grad(loss, argnums=[0])
    gout = grad(x, act_seq, rel_seq, init_state)
    grads.append(gout)
    x = gradStep(gout, x, 0.01)
# print(jax.nn.sigmoid(x))
# test(act_seq, rel_seq, x, init_state)
print("GD Loss:" ,end=" ")
print(loss(x, act_seq, rel_seq, init_state))


x = np.array([.9, .1, .9, 0.1])
x = gridSearch(inspec_seq)
print("Scipy Loss:" ,end=" ")
print(loss(x, act_seq, rel_seq, init_state))
# trans_matrix_dict = {}
# for key in inspect_data.keys():
#     p_matrix = findTransitionMatrix(inspect_data[key])
#     trans_matrix_dict[key] = p_matrix.tolist()
#
#
# json_file = open("transitions.json", "w")
# json.dump(trans_matrix_dict, json_file)
