import numpy as np
from util import softmax


def rnn_cell_forward(x_t,s_prev,parameters):

    U = parameters["U"]
    W = parameters["W"]
    V = parameters["V"]
    ba = parameters["ba"]
    by = parameters["by"]
    s_next = np.tanh(np.dot(U,x_t) + np.dot(W, s_prev) + ba)

    out_pred = softmax(np.dot(V, s_next) + by)

    cache = (s_next,s_prev,x_t,parameters)

    return s_next, out_pred, cache


def rnn_forward(x,s0,parameters):
    caches = []
    m,_,T = x.shape
    m,n = parameters["V"].shape
    s_next = s0
    s = np.zeros((n,1,T))
    y = np.zeros((m,1,T))
    for t in range(T):
        s_next,out_pred,cache = rnn_cell_forward(x[:,:,t],s_next, parameters)
        s[:,:,t] = s_next
        y[:,:,t] = out_pred

        caches.append(cache)
    return s,y,caches

def rnn_cell_backward(ds_next, cache):
    (s_next, s_prev, x_t,parameters) = cache
    U = parameters["U"]
    W = parameters["W"]
    V = parameters["V"]
    ba = parameters["ba"]
    by = parameters["by"]

    dtanh = (1 - s_next ** 2) * ds_next

    dU = np.dot(dtanh,x_t.T)
    dW = np.dot(dtanh,s_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims=1)
    dx_t = np.dot(U.T, dtanh)

    ds_prev = np.dot(W.T,dtanh)

    gradients = {"dtanh":dtanh,"dU":dU,"dW":dW,"dba":dba,"dx_t":dx_t,"ds_prev":ds_prev}

    return gradients

def rnn_backward(ds,caches):
    n,_,T = ds.shape
    (s1,s0,x_1,parameters) = caches[0]
    m,_ = x_1.shape
    dU = np.zeros((n,m))
    dW = np.zeros((n,n))
    dba = np.zeros((n,1))
    dx = np.zeros((m,1,T))

    ds_prevt = np.zeros((n,1))
    for t in reversed(range(T)):
        gradients = rnn_cell_backward(ds[:,:,t] + ds_prevt,caches[t])
        dU += gradients["dU"]
        dW += gradients["dW"]
        dba += gradients["dba"]
        dx[:,:,t]=gradients["dx_t"]

    gradients = {"dU": dU, "dW": dW, "dba": dba, "dx": dx}