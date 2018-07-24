import os
import time
import math
import json
import joblib
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from datasets import pw, rocstories
from analysis import rocstories as rocstories_analysis
from analysis import pw  as pw_analysis
from text_utils import TextEncoder
from utils import encode_dataset, flatten, iter_data, find_trainable_variables, get_ema_vars, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

############ NN FUNCS ############ 
def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    #final linear layer
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b

def clf_pw(x, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False, ordinal=False):
    out_sz = 5 if ordinal else 2
    #final linear layer, for single input
    with tf.variable_scope('clf_pw'):
        nx = shape_list(x)[-1]
        w = tf.get_variable('w', [nx, out_sz], initializer=w_init)
        b = tf.get_variable('b', [out_sz], initializer=b_init)
        return tf.matmul(x, w)+b

#MAIN MODELS
def model_pw(X, M, Y, train=False, reuse=False, ordinal=False):
    """
        X: [batch, n_ctx, 2]
        M: [batch, n_ctx]
        Y: [batch]
    """
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        #transformer blocks
        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        #language modeling objective
        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        clf_h = tf.reshape(h, [-1, n_embd])
        #get length of each example
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        #just takes the state of the transformer at the end of the input, I think?
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

        #reshape to [batch, embed size]
        clf_h = tf.reshape(clf_h, [-1, 1, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        #put tensor back into (batch, embed size) shape
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        #linear layer
        clf_logits = clf_pw(clf_h, train=train, ordinal=ordinal)

        #final softmax
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)

        return clf_logits, clf_losses, lm_losses

def model_roc(X, M, Y, train=False, reuse=False):
    """
        X: [batch, n_answers, n_ctx, 2]
        M: [batch, n_answers, n_ctx]
        Y: [batch]
    """
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        #turn n_answer into a 'batch' dimension, so we have [batch*n_answers, n_ctx, 2]
        X = tf.reshape(X, [-1, n_ctx, 2])
        M = tf.reshape(M, [-1, n_ctx])

        #transformer blocks
        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        #language modeling objective
        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)


        #concat doc with question with each of two answers, then linear + softmax
        #output of transformer: [batch*n_answer, embed]
        clf_h = tf.reshape(h, [-1, n_embd])

        #first, check if last dim of X is equal to end token
        #then, cast to float32, probably so we can use argmax
        #then, take argmax?
        #finally, cast to int32
        #I think this is just getting the length of each example in batch
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        #just takes the state of the transformer at the end of the input, I think?
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

        #reshape to [batch, n_answers, embed size]
        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        #put tensor back into (batch*n_answers, embed size) shape
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        #linear layer
        clf_logits = clf(clf_h, 1, train=train)
        #reshape to [batch, n_answers]
        clf_logits = tf.reshape(clf_logits, [-1, 2])

        #final softmax
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)

        return clf_logits, clf_losses, lm_losses

def mgpu_train(*xs, **kwargs):
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            if dataset == 'rocstories':
                clf_logits, clf_losses, lm_losses = model_roc(*xs, train=True, reuse=do_reuse)
            elif dataset == 'pw':
                clf_logits, clf_losses, lm_losses = model_pw(*xs, train=True, reuse=do_reuse, ordinal=ordinal)
            if lm_coef > 0:
                train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
            else:
                train_loss = tf.reduce_mean(clf_losses)
            params = find_trainable_variables("model/clf" if freeze_lm else "model")
            print(params)
            grads = tf.gradients(train_loss, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    grads = average_grads(gpu_grads)
    grads = [g for g, p in grads]
    train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    return [train]+ops

def mgpu_predict(*xs, **kwargs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            if dataset == 'rocstories':
                clf_logits, clf_losses, lm_losses = model_roc(*xs, train=False, reuse=True)
            elif dataset == 'pw':
                clf_logits, clf_losses, lm_losses = model_pw(*xs, train=False, reuse=True, ordinal=ordinal)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops
############ NN FUNCS ############ 

def transform_roc(X1, X2, X3):
    """
        Glue stories together with delimiter and stuff, and add position tokens
    """
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        #concatenate
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        #total length
        l12 = len(x12)
        l13 = len(x13)
        #set np array
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        #mask?
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    #position tokens
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def transform_pw(X1, X2):
    """
        Glue stories together with delimiter and stuff, and add position tokens
    """
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2) in enumerate(zip(X1, X2)):
        #concatenate
        x = [start] + x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        l = len(x)
        #set np array
        xmb[i,:l,0] = x
        #mask
        mmb[i,:l] = 1
    #position tokens
    xmb[:,:,1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def iter_apply(Xs, Ms, Ys):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            res = sess.run([eval_mgpu_logits, eval_mgpu_clf_loss], {X_train:xmb, M_train:mmb, Y_train:ymb})
        else:
            res = sess.run([eval_logits, eval_clf_loss], {X:xmb, M:mmb, Y:ymb})
        res = [r*n for r in res]
        results.append(res)
    results = zip(*results)
    return [fn(res) for res, fn in zip(results, fns)]

def iter_predict(Xs, Ms):
    logits = []
    for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train:xmb, M_train:mmb}))
        else:
            logits.append(sess.run(eval_logits, {X:xmb, M:mmb}))
    logits = np.concatenate(logits, 0)
    return logits

def save(path):
    ps = sess.run(params)
    joblib.dump(ps, make_path(path))

def log():
    global best_score
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost/len(trY[:n_valid])
    va_cost = va_cost/n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1))*100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1))*100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f'%(n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            save(os.path.join(save_dir, desc, 'best_params.jl'))

argmax = lambda x:np.argmax(x, 1)

pred_fns = {
    'rocstories':argmax,
    'pw': argmax
}

filenames = {
    'rocstories':'ROCStories.tsv',
    'pw': 'pw_preds.tsv'
}

label_decoders = {
    'rocstories':None,
    'pw': None
}

analyses = {
    'rocstories': rocstories_analysis,
    'pw': pw_analysis
}

def predict(test):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    if test:
        predictions = pred_fn(iter_predict(teX, teM))
    else:
        predictions = pred_fn(iter_predict(vaX, vaM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc',           type=str, help="name for the run")
    parser.add_argument('--dataset',        choices=['rocstories', 'pw'])
    parser.add_argument('--log_dir',        type=str, default='log/')
    parser.add_argument('--save_dir',       type=str, default='save/')
    parser.add_argument('--data_dir',       type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit',         action='store_true', help="flag to get test predictions")
    parser.add_argument('--analysis',       action='store_true', help="flag to run analysis")
    parser.add_argument('--ordinal',        action='store_true', help="flag to do 5-class prediction instead of binary")
    parser.add_argument('--test',           action='store_true', help="flag to run on test")
    parser.add_argument('--freeze_lm',      action='store_true', help="flag to freeze (not update) LM weights - only train the classifier")
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--n_iter',         type=int, default=3, help="epochs of finetuning")
    parser.add_argument('--n_batch',        type=int, default=8, help="number in batch per gpu")
    parser.add_argument('--max_grad_norm',  type=int, default=1)
    parser.add_argument('--lr',             type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup',      type=float, default=0.002)
    parser.add_argument('--n_ctx',          type=int, default=512, help="if longest input is longer than this, crop it to this size")
    parser.add_argument('--n_embd',         type=int, default=768, help="embedding size")
    parser.add_argument('--n_head',         type=int, default=12, help="attention heads")
    parser.add_argument('--n_layer',        type=int, default=12, help="transformer blocks")
    parser.add_argument('--embd_pdrop',     type=float, default=0.1)
    parser.add_argument('--attn_pdrop',     type=float, default=0.1)
    parser.add_argument('--resid_pdrop',    type=float, default=0.1)
    parser.add_argument('--clf_pdrop',      type=float, default=0.1)
    parser.add_argument('--l2',             type=float, default=0.01)
    parser.add_argument('--vector_l2',      action='store_true')
    parser.add_argument('--n_gpu',          type=int, default=4)
    parser.add_argument('--opt',            type=str, default='adam')
    parser.add_argument('--afn',            type=str, default='gelu')
    parser.add_argument('--lr_schedule',    type=str, default='warmup_linear')
    parser.add_argument('--encoder_path',   type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path',       type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer',     type=int, default=12, help="transformer blocks to keep fixed (default to all)")
    parser.add_argument('--lm_coef',        type=float, default=0.5)
    parser.add_argument('--b1',             type=float, default=0.9)
    parser.add_argument('--b2',             type=float, default=0.999)
    parser.add_argument('--e',              type=float, default=1e-8)

    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    log_file = os.path.join(log_dir, '{}.jsonl'.format(desc))
    logger = ResultLogger(path=log_file, **args.__dict__)
    # formatting stuff
    text_encoder = TextEncoder(encoder_path, bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    if args.dataset == 'rocstories':
        (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(rocstories(data_dir), encoder=text_encoder)
    elif args.dataset == 'pw':
        (trX1, trX2, trY), (vaX1, vaX2, vaY), (teX1, teX2, teY) = encode_dataset(pw(data_dir, args.ordinal), encoder=text_encoder)
    #output: unpadded lists of word indices
    #special tokens
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']

    #number of special characters
    n_special = 3

    max_len = n_ctx//2-2
    #get max length of story + answer in train, val, test
    #take min of (that + 3), n_ctx
    #the 3 is to take care of start, delimiter, end tokens
    if args.dataset == 'rocstories':
        n_ctx = min(max([len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)])+n_special, n_ctx)
    elif args.dataset == 'pw':
        n_ctx = min(max([len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(trX1, trX2)]+[len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(vaX1, vaX2)]+[len(x1[:max_len])+len(x2[:max_len]) for x1, x2 in zip(teX1, teX2)])+n_special, n_ctx)

    if args.dataset == 'rocstories':
        trX, trM = transform_roc(trX1, trX2, trX3)
        vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
        if submit:
            teX, teM = transform_roc(teX1, teX2, teX3)
    elif args.dataset == 'pw':
        trX, trM = transform_pw(trX1, trX2)
        vaX, vaM = transform_pw(vaX1, vaX2)
        if submit:
            teX, teM = transform_pw(teX1, teX2)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter


    #data placeholders
    if args.dataset == 'rocstories':
        X_train = tf.placeholder(tf.int32, [n_batch_train, 2, n_ctx, 2])
        M_train = tf.placeholder(tf.float32, [n_batch_train, 2, n_ctx])
        X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
        M = tf.placeholder(tf.float32, [None, 2, n_ctx])

        Y_train = tf.placeholder(tf.int32, [n_batch_train])
        Y = tf.placeholder(tf.int32, [None])
    elif args.dataset == 'pw':
        X_train = tf.placeholder(tf.int32, [n_batch_train, n_ctx, 2])
        M_train = tf.placeholder(tf.float32, [n_batch_train, n_ctx])
        X = tf.placeholder(tf.int32, [None, n_ctx, 2])
        M = tf.placeholder(tf.float32, [None, n_ctx])

        Y_train = tf.placeholder(tf.int32, [n_batch_train])
        Y = tf.placeholder(tf.int32, [None])

    #train setup
    train, logits, clf_losses, lm_losses = mgpu_train(X_train, M_train, Y_train, dataset=args.dataset, ordinal=args.ordinal)
    clf_loss = tf.reduce_mean(clf_losses)

    params = find_trainable_variables('model')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    #load saved params
    shapes = json.load(open('model/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:n_ctx]
    init_params[0] = np.concatenate([init_params[1], (np.random.randn(n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
    del init_params[1]

    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1+n_transfer*12
    sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])

    #more setup
    eval_mgpu_logits, eval_mgpu_clf_losses, eval_mgpu_lm_losses = mgpu_predict(X_train, M_train, Y_train, dataset=args.dataset, ordinal=args.ordinal)
    if args.dataset == 'rocstories':
        eval_logits, eval_clf_losses, eval_lm_losses = model_roc(X, M, Y, train=False, reuse=True)
    elif args.dataset == 'pw':
        eval_logits, eval_clf_losses, eval_lm_losses = model_pw(X, M, Y, train=False, reuse=True, ordinal=args.ordinal)
    eval_clf_loss = tf.reduce_mean(eval_clf_losses)
    eval_mgpu_clf_loss = tf.reduce_mean(eval_mgpu_clf_losses)

    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY
    if submit:
        save(os.path.join(save_dir, desc, 'best_params.jl'))
    best_score = 0
    # main training epoch loop
    for i in range(n_iter):
        for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random), n_batch=n_batch_train, truncate=True, verbose=True):
            cost, _ = sess.run([clf_loss, train], {X_train:xmb, M_train:mmb, Y_train:ymb})
            n_updates += 1
            if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
                log()
        n_epochs += 1
        log()

    # for ROC submission
    if submit:
        sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
        predict(args.test)
        if analysis:
            analy_fn = analyses[dataset]
            analy_fn(data_dir, os.path.join(submission_dir, filenames[dataset]), os.path.join(log_dir, log_file), test=args.test, ordinal=args.ordinal)
