#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

This code is a custom loop version of train_ptb.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
import argparse
import os
import sys

import numpy as np

import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers
#import chainerx

import default
import ptb
from cell import *
sys.path.append("../")
import util



INNER_NODE = 3


def main(mode, dump_dir, SEARCH, mode3=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of examples in each mini-batch')

    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    if SEARCH:
        parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    else:
        parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')


    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')

    parser.add_argument('--gradclip', '-c', type=float, default=0.25,
                        help='Gradient norm threshold to clip')

    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Directory that has `rnnln.model`'
                        ' and `rnnlm.state`')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of LSTM units in each layer')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    #device = chainer.get_device(args.device)
    """
    if device.xp is chainerx:
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)
    """
    device = 0
    chainer.backends.cuda.get_device_from_id(device).use()

    def evaluate(model, iter):
        # Evaluation routine to be used for validation and test.
        evaluator = model.copy()  # to use different state
        evaluator.predictor.reset_state()  # initialize state
        sum_perp = 0
        data_count = 0
        # Enable evaluation mode.
        with configuration.using_config('train', False):
            # This is optional but can reduce computational overhead.
            with chainer.using_config('enable_backprop', False):
                iter.reset()
                for batch in iter:
                    x, t = convert.concat_examples(batch, device)
                    loss = evaluator(x, t)
                    sum_perp += loss.array
                    data_count += 1

                    #print(data_count, loss)
        return np.exp(float(sum_perp) / data_count)

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = ptb.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab = {}'.format(n_vocab))

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    # Create the dataset iterators
    train_iter = default.ParallelSequentialIterator(train, args.batchsize)
    val_iter = default.ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = default.ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = default.RNNForLM(n_vocab, args.unit, inner_node=INNER_NODE, search=SEARCH, sharing=False)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    model.to_gpu(device)

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=10.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Load model and optimizer
    if args.resume is not None:
        resume = args.resume
        if os.path.exists(resume):
            serializers.load_npz(os.path.join(resume, 'rnnlm.model'), model)
            serializers.load_npz(
                os.path.join(resume, 'rnnlm.state'), optimizer)
        else:
            raise ValueError(
                '`args.resume` ("{}") is specified,'
                ' but it does not exist'.format(resume)
            )

    sum_perp = 0
    count = 0
    iteration = 0

    ##########################################


    with_zero_op = True
    
    if mode == 'A' and with_zero_op:
    #if False:
        options = {'zero_pos':4, 'prior':'uniform'}
    else:
        options = {'zero_pos':None, 'prior':'uniform'}

    npos = sum([i for i in range(1,INNER_NODE+1)])

    if with_zero_op:
        ncand = 5
    else:
        ncand = 4
        
    normal = util.Param(mode=mode, shape=(npos,ncand), max_iter=(args.epoch)*(26568/args.batchsize), options=options, mode2="CAT")
    rnn.l1.mode = mode

    normal.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(1.0))


    if False:
        a = normal.deter()
        a.data *= 0.
        a.data[:,4] = 1.
        rnn.l1.a = a


    if not SEARCH:
        f = open('./result/'+dump_dir+'/'+mode,'rb')
        while True:
            try:
                [_, _, _, best_arc, a] = pickle.load(f)
            except:
                break

        f.close()

        if mode3 == "PROC":
            a = best_arc

        rnn.l1.a = a

    #eval_rnn.l1.a = a
    #eval_rnn.l1.mode = mode


    baseline = util.EMA(coef=0.05)

    best_arc = normal.deter()
    best_loss = 99999999.


    f = open('./result/'+dump_dir+'/'+mode+"_eval_loss_log"+str(SEARCH),'w')
    f.close()
    ##########################################

    while train_iter.epoch < args.epoch:
        loss = 0
        iteration += 1
        
        ####
        if SEARCH:
            rnn.l1.a = normal.draw()
        else:
            normal.cleargrads()

        if iteration % 2 == 0 and SEARCH:
            update_w = False
            update_a = True
        else:
            update_w = True
            update_a = False


        lrcoef = (math.cos(train_iter.epoch*math.pi/args.epoch)+1.0)/2.0
        optimizer.lr = 20.0 * lrcoef
        normal.optimizer.alpha = 0.003 * lrcoef

        ####

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(args.bproplen):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = convert.concat_examples(batch, device)
            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(x, t)
            count += 1

        sum_perp += loss.array
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward(retain_grad=True)  # Backprop
        loss.unchain_backward()  # Truncate the graph

        ###
        baseline(float(loss.data))
        if update_w:
            optimizer.update()  # Update the parameters
        if update_a:
            if mode == "R":
                oarg = [float(loss.data) - baseline.get()]
            elif mode == "PA":
                oarg = [float(loss.data)]
            else:
                oarg = []

            normal.update(oarg)


        if float(loss.data) < best_loss:
            best_loss = float(loss.data)
            best_arc = rnn.l1.a


        ###

        if iteration % 10 == 0:
            if SEARCH:
                f = open('./result/'+dump_dir+'/'+mode,'ab')
                pickle.dump([train_iter.epoch, iteration, best_loss, best_arc, normal.deter()], f)
                f.close()
            
            f = open('./result/'+dump_dir+'/'+mode+"_eval_loss_log"+str(SEARCH),'a')
            f.write(str(train_iter.epoch)+", iter= "+str(iteration)+", train_perp="+str(np.exp(float(sum_perp) / count)) + "\r\n")
            f.close()



            print('iteration: {}'.format(iteration))
            print('training perplexity: {}'.format(
                np.exp(float(sum_perp) / count)))
            sum_perp = 0
            count = 0

        if train_iter.is_new_epoch:
            print('epoch: {}'.format(train_iter.epoch))
        #    print('validation perplexity: {}'.format(
        #        evaluate(model, val_iter)))

    # Evaluate on test dataset

    if not SEARCH:
        test_perp = evaluate(model, test_iter)
        val_perp = evaluate(model, val_iter)
        
        f = open('./result/'+dump_dir+'/'+mode+'_eval_log','w')
        f.write(str(best_loss)+", val_perp = "+str(val_perp)+", test_perp = "+str(test_perp)+"\r\n")
        f.close()

        f = open('./result/'+dump_dir+'/'+mode+'_eval_model','wb')
        pickle.dump(model, f)
        pickle.dump(a, f)
        f.close()

    # Save the model and the optimizer
    #out = args.out
    #if not os.path.exists(out):
    #    os.makedirs(out)
    #print('save the model')
    #serializers.save_npz(os.path.join(out, 'rnnlm.model'), model)
    #print('save the optimizer')
    #serializers.save_npz(os.path.join(out, 'rnnlm.state'), optimizer)


modes = ["A"]#,"R","P","GD","PA"]
modes3 = ["ARGM"]#,"PROC","ARGM","PROC","ARGM"]

dump_dir = "rb4"

def search():

    os.mkdir("./result/"+dump_dir)
    SEARCH = True

    for i in range(len(modes)):
        main(mode=modes[i], dump_dir=dump_dir, SEARCH=SEARCH)


#search()


def eval():

    SEARCH = False

    for i in range(len(modes)):
        main(mode=modes[i], dump_dir=dump_dir, SEARCH=SEARCH, mode3=modes3[i])




dirbase = "rb"
ntrial = 1
offset = 4

def make_table():


    for i in range(len(modes)):
        mode = modes[i]

        total = [[], []]
        for trial in range(ntrial):
            dump_dir = dirbase+str(trial+offset)
            f = open('./result/'+dump_dir+'/'+mode+"_eval_log",'r')
            r = f.read()
            f.close()

            r = r.split(" ")
            valp = float(r[3].replace(",",""))
            testp = float(r[6])

            total[0].append(valp)
            total[1].append(testp)


        for j in range(2):
            res = []
            res.append(float(np.mean(np.array(total[j]))))
            res.append(float(np.std(np.array(total[j]))))

            for k in range(2):
                res[k] = round(res[k],2)
                res[k] = str(res[k]).ljust(5,'0')

            
            print(mode, ["val","test"][j], res[0], "\\pm" , res[1])

def make_table_param():

    for i in range(len(modes)):
        mode = modes[i]

        total = []
        for trial in range(ntrial):
            dump_dir = dirbase+str(trial+offset)

            f = open('./result/'+dump_dir+'/'+mode+'_eval_model','rb')
            _ = pickle.load(f)
            a = pickle.load(f)
            f.close()

            ni = 300
            nv = 10000
            
            np = 0.
            #np += 2*nv*ni
            np += 2*ni*ni*(float(xp.sum(a.data[:,1:]))+2)
            total.append(np/1000/1000)

        mean = float(xp.mean(xp.array(total)))
        mean = round(mean,2)

        std = float(xp.std(xp.array(total)))
        std = round(std,2)
        print(mode, mean, std)



make_table()
#eval()


"""
ENAS

Deriving Architectures. We discuss how to derive novel
architectures from a trained ENAS model. We first sample several models from the trained policy π(m, θ). For
each sampled model, we compute its reward on a single
minibatch sampled from the validation set. We then take
only the model with the highest reward to re-train from
scratch. 





PARSEC

 In this work, we
consider the mode of the architecture distribution and train it from scratch. An alternative
strategy is to train multiple samples from the learnt distribution from scratch and consider
model ensembling. We plan to explore this direction in future work.

"""


