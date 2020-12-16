
import os

with open(os.path.abspath(__file__).replace("cell.py","../setting.py"), "r") as f:
    code = f.read()
    exec(code)


def edge_act(x, a, mode, SEARCH):

    y = 0.

    for op_type in range(len(a)):

        if (mode in ONEHOTTER or not SEARCH) and float(a[op_type].data) == 0.:
            skip = True
        else:
            if op_type == 0:
                h = 1. * x
            if op_type == 1:
                h = F.tanh(x)
            if op_type == 2:
                h = F.sigmoid(x)
            if op_type == 3:
                h = F.relu(x)
            if op_type == 4:
                h = 0. * x

            if SEARCH:
                y += F.scale(h, a[op_type])
            else:
                y += h


        if SEARCH and mode == "GD" and float(a[op_type].data) == 0.:
            c = 1. + (x.data * 0.)
            y += F.scale(c, a[op_type])

    return y



# ENAS implementation https://github.com/melodyguan/enas/blob/master/src/ptb/ptb_enas_child.py
# DARTS https://github.com/quark0/darts/blob/master/rnn/model_search.py
# DARTS https://github.com/quark0/darts/blob/master/rnn/model.py

class Cell(chainer.ChainList):

    def __init__(self, nnode, inner_node, search, sharing):
        super(Cell, self).__init__()
        with self.init_scope():

            self.inner_node = inner_node
            self.SEARCH = search
            self.SHARING = sharing

            for i in range(4):
                layer = L.Linear(nnode, nnode, nobias=True)
                #layer.W.data = xp.random.rand(nnode,nnode).astype(xp.float32)*INITRANGE
                self.add_link(layer)

           
            for i in range(1,self.inner_node+1):
                if self.SHARING:
                    numw = 1
                else:
                    numw = i

                for j in range(numw):
                    for k in range(2):
                        layer = L.Linear(nnode, nnode, nobias=True)
                        self.add_link(layer)

                    bn = L.BatchNormalization(nnode, use_gamma=False, use_beta=False)
                    self.add_link(bn)
                

            self.h = None


    def reset_state(self):
        self.h = None

    def __call__(self, x):

        a = self.a
        mode = self.mode

        # start gen init state
        edges = self.children()

        c0 = 0.
        h0 = 0.

        for i in range(4):
            layer = edges.__next__()

            if not (self.h is None):
                if i == 0:
                    c0 += layer(self.h)
                if i == 1:
                    h0 += layer(self.h)
            if i == 2:
                c0 += layer(x)
            if i == 3:
                h0 += layer(x)

        c0 = F.sigmoid(c0)
        h0 = F.tanh(h0)

        if not (self.h is None):
            node = [self.h + c0 * (h0 - self.h)]
        else:
            node = [c0 * h0]

        # end gen init state

        pos = 0
        for i in range(1,self.inner_node+1):
            
            node.append(None)
            if self.SHARING:
                lh = edges.__next__()
                lc = edges.__next__()
                bn = edges.__next__()


            c2 = 0.
            h = 0.
            for j in range(0,i):
                if not self.SHARING:
                    lh = edges.__next__()
                    lc = edges.__next__()
                    bn = edges.__next__()

                h1 = node[j]
                #if self.SEARCH:
                if False:
                    h1 = bn(h1)

                #h1 = F.dropout(h1, 0.25)

                c2 = F.sigmoid(lc(h1))
                h = lh(h1)
                h = edge_act(h, a[pos], mode, self.SEARCH)

                h2 = h1 + c2 * (h - h1)

                if node[i] is None:
                    node[i] = h2
                else:
                    node[i] = F.add(node[i],h2)

                pos += 1

            node[i] = F.scale(node[i], xp.array(1./i).astype(xp.float32))


        y = 0.
        for i in range(1,self.inner_node+1):
            y += node[i]

        y *= 1./self.inner_node
        self.h = y
        return self.h
