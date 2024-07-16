# Based on: https://github.com/facebookresearch/fastText/blob/master/alignment/unsup_align.py
# and https://github.com/facebookresearch/fastText/blob/master/alignment/utils.py

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time, math, ot
import torch
from tqdm import trange


def sqrt_eig(x):
    U, s, VT = torch.linalg.svd(x, full_matrices=False)
    return torch.matmul(U, torch.matmul(torch.diag(torch.sqrt(s)), VT))


def procrustes(X_src, Y_tgt):
    U, s, V = torch.linalg.svd(torch.matmul(Y_tgt.T, X_src))
    return torch.matmul(U, V)


class WassersteinAligner:

    def __init__(self, reg_init, reg_ws, batchsize, lr, n_iter_init, n_iter_ws, n_epoch, vocab_size, lr_decay,
                 apply_sqrt, early_stopping, seed=42, verbose=True, min_epsilon=0.005):
        # np.random.seed(seed)
        self.reg_init = reg_init
        self.reg_ws = reg_ws
        self.batchsize = batchsize
        self.lr = lr
        self.n_iter_ws = n_iter_ws
        self.n_iter_init = n_iter_init
        self.n_epoch = n_epoch
        self.vocab_size = vocab_size
        self.lr_decay = lr_decay
        self.apply_sqrt = apply_sqrt
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.min_epsilon = min_epsilon

        self.X = None
        self.Y = None
        self.use_gpu = False
        if torch.cuda.is_available():
            self.use_gpu = True
            if self.verbose:
                for i in range(torch.cuda.device_count()):
                    print("Found GPU " + torch.cuda.get_device_properties(i).name)
                print("Will use GPU for acceleration.")

    def objective(self, R, n=1000):
        Xn = self.X[torch.randperm(len(self.X))[:n]]
        Yn = self.Y[torch.randperm(len(self.Y))[:n]]
        C = -torch.matmul(torch.matmul(Xn, R), Yn.T)
        if self.use_gpu:
            P = ot.sinkhorn(torch.ones(n, device='cuda'), torch.ones(n, device='cuda'), C, 0.9, stopThr=1e-3,
                            numItermax=1000, method="sinkhorn_log")
        else:
            P = ot.sinkhorn(torch.ones(n), torch.ones(n), C, 0.9, stopThr=1e-3, numItermax=1000)
        return 1000 * torch.linalg.norm(torch.matmul(Xn, R) - torch.matmul(P, Yn)) / n

    def solve_procrustes(self, R):
        assert self.X is not None and self.Y is not None, "Matrices must not be empty!"
        no_improvement = 0
        first_obj = -1
        t = 0.5
        prev_obj = float("inf")
        best_obj = float("inf")
        best_R = R
        #print("R: ", R)

        for epoch in range(1, self.n_epoch + 1):
            if self.early_stopping > 0 and no_improvement >= self.early_stopping:
                if self.verbose:
                    print("Objective didn't improve for %i epochs. Early Stopping..." % self.early_stopping)
                    print("Improvement: %f" % (first_obj - best_obj))
                return best_R, best_obj, False

            for _it in trange(1, self.n_iter_ws + 1, desc="Iteration", disable=not self.verbose):
                # sample mini-batch
                xt = self.X[torch.randperm(len(self.X))[:self.batchsize]]
                yt = self.Y[torch.randperm(len(self.Y))[:self.batchsize]]

                # compute OT on minibatch
                C = -torch.matmul(torch.matmul(xt, R), yt.T)

                if self.use_gpu:
                    P = ot.sinkhorn(torch.ones(self.batchsize, device='cuda'),
                                    torch.ones(self.batchsize, device='cuda'), C, self.reg_ws, stopThr=t,
                                    numItermax=100000, method="sinkhorn_log")

                else:
                    P = ot.sinkhorn(torch.ones(self.batchsize), torch.ones(self.batchsize), C, self.reg_ws,
                                    stopThr=t,
                                    numItermax=100000)

                # compute gradient
                G = - torch.matmul(xt.T, torch.matmul(P, yt))
                R -= self.lr / self.batchsize * G
                # project on orthogonal matrices
                U, s, VT = torch.linalg.svd(R)
                R = torch.matmul(U, VT)
                if first_obj == -1:
                    obj = 0
                    for i in range(5):
                        obj += self.objective(R, n=min(20000, min(self.Y.shape[0], self.X.shape[0])))

                    obj /= 5
                    first_obj = obj

            self.lr *= self.lr_decay
            self.lr = max(self.lr, 5)

            obj = 0
            for i in range(5):
                obj += self.objective(R, n=min(20000, min(self.Y.shape[0], self.X.shape[0])))

            obj /= 5

            if obj < best_obj:
                improved_ratio = obj / first_obj
                if best_obj != float("inf") and improved_ratio < 0.85:
                    if self.verbose:
                        print("Objective of  %.3f is %2.2f %% of initial value. Early stopping..." % (
                        obj, improved_ratio * 100))
                    return R, obj, True
                best_obj = obj
                best_R = R

            if self.verbose or self.early_stopping > 0:
                if self.verbose:
                    print("epoch: %d  obj: %.3f  best: %.3f" % (epoch, obj, best_obj))

                if (prev_obj - best_obj) < self.min_epsilon:
                    no_improvement += 1
                else:
                    no_improvement = 0
                prev_obj = best_obj

        return best_R, best_obj, False

    def convex_init(self, X=None, Y=None):
        if X is not None or Y is not None:
            self.X = torch.from_numpy(X)
            self.Y = torch.from_numpy(Y)

        # If the two matrices contain a different number of records, reduce the size to the smaller of the two
        # by random subsampling.
        #print(self.X.shape[0])
        #print(self.Y.shape[1])
        #print(type(self.X.shape))

        if self.X.shape[0] < self.Y.shape[0]:  # <
            X_c = self.X
            Y_c = self.Y[torch.randperm(len(self.Y))[:self.X.shape[0]]]
        elif self.X.shape[0] > self.Y.shape[0]:  # >
            X_c = self.X[torch.randperm(len(self.X))[:self.Y.shape[0]]]
            Y_c = self.Y
        else:
            X_c = self.X
            Y_c = self.Y

        if X_c.shape[0] > 10000:
            X_c = X_c[torch.randperm(len(X_c))[:10000]]

        if Y_c.shape[0] > 10000:
            Y_c = Y_c[torch.randperm(len(Y_c))[:10000]]

        n, d = X_c.shape

        if self.apply_sqrt:
            X_c, Y_c = sqrt_eig(X_c), sqrt_eig(Y_c)

        K_X, K_Y = torch.matmul(X_c, X_c.T), torch.matmul(Y_c, Y_c.T)
        K_Y *= torch.linalg.norm(K_X) / torch.linalg.norm(K_Y)
        K2_X, K2_Y = torch.matmul(K_X, K_X), torch.matmul(K_Y, K_Y)
        P = torch.ones(n, n) / float(n)
        if self.use_gpu:
            P = P.to('cuda')
        for it in trange(1, self.n_iter_init + 1, disable=not self.verbose):
            G = torch.matmul(P, K2_X) + torch.matmul(K2_Y, P) - 2 * torch.matmul(K_Y, torch.matmul(P, K_X))
            if self.use_gpu:
                q = ot.sinkhorn(torch.ones(n, device='cuda'), torch.ones(n, device='cuda'), G, self.reg_init,
                                stopThr=3, method="sinkhorn_log", numItermax=10000)
            else:
                q = ot.sinkhorn(torch.ones(n), torch.ones(n), G, self.reg_init, stopThr=3, numItermax=10000)
            alpha = 2.0 / float(2.0 + it)
            P = alpha * q + (1.0 - alpha) * P
        R0 = procrustes(torch.matmul(P, X_c), Y_c).T
        obj = self.objective(R0, n=min(10000, min(self.Y.shape[0], self.X.shape[0])))
        if self.verbose:
            print("Objective after convex initialization: %f" % obj)
        #print("R0: ", R0)
        #print("obj: ", obj)
        return R0, obj

    def align(self, src, tgt):

        self.X = torch.from_numpy(src)
        self.Y = torch.from_numpy(tgt)

        if self.use_gpu:
            self.X = self.X.to('cuda')
            self.Y = self.Y.to('cuda')

        t0 = time.time()

        best_obj = float("inf")
        for i in range(3):
            if self.verbose:
                print("\nComputing initial mapping with convex relaxation...")
            t0 = time.time()
            R0, _ = self.convex_init()
            if self.verbose:
                print("Done [%03d sec]" % math.floor(time.time() - t0))
                print("\nComputing mapping with Wasserstein Procrustes...")

            tmp_r, tmp_obj, success = self.solve_procrustes(R0)
            if tmp_obj < best_obj:
                R = tmp_r
                best_obj = tmp_obj
            if success:
                break

        if self.verbose:
            print("Done [%03d sec]" % math.floor(time.time() - t0))
        torch.cuda.empty_cache()
        return R.numpy(force=True)
