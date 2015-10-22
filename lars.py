import sys
import os
import h5py
import scipy.sparse
import sklearn.linear_model
import sklearn.svm
import numpy
import argparse
import warnings
from sklearn.utils import ConvergenceWarning
from multiprocessing import Process, Queue, cpu_count


parser = argparse.ArgumentParser(description="Forest garrote script")
parser.add_argument("-f", type=str, required=True,
                    help="filename of the sparse input matrix")
parser.add_argument("-n", type=str, required=True,
                    help="name of the algorithm")
parser.add_argument("-a", type=float, default=0.0003, help="alpha value of forest garrote")
parser.add_argument("-g", type=int, nargs="*", default=[], help="indices where the group splits are made")
parser.add_argument("--n_threads", type=int, default=1, help="number of threads")


def lars_worker(in_qu, out_qu, m, labels, alpha):
    while True:
        item = in_qu.get()
        if item is None:
            break
        i, begin, end = item
        sub_m = m[:, begin:end]
        coefs = sklearn.linear_model.lasso_path(sub_m, labels, positive=True, max_iter=100, alphas=[alpha])[1]
        coefs = coefs[:, -1]
        out_qu.put((i, coefs))


def main(args):
    # Get the arguments.
    filename_lars = args.f
    algoname = args.n
    alpha = args.a
    group_splits = args.g
    assert os.path.isfile(filename_lars)
    
    warnings.simplefilter("ignore", ConvergenceWarning)

    # Read the files.
    with h5py.File(filename_lars) as f:
        values = f["values"].value
        col_index =  f["col_index"].value
        row_ptr = f["row_ptr"].value
        labels = f["labels"].value
    m = scipy.sparse.csr_matrix((values, col_index, row_ptr))

    if algoname == "forest_garrote" and len(group_splits) == 0:
        # Do the lasso.
        coefs = sklearn.linear_model.lasso_path(m, labels, positive=True, max_iter=100, alphas=[alpha])[1]
        coefs = coefs[:, -1]
    elif algoname == "l2_svm":
        # Use an l2 svm.
        svm = sklearn.svm.LinearSVC(C=1.0, penalty="l2")
        svm.fit(m, labels)
        coefs = svm.coef_[0, :]
    elif algoname == "l1_svm":
        # Use an l1 svm.
        svm = sklearn.svm.LinearSVC(C=1.0, penalty="l1", dual=False)
        svm.fit(m, labels)
        coefs = svm.coef_[0, :]
    elif algoname == "forest_garrote" and len(group_splits) > 0:
        # Make groups and run the forest garrote on each group.
        group_splits = [0] + group_splits + [m.shape[1]]
        n_groups = len(group_splits)-1

        if args.n_threads < 1:
            args.n_threads = cpu_count()
        n_threads = min(n_groups, args.n_threads)
        if n_threads == 1:
            coef_list = []
            for i in xrange(n_groups):
                begin = group_splits[i]
                end = group_splits[i+1]
                sub_m = m[:, begin:end]
                coefs = sklearn.linear_model.lasso_path(sub_m, labels, positive=True, max_iter=100, alphas=[alpha])[1]
                coefs = coefs[:, -1]
                coef_list.append(coefs)
            coefs = numpy.concatenate(coef_list)
            coefs = coefs/n_groups
        else:
            in_qu = Queue()
            out_qu = Queue()
            procs = [Process(target=lars_worker, args=(in_qu, out_qu, m, labels, alpha)) for _ in xrange(n_threads)]
            for i in xrange(n_groups):
                begin = group_splits[i]
                end = group_splits[i+1]
                in_qu.put((i, begin, end))
            for p in procs:
                in_qu.put(None)
                p.start()
            coef_list = [None] * n_groups
            for i in xrange(n_groups):
                k, coefs = out_qu.get()
                coef_list[k] = coefs
            for p in procs:
                p.join()
            coefs = numpy.concatenate(coef_list)
            coefs = coefs/n_groups

        # Use an additional l2 svm to refine the weights.
        nnz = coefs.nonzero()[0]
        m_sub = m[:, nnz]
        svm = sklearn.svm.LinearSVC(C=1.0, penalty="l2")
        svm.fit(m_sub, labels)
        new_coefs = svm.coef_[0, :]
        coefs = numpy.zeros(m.shape[1])
        coefs[nnz] = new_coefs
    else:
        raise Exception("Unknown algorithm: " + algoname)

    # Save the results.
    nnz = coefs.nonzero()[0]
    nnz_coefs = coefs[nnz]
    with h5py.File(filename_lars) as f:
        if "result_nnz" in f:
            del f["result_nnz"]
        if "result_nnz_coefs" in f:
            del f["result_nnz_coefs"]
        f.create_dataset("result_nnz", data=nnz, compression="gzip", compression_opts=5)
        f.create_dataset("result_nnz_coefs", data=nnz_coefs, compression="gzip", compression_opts=5)


if __name__ == "__main__":
    main(parser.parse_args())
    sys.exit(0)

