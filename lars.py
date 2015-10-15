import sys
import os
import h5py
import scipy.sparse
import sklearn.linear_model
import sklearn.svm
import numpy


def main():
    # Get and check the filenames and the alpha.
    assert len(sys.argv) > 3
    filename_lars = sys.argv[1]
    algoname = sys.argv[2]
    alpha = float(sys.argv[3])
    assert os.path.isfile(filename_lars)

    # Read the files.
    with h5py.File(filename_lars) as f:
        values = f["values"].value
        col_index =  f["col_index"].value
        row_ptr = f["row_ptr"].value
        labels = f["labels"].value
    m = scipy.sparse.csr_matrix((values, col_index, row_ptr))

    if algoname == "forest_garrote":
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
    main()
    sys.exit(0)
