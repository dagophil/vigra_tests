import sys
import os
import h5py
import scipy.sparse
import sklearn.linear_model


def main():
    # Get and check the filenames and the alpha.
    assert len(sys.argv) > 2
    filename_lars = sys.argv[1]
    alpha = float(sys.argv[2])
    assert os.path.isfile(filename_lars)

    # Read the files.
    with h5py.File(filename_lars) as f:
        values = f["values"].value
        col_index =  f["col_index"].value
        row_ptr = f["row_ptr"].value
        labels = f["labels"].value
    m = scipy.sparse.csr_matrix((values, col_index, row_ptr))

    # Do the lasso.
    coefs = sklearn.linear_model.lasso_path(m, labels, positive=True, alphas=[alpha])[1]
    coefs = coefs[:, -1]

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
