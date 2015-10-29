#ifndef DATA_UTILITY_HXX
#define DATA_UTILITY_HXX

#include <vector>
#include <stdexcept>

#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>



template <typename S, typename T>
void load_data(
        std::string const & train_filename,
        std::string const & test_filename,
        vigra::MultiArray<2, S> & train_x,
        vigra::MultiArray<1, T> & train_y,
        vigra::MultiArray<2, S> & test_x,
        vigra::MultiArray<1, T> & test_y,
        std::vector<T> const & labels = {},
        std::string const & features_key = "data",
        std::string const & labels_key = "labels"
){
    using namespace vigra;

    // Load the data.
    MultiArray<2, S> tmp_train_x;
    MultiArray<1, T> tmp_train_y;
    MultiArray<2, S> tmp_test_x;
    MultiArray<1, T> tmp_test_y;
    HDF5ImportInfo info(train_filename.c_str(), features_key.c_str());
    tmp_train_x.reshape(Shape2(info.shape().begin()));
    readHDF5(info, tmp_train_x);
    info = HDF5ImportInfo(train_filename.c_str(), labels_key.c_str());
    tmp_train_y.reshape(Shape1(info.shape().begin()));
    readHDF5(info, tmp_train_y);
    info = HDF5ImportInfo(test_filename.c_str(), features_key.c_str());
    tmp_test_x.reshape(Shape2(info.shape().begin()));
    readHDF5(info, tmp_test_x);
    info = HDF5ImportInfo(test_filename.c_str(), labels_key.c_str());
    tmp_test_y.reshape(Shape1(info.shape().begin()));
    readHDF5(info, tmp_test_y);

    if (tmp_train_x.shape()[0] == tmp_train_y.size())
    {}
    else if (tmp_train_x.shape()[1] == tmp_train_y.size())
    {
        auto tmp_view = tmp_train_x.transpose();
        MultiArray<2, S> tmp_copy = tmp_view;
        tmp_train_x = tmp_copy;
    }
    else
    {
        vigra_fail("Wrong number of training instances.");
    }

    if (tmp_test_x.shape()[0] == tmp_test_y.size())
    {}
    else if (tmp_test_x.shape()[1] == tmp_test_y.size())
    {
        auto tmp_view = tmp_test_x.transpose();
        MultiArray<2, S> tmp_copy = tmp_view;
        tmp_test_x = tmp_copy;
    }
    else
    {
        vigra_fail("Wrong number of test instances.");
    }

    if (labels.size() == 0)
    {
        train_x = tmp_train_x;
        train_y = tmp_train_y;
        test_x = tmp_test_x;
        test_y = tmp_test_y;
        return;
    }

    // Restrict the training data to the given label subset.
    std::vector<size_t> train_indices;
    for (size_t i = 0; i < tmp_train_y.size(); ++i)
    {
        for (auto const & label : labels)
        {
            if (tmp_train_y[i] == label)
            {
                train_indices.push_back(i);
                break;
            }
        }
    }
    train_x.reshape(Shape2(train_indices.size(), tmp_train_x.shape()[1]));
    train_y.reshape(Shape1(train_indices.size()));
    for (size_t i = 0; i < train_x.shape()[0]; ++i)
    {
        for (size_t k = 0; k < train_x.shape()[1]; ++k)
        {
            train_x(i, k) = tmp_train_x(train_indices[i], k);
        }
        train_y[i] = tmp_train_y[train_indices[i]];
    }

    // Restrict the test data to the given label subset.
    std::vector<size_t> test_indices;
    for (size_t i = 0; i < tmp_test_y.size(); ++i)
    {
        for (auto const & label : labels)
        {
            if (tmp_test_y[i] == label)
            {
                test_indices.push_back(i);
                break;
            }
        }
    }
    test_x.reshape(Shape2(test_indices.size(), tmp_test_x.shape()[1]));
    test_y.reshape(Shape1(test_indices.size()));
    for (size_t i = 0; i < test_x.shape()[0]; ++i)
    {
        for (size_t k = 0; k < test_x.shape()[1]; ++k)
        {
            test_x(i, k) = tmp_test_x(test_indices[i], k);
        }
        test_y[i] = tmp_test_y[test_indices[i]];
    }
}



template <typename FeatureType, typename LabelType>
void create_real_kfolds(
        size_t n_kfolds,
        vigra::MultiArray<2, FeatureType> const & features,
        vigra::MultiArray<1, LabelType> const & labels,
        std::vector<vigra::MultiArray<2, FeatureType> > & kfold_train_features,
        std::vector<vigra::MultiArray<1, LabelType> > & kfold_train_labels,
        std::vector<vigra::MultiArray<2, FeatureType> > & kfold_test_features,
        std::vector<vigra::MultiArray<1, LabelType> > & kfold_test_labels
){
    using namespace std;
    using namespace vigra;

    vigra_precondition(features.shape()[0] == labels.size(), "create_real_kfolds(): Shape mismatch.");

    size_t const num_features = features.shape()[1];

    vector<size_t> indices(features.shape()[0]);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    for (size_t iii = 0; iii < n_kfolds; ++iii)
    {
        size_t const test_begin = indices.size() * static_cast<double>(iii) / n_kfolds;
        size_t const test_end = indices.size() * static_cast<double>(iii+1) / n_kfolds;
        vector<size_t> test_indices(indices.begin()+test_begin, indices.begin()+test_end);
        vector<size_t> train_indices(indices.begin(), indices.end());
        train_indices.erase(train_indices.begin()+test_begin, train_indices.begin()+test_end);
        sort(test_indices.begin(), test_indices.end());
        sort(train_indices.begin(), train_indices.end());

        // Build the training set.
        kfold_train_features.push_back(MultiArray<2, FeatureType>(Shape2(train_indices.size(), num_features)));
        kfold_train_labels.push_back(MultiArray<1, LabelType>(Shape1(train_indices.size())));
        auto & train_x = kfold_train_features.back();
        auto & train_y = kfold_train_labels.back();
        for (size_t y = 0; y < num_features; ++y)
        {
            for (size_t j = 0; j < train_indices.size(); ++j)
            {
                train_x(j, y) = features(train_indices[j], y);
            }
        }
        for (size_t j = 0; j < train_indices.size(); ++j)
        {
            train_y(j) = labels(train_indices[j]);
        }

        // Build the test set.
        kfold_test_features.push_back(MultiArray<2, FeatureType>(Shape2(test_indices.size(), num_features)));
        kfold_test_labels.push_back(MultiArray<1, LabelType>(Shape1(test_indices.size())));
        auto & test_x = kfold_test_features.back();
        auto & test_y = kfold_test_labels.back();
        for (size_t y = 0; y < num_features; ++y)
        {
            for (size_t j = 0; j < test_indices.size(); ++j)
            {
                test_x(j, y) = features(test_indices[j], y);
            }
        }
        for (size_t j = 0; j < test_indices.size(); ++j)
        {
            test_y(j) = labels(test_indices[j]);
        }
    }
}



template <typename ARR0, typename ARR1>
size_t count_equal_values(
        ARR0 const & a,
        ARR1 const & b
){
    if (a.size() != b.size())
        throw std::runtime_error("count_equal_values(): Shape mismatch.");
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a(i) == b(i))
            ++count;
    return count;
}



#endif
