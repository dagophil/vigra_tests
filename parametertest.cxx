#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <set>

#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/sampling.hxx>
#include <vigra/pool.hxx>

using namespace std;
using namespace vigra;

template <typename FEATURES, typename LABELS>
void load_neuro_data(FEATURES & features, LABELS & labels)
{
    string const data_x_filename = "/home/philip/data/neuro/train/ffeat_br_segid0.h5";
    string const data_x_h5key = "ffeat_br";
    string const data_y_filename = "/home/philip/data/neuro/train/gt_face_segid0.h5";
    string const data_y_h5key = "gt_face";

    HDF5File data_x_file(data_x_filename, HDF5File::ReadWrite);
    data_x_file.readAndResize(data_x_h5key, features);
    features = features.transpose();

    HDF5File data_y_file(data_y_filename, HDF5File::ReadWrite);
    data_y_file.readAndResize(data_y_h5key, labels);
}

struct RFParam
{
public:
    RFParam(
        size_t p_n_trees = 100,
        string p_split = "gini",
        bool p_bootstrap_sampling = true,
        size_t p_resample_count = 0
    )   :
        n_trees(p_n_trees),
        split(p_split),
        bootstrap_sampling(p_bootstrap_sampling),
        resample_count(p_resample_count)
    {}
    size_t n_trees;
    string split;
    bool bootstrap_sampling;
    size_t resample_count;
};

vector<RFParam> create_params()
{
    vector<size_t> vec_n_trees = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    vector<string> vec_split = {"gini", "ksd", "entropy"};
    vector<size_t> vec_resample_count = {8, 16, 32, 64, 128, 256, 512, 1024};
    vector<bool> vec_bootstrap_sampling = {true, false};

    vector<RFParam> params;
    for (auto n_trees : vec_n_trees)
    {
        for (auto split : vec_split)
        {
            for (auto resample_count : vec_resample_count)
            {
                params.push_back({n_trees, split, false, resample_count});
            }
            for (auto bootstrap_sampling : vec_bootstrap_sampling)
            {
                params.push_back({n_trees, split, bootstrap_sampling, 0});
            }
        }
    }
    return params;
}



template <typename FEATURES, typename LABELS>
void do_test(
    vector<FEATURES> const & kfold_features,
    vector<LABELS> const & kfold_labels,
    RFParam const & params
){
    cout << "do_test(): Not implemented yet." << endl;
}



int main()
{
    typedef float FeatureType;
    typedef UInt32 LabelType;

    size_t const n_threads = 4;
    size_t const n_kfolds = 10;

    // Get the data.
    MultiArray<2, FeatureType> data_x;
    MultiArray<1, LabelType> data_y;
    load_neuro_data(data_x, data_y);
    size_t const num_features = data_x.shape()[1];

    // Create the kfolds.
    vector<MultiArray<2, FeatureType> > kfold_features;
    vector<MultiArray<1, LabelType> > kfold_labels;
    Sampler<> sampler(data_x.shape()[0], SamplerOptions().withoutReplacement().sampleProportion(1.0/n_kfolds));
    for (size_t i = 0; i < n_kfolds; ++i)
    {
        sampler.sample();
        size_t const num_instances = sampler.sampledIndices().size();
        kfold_features.push_back(MultiArray<2, FeatureType>(Shape2(num_instances, num_features)));
        auto & current_data = kfold_features.back();
        for (size_t y = 0; y < num_features; ++y)
        {
            size_t j = 0;
            for (auto x : sampler.sampledIndices())
            {
                current_data(j, y) = data_x(x, y);
                ++j;
            }
        }
    }

    // Get the params.
    vector<RFParam> params = create_params();

    // Start the parametertest.
    inferno::utilities::ThreadPool pool(n_threads);
    for (auto const & p : params)
    {
        pool.enqueue([&kfold_features, &kfold_labels, &p](size_t thread_id)
            {
                do_test(kfold_features, kfold_labels, p);
            }
        );
    }
    pool.waitFinished();

    cout << "done" << endl;
}
