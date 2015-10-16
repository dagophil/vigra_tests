#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <set>
#include <mutex>
#include <fstream>
#include <ostream>
#include <chrono>
#include <type_traits>
#include <sstream>

#include <vigra/random_forest_new/forest_garrote.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/sampling.hxx>
#include <vigra/pool.hxx>
#include <vigra/random_forest_new.hxx>

using namespace std;
using namespace vigra;

static string const logfilename = "PARAMETERLOG.txt";

template <typename FEATURES, typename LABELS>
void load_neuro_data(FEATURES & features, LABELS & labels)
{
    //string const data_x_filename = "/home/philip/data/neuro/train/ffeat_br_segid0.h5";
    string const data_x_filename = "/home/pschill/data/neuro/train/ffeat_br_segid0.h5";
    string const data_x_h5key = "ffeat_br";
    //string const data_y_filename = "/home/philip/data/neuro/train/gt_face_segid0.h5";
    string const data_y_filename = "/home/pschill/data/neuro/train/gt_face_segid0.h5";
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
        size_t p_resample_count = 0,
        string p_weight_method = "forest_garrote",
        size_t p_id = 0
    )   :
        n_trees(p_n_trees),
        split(p_split),
        bootstrap_sampling(p_bootstrap_sampling),
        resample_count(p_resample_count),
        weight_method(p_weight_method),
        id(p_id)
    {}
    size_t n_trees;
    string split;
    bool bootstrap_sampling;
    size_t resample_count;
    string weight_method;
    size_t id;
};

vector<RFParam> create_params()
{
    vector<size_t> vec_n_trees = {1, 2, 4, 8, 16, 32, 64, 128};
    //vector<string> vec_split = {"gini", "ksd", "entropy"};
    vector<string> vec_split = {"gini", "ksd"};
    //vector<size_t> vec_resample_count = {8, 16, 32, 64, 128, 256, 512, 1024};
    vector<size_t> vec_resample_count = {0, 16, 32, 64, 128, 256, 512};
    //vector<bool> vec_bootstrap_sampling = {true, false};
    vector<string> vec_weight_method = {"forest_garrote", "l1_svm", "l2_svm"};

    vector<RFParam> params;
    for (auto n_trees : vec_n_trees)
    {
        for (auto split : vec_split)
        {
            for (auto resample_count : vec_resample_count)
            {
                for (auto weight_method : vec_weight_method)
                {
                    params.push_back({n_trees, split, false, resample_count, weight_method, params.size()});
                }
            }
        }
    }
    return params;
}

class Result
{
public:
    double train_time;
    double predict_time;
    double performance;
    double num_nodes;
    double split_counts;

    Result(double p_train_time = 0, double p_predict_time = 0, double p_performance = 0, double p_num_nodes = 0, double p_split_counts = 0)
        :
        train_time(p_train_time),
        predict_time(p_predict_time),
        performance(p_performance),
        num_nodes(p_num_nodes),
        split_counts(p_split_counts)
    {}

    Result & operator+=(Result const & other)
    {
        train_time += other.train_time;
        predict_time += other.predict_time;
        performance += other.performance;
        num_nodes += other.num_nodes;
        split_counts += other.split_counts;
    }

    Result & operator-=(Result const & other)
    {
        train_time -= other.train_time;
        predict_time -= other.predict_time;
        performance -= other.performance;
        num_nodes -= other.num_nodes;
        split_counts -= other.split_counts;
    }

    Result & operator*=(Result const & other)
    {
        train_time *= other.train_time;
        predict_time *= other.predict_time;
        performance *= other.performance;
        num_nodes *= other.num_nodes;
        split_counts *= other.split_counts;
    }

    Result & operator*=(double d)
    {
        train_time *= d;
        predict_time *= d;
        performance *= d;
        num_nodes *= d;
        split_counts *= d;
    }

    Result & operator/=(double d)
    {
        train_time /= d;
        predict_time /= d;
        performance /= d;
        num_nodes /= d;
        split_counts /= d;
    }

    Result & operator/=(Result const & other)
    {
        train_time /= other.train_time;
        predict_time /= other.predict_time;
        performance /= other.performance;
        num_nodes /= other.num_nodes;
        split_counts /= other.split_counts;
    }
};

Result operator+(Result a, Result const & b)
{
    a += b;
    return a;
}

Result operator-(Result a, Result const & b)
{
    a -= b;
    return a;
}

Result operator*(Result a, Result const & b)
{
    a *= b;
    return a;
}

Result operator*(Result a, double b)
{
    a *= b;
    return a;
}

Result operator/(Result a, Result const & b)
{
    a /= b;
    return a;
}

Result operator/(Result a, double b)
{
    a /= b;
    return a;
}

ostream & operator<<(ostream & out, Result const & a)
{
    out << a.train_time << " " << a.predict_time << " " << a.performance << " " << a.num_nodes << " " << a.split_counts;
    return out;
}

Result sqrt(Result a)
{
    a.train_time = sqrt(a.train_time);
    a.predict_time = sqrt(a.predict_time);
    a.performance = sqrt(a.performance);
    a.num_nodes = sqrt(a.num_nodes);
    a.split_counts = sqrt(a.split_counts);
    return a;
}



template <typename FEATURES, typename LABELS, typename SPLIT>
void do_test_impl(
    size_t thread_id,
    vector<FEATURES> const & kfold_train_features,
    vector<LABELS> const & kfold_train_labels,
    vector<FEATURES> const & kfold_test_features,
    vector<LABELS> const & kfold_test_labels,
    RFParam const & params,
    mutex & mex
){
    typedef typename FEATURES::value_type FeatureType;
    typedef typename LABELS::value_type LabelType;

    chrono::steady_clock::time_point starttime, endtime;
    double sec;
    size_t const n_kfolds = kfold_train_features.size();
    RandomForestOptions const opts = RandomForestOptions().tree_count(params.n_trees).bootstrap_sampling(params.bootstrap_sampling).resample_count(params.resample_count);

    vector<pair<Result, Result> > results;
    for (size_t i = 0; i < n_kfolds; ++i)
    {
        auto const & train_x = kfold_train_features[i];
        auto const & train_y = kfold_train_labels[i];
        auto const & test_x = kfold_test_features[i];
        auto const & test_y = kfold_test_labels[i];

        Result rf_result;
        Result fg_result;

        // Train the random forest.
        starttime = chrono::steady_clock::now();
        auto const rf = random_forest<FEATURES, LABELS, SPLIT, PurityStop, ArgMaxVectorAcc<size_t> >(
            train_x,
            train_y,
            opts,
            1
        );
        endtime = chrono::steady_clock::now();
        sec = chrono::duration<double>(endtime-starttime).count();
        rf_result.train_time = sec;
        rf_result.num_nodes = rf.num_nodes();

        // Predict with the random forest.
        {
            MultiArray<1, LabelType> pred(Shape1(test_y.size()));
            starttime = chrono::steady_clock::now();
            rf_result.split_counts = rf.predict(test_x, pred, 1);
            endtime = chrono::steady_clock::now();
            sec = chrono::duration<double>(endtime-starttime).count();
            rf_result.predict_time = sec;

            // Find the performance.
            size_t count = 0;
            for (size_t i = 0; i < test_y.size(); ++i)
                if (pred(i) == test_y(i))
                    ++count;
            rf_result.performance = static_cast<double>(count) / test_y.size();
        }

        // Apply the forest garrote.
        string fg_filename = "/mnt/CLAWS1/pschill/tmp/fg_" + to_string(thread_id) + ".h5";
        starttime = chrono::steady_clock::now();
        auto const fg = forest_garrote(rf, train_x, train_y, 1, fg_filename, 0.0001, params.weight_method);
        endtime = chrono::steady_clock::now();
        sec = chrono::duration<double>(endtime-starttime).count();
        fg_result.train_time = sec;
        fg_result.num_nodes = fg.num_nodes();

        // Predict with the forest garrote.
        {
            MultiArray<1, LabelType> pred(Shape1(test_y.size()));
            starttime = chrono::steady_clock::now();
            fg_result.split_counts = fg.predict(test_x, pred, 1);
            endtime = chrono::steady_clock::now();
            sec = chrono::duration<double>(endtime-starttime).count();
            fg_result.predict_time = sec;

            // Find the performance.
            size_t count = 0;
            for (size_t i = 0; i < test_y.size(); ++i)
                if (pred(i) == test_y(i))
                    ++count;
            fg_result.performance = static_cast<double>(count) / test_y.size();
        }

        results.push_back({rf_result, fg_result});
    }

    // Find the mean of the results.
    Result mean_rf_result;
    Result mean_fg_result;
    for (auto const & p : results)
    {
        mean_rf_result += p.first;
        mean_fg_result += p.second;
    }
    mean_rf_result /= static_cast<double>(n_kfolds);
    mean_fg_result /= static_cast<double>(n_kfolds);

    // Find the standard deviation of the result.
    Result std_rf_result;
    Result std_fg_result;
    for (auto const & p : results)
    {
        std_rf_result += (p.first - mean_rf_result) * (p.first - mean_rf_result);
        std_fg_result += (p.second - mean_fg_result) * (p.second - mean_fg_result);
    }
    std_rf_result /= static_cast<double>(n_kfolds-1.0);
    std_fg_result /= static_cast<double>(n_kfolds-1.0);
    std_rf_result = sqrt(std_rf_result);
    std_fg_result = sqrt(std_fg_result);

    // Write the result.
    stringstream ss;
    ss << "# Parameter " + to_string(params.id) + "\n";
    ss << "{n_trees: " + to_string(params.n_trees) + ", bootstrap_sampling: " + to_string(params.bootstrap_sampling) + ", resample_count: " + to_string(params.resample_count) + ", split: \"" + params.split + "\", weight_method: \"" + params.weight_method + "\"}\n";
    ss << "RF_mean: " << mean_rf_result << "\n";
    ss << "RF_std: " << std_rf_result << "\n";
    ss << "FG_mean: " << mean_fg_result << "\n";
    ss << "FG_std: " << std_fg_result << "\n";

    {
        lock_guard<mutex> lock(mex);
        ofstream f(logfilename, ios::app);
        f << ss.str();
        f.close();
    }
}

template <typename FEATURES, typename LABELS>
void do_test(
    size_t thread_id,
    vector<FEATURES> const & kfold_train_features,
    vector<LABELS> const & kfold_train_labels,
    vector<FEATURES> const & kfold_test_features,
    vector<LABELS> const & kfold_test_labels,
    RFParam const & params,
    mutex & mex
){
    // Forward the split as template argument.
    if (params.split == "gini")
        do_test_impl<FEATURES, LABELS, GiniScorer>(thread_id, kfold_train_features, kfold_train_labels, kfold_test_features, kfold_test_labels, params, mex);
    else if (params.split == "ksd")
        do_test_impl<FEATURES, LABELS, KSDScorer>(thread_id, kfold_train_features, kfold_train_labels, kfold_test_features, kfold_test_labels, params, mex);
    else if (params.split == "entropy")
        do_test_impl<FEATURES, LABELS, EntropyScorer>(thread_id, kfold_train_features, kfold_train_labels, kfold_test_features, kfold_test_labels, params, mex);
    else
        std::cout << "Warning: Unknown split: " << params.split << std::endl;
}

template <typename FeatureType, typename LabelType>
void create_kfolds(
        size_t n_kfolds,
        MultiArray<2, FeatureType> const & features,
        MultiArray<1, LabelType> const & labels,
        vector<MultiArray<2, FeatureType> > & kfold_train_features,
        vector<MultiArray<1, LabelType> > & kfold_train_labels,
        vector<MultiArray<2, FeatureType> > & kfold_test_features,
        vector<MultiArray<1, LabelType> > & kfold_test_labels
){
    vigra_precondition(features.shape()[0] == labels.size(), "create_kfolds(): Shape mismatch.");

    size_t const num_features = features.shape()[1];
    Sampler<> sampler(features.shape()[0], SamplerOptions().withoutReplacement().sampleProportion(1.0/n_kfolds));
    for (size_t i = 0; i < n_kfolds; ++i)
    {
        sampler.sample();
        size_t const num_instances = sampler.sampledIndices().size();

        // Fill the test set.
        std::vector<bool> keep(labels.size(), true);
        {
            kfold_test_features.push_back(MultiArray<2, FeatureType>(Shape2(num_instances, num_features)));
            kfold_test_labels.push_back(MultiArray<1, LabelType>(Shape1(num_instances)));
            auto & current_features = kfold_test_features.back();
            auto & current_labels = kfold_test_labels.back();
            for (size_t j = 0; j < sampler.sampleSize(); ++j)
            {
                current_labels(j) = labels(sampler[j]);
                keep[sampler[j]] = false;
            }
            for (size_t y = 0; y < num_features; ++y)
            {
                for (size_t j = 0; j < sampler.sampleSize(); ++j)
                {
                    current_features(j, y) = features(sampler[j], y);
                }
            }
        }

        // Fill the training set.
        size_t const keep_count = std::count(keep.begin(), keep.end(), true);
        {
            kfold_train_features.push_back(MultiArray<2, FeatureType>(Shape2(keep_count, num_features)));
            kfold_train_labels.push_back(MultiArray<1, LabelType>(Shape1(keep_count)));
            auto & current_features = kfold_train_features.back();
            auto & current_labels = kfold_train_labels.back();
            size_t j = 0;
            for (size_t i = 0; i < labels.size(); ++i)
            {
                if (keep[i])
                {
                    current_labels(j) = labels(i);
                    ++j;
                }
            }
            for (size_t y = 0; y < num_features; ++y)
            {
                j = 0;
                for (size_t i = 0; i < features.shape()[0]; ++i)
                {
                    if (keep[i])
                    {
                        current_features(j, y) = features(i, y);
                        ++j;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    typedef float FeatureType;
    typedef UInt32 LabelType;

    size_t const n_threads = argc <= 1 ? 4 : stoi(argv[1]);
    size_t const n_kfolds = argc <= 2 ? 10 : stoi(argv[2]);

    // Get the data.
    MultiArray<2, FeatureType> data_x;
    MultiArray<1, LabelType> data_y;
    load_neuro_data(data_x, data_y);
    size_t const num_features = data_x.shape()[1];

    // Create the kfolds.
    vector<MultiArray<2, FeatureType> > kfold_train_features;
    vector<MultiArray<1, LabelType> > kfold_train_labels;
    vector<MultiArray<2, FeatureType> > kfold_test_features;
    vector<MultiArray<1, LabelType> > kfold_test_labels;
    create_kfolds(n_kfolds, data_x, data_y, kfold_train_features, kfold_train_labels, kfold_test_features, kfold_test_labels);

    // Get the params.
    vector<RFParam> params = create_params();
    //params.erase(params.begin(), params.begin()+213);

    // Create the write-to-file mutex.
    mutex mex;

    // Start the parametertest.
    inferno::utilities::ThreadPool pool(n_threads);
    for (auto const & p : params)
    {
        pool.enqueue([&](size_t thread_id)
            {
                do_test(thread_id, kfold_train_features, kfold_train_labels, kfold_test_features, kfold_test_labels, p, mex);
            }
        );
    }
    pool.waitFinished();
}
