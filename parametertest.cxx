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

#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/sampling.hxx>
#include <vigra/threadpool.hxx>
#include <vigra/random_forest_new.hxx>

#include "data_utility.hxx"

using namespace std;
using namespace vigra;

static string logfilename = "";

struct RFParam
{
public:
    RFParam(
        RandomForestNewOptions p_options = RandomForestNewOptions(),
        size_t p_id = 0
    )   :
        options(p_options),
        id(p_id)
    {}
    RandomForestNewOptions options;
    size_t id;
};

vector<RFParam> create_stopping_params()
{
    // vector<double> taus = {1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21, 1e-24, 1e-27, 1e-30, 1e-33, 1e-36, 1e-39, 1e-42, 1e-45, 1e-48};
    // vector<size_t> min_num_instances = {1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 140, 180, 220, 260, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000};
    // vector<size_t> depths = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    vector<double> taus = {3e-1, 1e-1, 3e-2, 1e-2, 3e-3};
    vector<size_t> depths = {30};
    vector<size_t> min_num_instances = {2, 3, 4};

    vector<RFParam> params;
    for (auto tau : taus)
    {
        params.push_back({RandomForestNewOptions().tree_count(100).node_complexity_tau(tau)});
        params.back().id = params.size()-1;
    }
    for (auto depth : depths)
    {
        params.push_back({RandomForestNewOptions().tree_count(100).max_depth(depth)});
        params.back().id = params.size()-1;
    }
    for (auto n : min_num_instances)
    {
        params.push_back({RandomForestNewOptions().tree_count(100).min_num_instances(n)});
        params.back().id = params.size()-1;
    }
    return params;
}

vector<RFParam> create_ksd_params()
{
    vector<size_t> n_trees = {4, 8, 16, 32, 64, 128};
    vector<RFParam> params;
    for (auto n : n_trees)
    {
        params.push_back({RandomForestNewOptions().tree_count(n).split(RF_GINI), params.size()});
        params.push_back({RandomForestNewOptions().tree_count(n).split(RF_ENTROPY), params.size()});
        params.push_back({RandomForestNewOptions().tree_count(n).split(RF_KSD), params.size()});
    }
    return params;
}

class Result
{
public:
    double performance;
    double num_nodes;
    double split_counts;
    bool valid;

    Result(double p_performance = 0, double p_num_nodes = 0, double p_split_counts = 0, bool p_valid = true)
        :
        performance(p_performance),
        num_nodes(p_num_nodes),
        split_counts(p_split_counts),
        valid(p_valid)
    {}

    Result & operator+=(Result const & other)
    {
        performance += other.performance;
        num_nodes += other.num_nodes;
        split_counts += other.split_counts;
    }

    Result & operator-=(Result const & other)
    {
        performance -= other.performance;
        num_nodes -= other.num_nodes;
        split_counts -= other.split_counts;
    }

    Result & operator*=(Result const & other)
    {
        performance *= other.performance;
        num_nodes *= other.num_nodes;
        split_counts *= other.split_counts;
    }

    Result & operator*=(double d)
    {
        performance *= d;
        num_nodes *= d;
        split_counts *= d;
    }

    Result & operator/=(double d)
    {
        performance /= d;
        num_nodes /= d;
        split_counts /= d;
    }

    Result & operator/=(Result const & other)
    {
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
    out << "performance: " << a.performance << ", num_nodes: " << a.num_nodes << ", split_counts:" << a.split_counts;
    return out;
}

Result sqrt(Result a)
{
    a.performance = sqrt(a.performance);
    a.num_nodes = sqrt(a.num_nodes);
    a.split_counts = sqrt(a.split_counts);
    return a;
}

Result mean(vector<Result> const & v)
{
    Result mean_result;
    size_t c = 0;
    for (auto const & r : v)
    {
        if (r.valid)
        {
            mean_result += r;
            ++c;
        }
    }
    mean_result /= static_cast<double>(c);
    return mean_result;
}

Result std_dev(vector<Result> const & v)
{
    Result std_result;
    auto m = mean(v);
    size_t c = 0;
    for (auto const & r : v)
    {
        if (r.valid)
        {
            std_result += (r - m) * (r - m);
            ++c;
        }
    }
    std_result /= (c-1.0);
    std_result = sqrt(std_result);
    return std_result;
}


template <typename FEATURES, typename LABELS>
void do_test_stopping_criteria(
    size_t thread_id,
    vector<FEATURES> const & kfold_train_x,
    vector<LABELS> const & kfold_train_y,
    vector<FEATURES> const & kfold_test_x,
    vector<LABELS> const & kfold_test_y,
    RFParam const & params,
    mutex & mex
){
    typedef typename FEATURES::value_type FeatureType;
    typedef typename LABELS::value_type LabelType;

    cout << "running param " << params.id << endl;

    size_t const n_kfolds = kfold_train_x.size();
    RandomForestNewOptions const opts = params.options;

    vector<Result> rf_results;

    for (size_t i = 0; i < n_kfolds; ++i)
    {
        auto const & train_x = kfold_train_x[i];
        auto const & train_y = kfold_train_y[i];
        auto const & test_x = kfold_test_x[i];
        auto const & test_y = kfold_test_y[i];

        // Get the random forest results.
        Result rf_result;
        auto const rf = random_forest<FEATURES, LABELS>(
            train_x, train_y, opts, 1
        );
        rf_result.num_nodes = rf.num_nodes();
        LABELS pred(Shape1(test_y.size()));
        rf_result.split_counts = rf.predict(test_x, pred, 1);
        size_t const count = count_equal_values(test_y, pred);
        rf_result.performance = static_cast<double>(count) / test_y.size();
        rf_results.push_back(rf_result);
    }

    // Find the mean and standard deviation of the results.
    Result mean_rf_result = mean(rf_results);
    Result std_rf_result = std_dev(rf_results);

    // Write the results to the file.
    stringstream ss;
    ss << "# Parameter " + to_string(params.id) + "\n";
    ss << "n_trees: " + to_string(params.options.tree_count_) + 
          ", tau: " + to_string((int)std::round(std::log10(params.options.node_complexity_tau_))) + 
          ", max_depth: " + to_string(params.options.max_depth_) + 
          ", min_num_instances:" + to_string(params.options.min_num_instances_) + "\n";
    ss << "RF_mean: " << mean_rf_result << "\n";
    ss << "RF_std: " << std_rf_result << "\n";

    {
        lock_guard<mutex> lock(mex);
        ofstream f(logfilename, ios::app);
        f << ss.str();
        f.close();
    }
}

template <typename FEATURES, typename LABELS>
void do_test_ksd(
    size_t thread_id,
    vector<FEATURES> const & kfold_train_x,
    vector<LABELS> const & kfold_train_y,
    vector<FEATURES> const & kfold_test_x,
    vector<LABELS> const & kfold_test_y,
    RFParam const & params,
    mutex & mex
){
    typedef typename FEATURES::value_type FeatureType;
    typedef typename LABELS::value_type LabelType;

    cout << "running param " << params.id << endl;
    std::map<RandomForestOptionTags, string> split_to_string = {
        {RF_GINI, "gini"},
        {RF_ENTROPY, "entropy"},
        {RF_KSD, "ksd"}
    };

    size_t const n_kfolds = kfold_train_x.size();
    RandomForestNewOptions const opts = params.options;

    vector<Result> rf_results;

    for (size_t i = 0; i < n_kfolds; ++i)
    {
        auto const & train_x = kfold_train_x[i];
        auto const & train_y = kfold_train_y[i];
        auto const & test_x = kfold_test_x[i];
        auto const & test_y = kfold_test_y[i];

        // Get the random forest results.
        Result rf_result;
        auto const rf = random_forest<FEATURES, LABELS>(
            train_x, train_y, opts, 1
        );
        rf_result.num_nodes = rf.num_nodes();
        LABELS pred(Shape1(test_y.size()));
        rf_result.split_counts = rf.predict(test_x, pred, 1);
        size_t const count = count_equal_values(test_y, pred);
        rf_result.performance = static_cast<double>(count) / test_y.size();
        rf_results.push_back(rf_result);
    }

    // Find the mean and standard deviation of the results.
    Result mean_rf_result = mean(rf_results);
    Result std_rf_result = std_dev(rf_results);

    // Write the results to the file.
    stringstream ss;
    ss << "# Parameter " + to_string(params.id) + "\n";
    ss << "n_trees: " + to_string(params.options.tree_count_) + ", split: " << split_to_string[params.options.split_] + "\n";
    ss << "RF_mean: " << mean_rf_result << "\n";
    ss << "RF_std: " << std_rf_result << "\n";

    {
        lock_guard<mutex> lock(mex);
        ofstream f(logfilename, ios::app);
        f << ss.str();
        f.close();
    }
}



int main(int argc, char** argv)
{
    typedef float FeatureType;
    typedef UInt32 LabelType;

    if (argc <= 1)
        throw runtime_error("You must specify the output filename as first command line argument.");
    logfilename = string(argv[1]);
    if (logfilename.size() < 4)
        throw runtime_error("The filename should at least have length 4.");

    size_t const n_threads = argc <= 2 ? 4 : stoi(argv[2]);
    size_t const n_kfolds = argc <= 3 ? 10 : stoi(argv[3]);

    // string const features_filename = "/home/pschill/data/neuro/train/ffeat_br_segid0.h5";
    // string const labels_filename = "/home/pschill/data/neuro/train/gt_face_segid0.h5";
    // string const features_key = "ffeat_br";
    // string const labels_key = "gt_face";
    string const features_filename = "/home/pschill/data/mnist/mnist_train_reshaped.h5";
    string const labels_filename = "/home/pschill/data/mnist/mnist_train_reshaped.h5";
    string const features_key = "images";
    string const labels_key = "labels";

    // Get the data.
    MultiArray<2, FeatureType> data_x;
    MultiArray<1, LabelType> data_y;
    HDF5File hfile = HDF5File(features_filename, HDF5File::ReadOnly);
    hfile.readAndResize(features_key, data_x);
    data_x = data_x.transpose();
    hfile = HDF5File(labels_filename, HDF5File::ReadOnly);
    hfile.readAndResize(labels_key, data_y);

    // Create the kfolds.
    vector<MultiArray<2, FeatureType> > kfold_train_features, kfold_test_features;
    vector<MultiArray<1, LabelType> > kfold_train_labels, kfold_test_labels;
    create_real_kfolds(n_kfolds,
                       data_x,
                       data_y,
                       kfold_train_features,
                       kfold_train_labels,
                       kfold_test_features,
                       kfold_test_labels);

    // Get the params.
    // vector<RFParam> params = create_params();
    //params.erase(params.begin()+1, params.end());
    //params.erase(params.begin(), params.begin()+213);
    vector<RFParam> params = create_stopping_params();

    // Create the write-to-file mutex.
    mutex mex;

    // Start the parametertest.
    ThreadPool pool(n_threads);
    for (auto const & p : params)
    {
        pool.enqueue([&](size_t thread_id)
            {
                do_test_stopping_criteria(
                        thread_id,
                        kfold_train_features,
                        kfold_train_labels,
                        kfold_test_features,
                        kfold_test_labels,
                        p,
                        mex);
            }
        );
    }
    pool.waitFinished();
}
