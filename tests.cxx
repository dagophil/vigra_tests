#include <iostream>
#include <map>
#include <vector>
#include <chrono>

#include <vigra/multi_array.hxx>
#include <vigra/graphs_new.hxx>
#include <vigra/random_forest_new.hxx>
#include <vigra/random_forest_new/random_forest_common.hxx>

#include "data_utility.hxx"

using namespace std;
using namespace vigra;

// TIC TOC macros to measure time.
chrono::steady_clock::time_point starttime, endtime;
double sec;
#define TIC starttime = chrono::steady_clock::now();
#define TOC(msg) endtime = chrono::steady_clock::now(); sec = chrono::duration<double>(endtime-starttime).count(); cout << msg << ": " << sec << " seconds" << endl;

void test_permutation_iterator()
{
    vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<size_t> perm = {5, 2, 3, 3, 8};
    vector<int> expected = {6, 3, 4, 4, 9};
    auto begin = make_permutation_iterator(vec.begin(), perm.begin());
    auto end = make_permutation_iterator(vec.begin(), perm.end());
    vector<int> result(begin, end);

    cout << "test_permutation_iterator(): Success!" << endl;
}

void test_binary_directed_graph()
{

    typedef BinaryDirectedGraph Graph;
    typedef Graph::Node Node;
    typedef Graph::Arc Arc;

    Graph gr;
    Node n0 = gr.addNode();
    Node n1 = gr.addNode();
    Node n2 = gr.addNode();
    Node n3 = gr.addNode();
    Node n4 = gr.addNode();
    Arc a01 = gr.addArc(n0, n1);
    Arc a02 = gr.addArc(n0, n2);
    Arc a13 = gr.addArc(n1, n3);
    Arc a14 = gr.addArc(n1, n4);
    Arc a24 = gr.addArc(n2, n4);

    // Check numNodes, numArcs, nmaxNodeId, maxArcId
    {
        vigra_assert(gr.numNodes() == 5, "Error in numNodes.");
        vigra_assert(gr.numArcs() == 5, "Error in numArcs.");
        gr.addArc(n1, n4);
        vigra_assert(gr.numArcs() == 5, "Error in addArc or numArcs.");
        vigra_assert(gr.maxNodeId() == 4, "Error in maxNodeId.");
        vigra_assert(gr.maxArcId() == 9, "Error in maxArcId.");
    }

    // Check source and target.
    {
        vigra_assert(gr.source(a01) == n0 && gr.source(a02) == n0 && gr.source(a13) == n1 &&
                     gr.source(a14) == n1 && gr.source(a24) == n2, "Error in source.");
        vigra_assert(gr.target(a01) == n1 && gr.target(a02) == n2 && gr.target(a13) == n3 &&
                     gr.target(a14) == n4 && gr.target(a24) == n4, "Error in target.");
    }

    // Check valid.
    {
        vigra_assert(gr.valid(n0) && gr.valid(n1) && gr.valid(n2) && gr.valid(n3) && gr.valid(n4),
                     "Error in gr.valid(Node).");
        vigra_assert(gr.valid(a01) && gr.valid(a02) && gr.valid(a13) && gr.valid(a14) && gr.valid(a24),
                     "Error in gr.valid(Arc).");
        vigra_assert(!gr.valid(Node(-1)) && !gr.valid(Node(5)), "Error in valid(Node).");
        vigra_assert(!gr.valid(Arc(-1)) && !gr.valid(Arc(6)), "Error in valid(Arc).");
    }

    // Check inDegree and outDegree.
    {
        vigra_assert(gr.inDegree(n0) == 0 && gr.inDegree(n1) == 1 && gr.inDegree(n2) == 1 &&
                     gr.inDegree(n3) == 1 && gr.inDegree(n4) == 2, "Error in inDegree.");
        vigra_assert(gr.outDegree(n0) == 2 && gr.outDegree(n1) == 2 && gr.outDegree(n2) == 1 &&
                     gr.outDegree(n3) == 0 && gr.outDegree(n4) == 0, "Error in outDegree.");
    }

    // Check getParent, getChild, getRoot.
    {
        vigra_assert(gr.getParent(n0) == lemon::INVALID && gr.getParent(n1) == n0 &&
                     gr.getParent(n2) == n0 && gr.getParent(n3) == n1 &&
                     gr.getParent(n4, 0) == n1 && gr.getParent(n4, 1) == n2, "Error in getParent.");
        vigra_assert(gr.getChild(n0, 0) == n1 && gr.getChild(n0, 1) == n2 && gr.getChild(n1, 0) == n3 &&
                     gr.getChild(n1, 1) == n4 && gr.getChild(n2, 0) == n4 && gr.getChild(n3) == lemon::INVALID &&
                     gr.getChild(n4) == lemon::INVALID, "Error in getChild.");
        vigra_assert(gr.getRoot(0) == n0 && gr.getRoot(1) == lemon::INVALID, "Error in getRoot.");
    }

    // Check merge.
    {
        Graph gr2;
        Node n0 = gr2.addNode();
        Node n1 = gr2.addNode();
        Node n2 = gr2.addNode();
        Node n3 = gr2.addNode();
        gr2.addArc(n0, n1);
        gr2.addArc(n1, n2);
        gr2.addArc(n0, n3);

        gr2.merge(gr);

        vigra_assert(gr2.numNodes() == 9, "Error in merge, wrong number of nodes.");
        vigra_assert(gr2.numArcs() == 8, "Error in merge, wrong number of arcs.");
        vigra_assert(gr2.getChild(n0, 0) == n1 && gr2.getChild(n0, 1) == n3 && gr2.getChild(n1) == n2,
                     "Error in merge, old children are wrong.");
        vigra_assert(gr2.getParent(n1) == n0 && gr2.getParent(n2) == n1 && gr2.getParent(n3) == n0,
                     "Error in merge, old parents are wrong.");
        Node n4 = gr.nodeFromId(4);
        Node n5 = gr.nodeFromId(5);
        Node n6 = gr.nodeFromId(6);
        Node n7 = gr.nodeFromId(7);
        Node n8 = gr.nodeFromId(8);
        vigra_assert(gr2.getChild(n4, 0) == n5 && gr2.getChild(n4, 1) == n6 && gr2.getChild(n5, 0) == n7 &&
                     gr2.getChild(n5, 1) == n8 && gr2.getChild(n6, 0) == n8,
                     "Error in merge, new children are wrong.");
        vigra_assert(gr2.getParent(n5) == n4 && gr2.getParent(n6) == n4 && gr2.getParent(n7) == n5 &&
                     gr2.getParent(n8, 0) == n5 && gr2.getParent(n8, 1) == n6,
                     "Error in merge, new parents are wrong.");
        vigra_assert(gr2.numRoots() == 2, "Error in merge, number of root nodes is wrong.");
        vigra_assert(gr2.getRoot(0) == n0 && gr2.getRoot(1) == n4, "Error in merge, root nodes are wrong.");
    }

    cout << "test_binary_directed_graph(): Success!" << endl;
}

void test_property_map()
{
    typedef BinaryDirectedGraph::Node Node;

    Node n0(2);
    Node n1(5);
    Node n2(10);

    // Check PropertyMap<Node, int, MapTag>.
    {
        PropertyMap<Node, int, MapTag> map0;
        map0[n0] = 27;
        map0[n2] = 73;
        vigra_assert(map0[n0] == 27 && map0[n2] == 73, "Error in operator[].");
        vigra_assert(map0.at(n0) == 27 && map0.at(n2) == 73, "Error in at.");
        vector<Node> keys;
        vector<Node> keys_expected = {n0, n2};
        vector<int> values;
        vector<int> values_expected = {27, 73};
        for (auto const & p : map0)
        {
            keys.push_back(p.first);
            values.push_back(p.second);
        }
        vigra_assert(keys == keys_expected, "Error in range-based loop over PropertyMap.");
        vigra_assert(values == values_expected, "Error in range-based loop over PropertyMap.");
    }

    // Check PropertyMap<Node, int, VectorTag>.
    {
        PropertyMap<Node, int, VectorTag> map0;
        map0[n0] = 27;
        map0[n2] = 73;
        vigra_assert(map0[n0] == 27 && map0[n2] == 73, "Error in operator[].");
        vigra_assert(map0.at(n0) == 27 && map0.at(n2) == 73, "Error in at.");
        vector<Node> keys;
        vector<Node> keys_expected = {n0, n2};
        vector<int> values;
        vector<int> values_expected = {27, 73};
        for (auto const & p : map0)
        {
            keys.push_back(p.first);
            values.push_back(p.second);
        }
        vigra_assert(keys == keys_expected, "Error in range-based loop over PropertyMap.");
        vigra_assert(values == values_expected, "Error in range-based loop over PropertyMap.");
    }

    cout << "test_property_map(): Success!" << endl;
}

void test_random_forest_class()
{
    typedef BinaryDirectedGraph Graph;
    typedef Graph::Node Node;
    typedef LessEqualSplitTest<double> SplitTest;
    typedef ArgMaxAcc Acc;
    typedef RandomForest<MultiArray<2, double>, MultiArray<1, int>, SplitTest, Acc> RF;

    Graph gr;
    PropertyMap<Node, SplitTest> split_tests;
    PropertyMap<Node, size_t> leaf_responses;
    {
        Node n0 = gr.addNode();
        Node n1 = gr.addNode();
        Node n2 = gr.addNode();
        Node n3 = gr.addNode();
        Node n4 = gr.addNode();
        Node n5 = gr.addNode();
        Node n6 = gr.addNode();
        gr.addArc(n0, n1);
        gr.addArc(n0, n2);
        gr.addArc(n1, n3);
        gr.addArc(n1, n4);
        gr.addArc(n2, n5);
        gr.addArc(n2, n6);

        split_tests[n0] = SplitTest(0, 0.6);
        split_tests[n1] = SplitTest(1, 0.25);
        split_tests[n2] = SplitTest(1, 0.75);
        leaf_responses[n3] = 0;
        leaf_responses[n4] = 1;
        leaf_responses[n5] = 2;
        leaf_responses[n6] = 3;
    }
    PropertyMap<Node, std::vector<size_t> > node_distributions;
    RF rf = RF(gr, split_tests, node_distributions, leaf_responses, {0, 1, -7, 3}, 2);

    double test_x_values[] = {
        0.2, 0.4, 0.2, 0.4, 0.7, 0.8, 0.7, 0.8,
        0.2, 0.2, 0.7, 0.7, 0.2, 0.2, 0.8, 0.8
    };
    MultiArray<2, double> test_x(Shape2(8, 2), test_x_values);
    int test_y_values[] = {
        0, 0, 1, 1, -7, -7, 3, 3
    };
    MultiArray<1, int> test_y(Shape1(8), test_y_values);
    MultiArray<1, int> pred_y(Shape1(8));
    rf.predict(test_x, pred_y);
    vigra_assert(
        std::vector<int>(test_y.begin(), test_y.end()) == std::vector<int>(pred_y.begin(), pred_y.end()),
        "Error in RandomForest prediction."
    );

    cout << "test_random_forest(): Success!" << endl;
}

void test_default_random_forest()
{
    typedef MultiArray<2, double> Features;
    typedef MultiArray<1, int> Labels;

    int const n_threads = 1;
    int const n_trees = 1;

    double train_x_values[] = {
        0.2, 0.4, 0.2, 0.4, 0.7, 0.8, 0.7, 0.8,
        0.2, 0.2, 0.7, 0.7, 0.2, 0.2, 0.8, 0.8
    };
    Features train_x(Shape2(8, 2), train_x_values);
    int train_y_values[] = {
        0, 0, 1, 1, -7, -7, 3, 3
    };
    Labels train_y(Shape1(8), train_y_values);
    Features test_x(train_x);
    Labels test_y(train_y);

    RandomForestOptions options = RandomForestOptions().tree_count(n_trees).bootstrap_sampling(false);

    auto rf = random_forest<Features, Labels, GiniScorer>(train_x, train_y, options, n_threads);
    Labels pred_y(Shape1(8));
    rf.predict(test_x, pred_y, n_threads);
    vigra_assert(
        std::vector<int>(test_y.begin(), test_y.end()) == std::vector<int>(pred_y.begin(), pred_y.end()),
        "Error in RandomForest prediction."
    );

    cout << "test_default_random_forest(): Success!" << endl;
}

void test_random_forest_mnist()
{
    typedef double FeatureType;
    typedef UInt8 LabelType;
    typedef MultiArray<2, FeatureType> Features;
    typedef MultiArray<1, LabelType> Labels;

    int const n_threads = -1;
    int const n_trees = 10;
    string const train_filename = "/home/philip/data/ml-koethe/train.h5";
    string const test_filename = "/home/philip/data/ml-koethe/test.h5";
    vector<LabelType> labels = {3, 8};
    RandomForestOptions options = RandomForestOptions().tree_count(n_trees).bootstrap_sampling(true);

    // Load the data.
    Features train_x, test_x;
    Labels train_y, test_y;
    load_data(train_filename, test_filename, train_x, train_y, test_x, test_y, labels);

    // Train the random forest.
    // TIC;
    auto rf = random_forest<Features, Labels, GiniScorer>(train_x, train_y, options, n_threads);
    // TOC("Random forest training");

    // Predict with the forest.
    Labels pred_y(Shape1(test_y.size()));
    // TIC;
    rf.predict(test_x, pred_y, n_threads);
    // TOC("Random forest prediction");

    // Count the correct predicted instances.
    size_t count = 0;
    for (size_t i = 0; i < test_y.size(); ++i)
        if (pred_y(i) == test_y(i))
            ++count;
    double performance = count / (float)pred_y.size();
    // cout << "Performance: " << (count / ((float) pred_y.size())) << " (" << count << " of " << pred_y.size() << ")" << endl;
    vigra_assert(performance > 0.95, "Expected performance of random forest is too low.");
    std::cout << "test_random_forest_mnist(): Success!" << std::endl;
}

int main()
{
    test_permutation_iterator();
    test_binary_directed_graph();
    test_property_map();
    test_random_forest_class();
    test_default_random_forest();
    test_random_forest_mnist();
}

