#include <iostream>
#include <map>
#include <vector>

#include <vigra/multi_array.hxx>
#include <vigra/graphs_new.hxx>
#include <vigra/random_forest_new.hxx>

using namespace std;
using namespace vigra;

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
    typedef RandomForest<double, int, SplitTest, Acc> RF;

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
    RF rf = RF(gr, split_tests, leaf_responses, {0, 1, -7, 3}, 2);

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
    cout << "test_default_random_forest(): Success!" << endl;
}

int main()
{
    test_binary_directed_graph();
    test_property_map();
    test_random_forest_class();
    test_default_random_forest();
}

