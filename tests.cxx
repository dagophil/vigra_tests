#include <iostream>
#include <map>
#include <vector>

#include <vigra/graphs_new.hxx>
#include <vigra/random_forest_new.hxx>

using namespace std;

void test_binary_directed_graph()
{
	using namespace vigra;

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


int main()
{
	test_binary_directed_graph();
	exit(0);





	typedef vigra::BinaryDirectedGraph Graph;
	typedef Graph::Node Node;

    Graph gr;

    Node n0(2);
    Node n1(5);
    Node n2(10);

    cout << "map0:" << endl;
    vigra::PropertyMap<Node, int> map0;
    map0[n0] = 27;
    map0[n2] = 73;
    cout << map0[n0] << endl;
    for (auto p : map0)
    {
    	cout << p.first << " => " << p.second << endl;
    }

    cout << "map1:" << endl;
    vigra::PropertyMap<Node, int> const map1(map0);
    cout << map1.at(n0) << endl;
    for (auto p : map1)
    {
    	cout << p.first << " => " << p.second << endl;
    }

    cout << "map2:" << endl;
    vigra::PropertyMap<Node, int, vigra::VectorTag > map2;
    map2[n0] = 28;
    map2[n2] = 74;
    cout << map2[n0] << endl;
    for (auto p : map2)
    {
    	cout << p.first << " => " << p.second << endl;
    }

    cout << "map3:" << endl;
    vigra::PropertyMap<Node, int, vigra::VectorTag > const map3(map2);
    cout << map3.at(n0) << endl;
    for (auto p : map3)
    {
    	cout << p.first << " => " << p.second << endl;
    }


    
    cout << "done" << endl;
}

