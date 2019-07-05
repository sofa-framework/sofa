#include <SofaTest/Sofa_test.h>

#include <SofaSimulationGraph/DAGNode.h>

using namespace sofa;
using namespace simulation::graph;

struct DAGNode_test : public BaseTest
{
    DAGNode_test() {}

    void test_findCommonParent()
    {
        DAGNode::SPtr root = core::objectmodel::New<DAGNode>("root");
        DAGNode::SPtr node1 = core::objectmodel::New<DAGNode>("node1");
        DAGNode::SPtr node2 = core::objectmodel::New<DAGNode>("node2");
        DAGNode::SPtr node3 = core::objectmodel::New<DAGNode>("node3");
        DAGNode::SPtr node11 = core::objectmodel::New<DAGNode>("node11");
        DAGNode::SPtr node12 = core::objectmodel::New<DAGNode>("node12");
        DAGNode::SPtr node31 = core::objectmodel::New<DAGNode>("node31");

        root->addChild(node1);
        root->addChild(node2);
        root->addChild(node3);

        node1->addChild(node11);
        node1->addChild(node12);

        node3->addChild(node31);

        simulation::Node* commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node11.get()) );
        EXPECT_STREQ(node1->getName().c_str(), commonParent->getName().c_str());

        commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node31.get()));
        EXPECT_STREQ(root->getName().c_str(), commonParent->getName().c_str());

        commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node1.get()));
        EXPECT_STREQ(root->getName().c_str(), commonParent->getName().c_str());
    }

    void test_findCommonParent_MultipleParents()
    {
        DAGNode::SPtr root = core::objectmodel::New<DAGNode>("root");
        DAGNode::SPtr node1 = core::objectmodel::New<DAGNode>("node1");
        DAGNode::SPtr node2 = core::objectmodel::New<DAGNode>("node2");
        DAGNode::SPtr node11 = core::objectmodel::New<DAGNode>("node11");
        DAGNode::SPtr node22 = core::objectmodel::New<DAGNode>("node22");
        DAGNode::SPtr node23 = core::objectmodel::New<DAGNode>("node23");

        root->addChild(node1);
        root->addChild(node2);

        node1->addChild(node11);
        node1->addChild(node22);

        node2->addChild(node11);
        node2->addChild(node22);  
        node2->addChild(node23);

        simulation::Node* commonParent = node11->findCommonParent(static_cast<simulation::Node*>(node22.get()));

        bool result = false;
        if (commonParent->getName().compare(node1->getName()) == 0 || commonParent->getName().compare(node2->getName()) == 0)
        {
            result = true;
        }
        EXPECT_TRUE(result);

        commonParent = node11->findCommonParent(static_cast<simulation::Node*>(node23.get()));
        EXPECT_STREQ(node2->getName().c_str(), commonParent->getName().c_str());
    }
};

TEST_F(DAGNode_test, test_findCommonParent) { test_findCommonParent(); }
TEST_F(DAGNode_test, test_findCommonParent_MultipleParents) { test_findCommonParent_MultipleParents(); }
