/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;

#include <sofa/simpleapi/SimpleApi.h>
using namespace sofa::simpleapi ;

#include "Node_test.h"
#include <unordered_set>

namespace sofa
{

TEST( Node_test, getPathName)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    /* create trivial DAG :
     *
     * A
     * |\
     * B C
     * |
     * D
     *
     */
    EXPECT_MSG_NOEMIT(Error, Warning);

    const Node::SPtr root = sofa::simpleapi::createNode("A");
    const Node::SPtr B = createChild(root, "B");
    const Node::SPtr D = createChild(B, "D");
    const BaseObject::SPtr C = core::objectmodel::New<Dummy>("C");
    root->addObject(C);

    EXPECT_STREQ(root->getPathName().c_str(), "/");
    EXPECT_STREQ(B->getPathName().c_str(), "/B");
    EXPECT_STREQ(C->getPathName().c_str(), "/C");
    EXPECT_STREQ(D->getPathName().c_str(), "/B/D");
}

TEST(Node_test, addObject)
{
    const sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");
    const BaseObject::SPtr A = core::objectmodel::New<Dummy>("A");
    const BaseObject::SPtr B = core::objectmodel::New<Dummy>("B");

    root->addObject(A);

    // adds a second object after the last one.
    root->addObject(B);
    const auto b = dynamic_cast< sofa::core::objectmodel::BaseContext*>(root.get());
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(b->getObjects()[0]->getName(), "A");
    ASSERT_EQ(b->getObjects()[1]->getName(), "B");
}

TEST(Node_test, addObjectAtFront)
{
    const sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");
    const BaseObject::SPtr A = core::objectmodel::New<Dummy>("A");
    const BaseObject::SPtr B = core::objectmodel::New<Dummy>("B");

    root->addObject(A);

    // adds a second object before the first one.
    root->addObject(B, sofa::core::objectmodel::TypeOfInsertion::AtBegin);
    const auto b = dynamic_cast< sofa::core::objectmodel::BaseContext*>(root.get());
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(b->getObjects()[0]->getName(), "B");
    ASSERT_EQ(b->getObjects()[1]->getName(), "A");
}

TEST(Node_test, addObjectPreventingSharedContext)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    const sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");

    const BaseObject::SPtr A = core::objectmodel::New<Dummy>("A");
    const BaseObject::SPtr B = core::objectmodel::New<Dummy>("B");

    const auto child1 = sofa::simpleapi::createChild(root, "child1");
    const auto child2 = sofa::simpleapi::createChild(root, "child2");

    // add the created object into the node named 'child1'
    child1->addObject(A);
    child1->addObject(B);

    // check that the two objects are in node1
    ASSERT_NE(child1->getObject(A->getName()), nullptr);
    ASSERT_NE(child1->getObject(B->getName()), nullptr);

    ASSERT_NE(A->getPathName(), "/root/child1/A");
    ASSERT_NE(B->getPathName(), "/root/child1/B");

    // try to add the object into a new context
    {
        EXPECT_MSG_EMIT(Error);
        child2->addObject(A);
    }

    // try to add the object into a new context
    {
        EXPECT_MSG_NOEMIT(Error);
        child2->moveObject(A);
    }

}

TEST(Node_test, getObjectsStdVector)
{
    const sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");
    const Dummy::SPtr A = core::objectmodel::New<Dummy>("A");
    const Dummy::SPtr B = core::objectmodel::New<Dummy>("B");

    root->addObject(A);
    root->addObject(B);

    std::vector<Dummy*> objects;
    root->BaseContext::getObjects(objects, core::objectmodel::BaseContext::SearchDirection::SearchDown);

    EXPECT_EQ(objects.size(), 2);

    EXPECT_NE(std::find(objects.begin(), objects.end(), A.get()), objects.end());
    EXPECT_NE(std::find(objects.begin(), objects.end(), B.get()), objects.end());
}

TEST(Node_test, getObjectsStdSet)
{
    const sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");
    const Dummy::SPtr A = core::objectmodel::New<Dummy>("A");
    const Dummy::SPtr B = core::objectmodel::New<Dummy>("B");

    root->addObject(A);
    root->addObject(B);

    std::set<Dummy*> objects;
    root->BaseContext::getObjects(objects, core::objectmodel::BaseContext::SearchDirection::SearchDown);

    EXPECT_EQ(objects.size(), 2);

    EXPECT_NE(objects.find(A.get()), objects.end());
    EXPECT_NE(objects.find(B.get()), objects.end());
}

TEST(Node_test, getObjectsStdUnorderedSet)
{
    const sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");
    const Dummy::SPtr A = core::objectmodel::New<Dummy>("A");
    const Dummy::SPtr B = core::objectmodel::New<Dummy>("B");

    root->addObject(A);
    root->addObject(B);

    std::unordered_set<Dummy*> objects;
    root->BaseContext::getObjects(objects, core::objectmodel::BaseContext::SearchDirection::SearchDown);

    EXPECT_EQ(objects.size(), 2);

    EXPECT_NE(objects.find(A.get()), objects.end());
    EXPECT_NE(objects.find(B.get()), objects.end());
}

class CounterVisitor : public simulation::MechanicalVisitor
{
public:
    using MechanicalVisitor::MechanicalVisitor;

    Result fwdMechanicalState(simulation::Node* node, sofa::core::behavior::BaseMechanicalState* state) override
    {
        SOFA_UNUSED(node);
        SOFA_UNUSED(state);
        m_counter++;
        return Result::RESULT_CONTINUE;
    }

    Result fwdMappedMechanicalState(simulation::Node* node, sofa::core::behavior::BaseMechanicalState* state) override
    {
        SOFA_UNUSED(node);
        SOFA_UNUSED(state);
        ++m_counter;
        return Result::RESULT_CONTINUE;
    }

    std::size_t m_counter = 0;
};

TEST(Node_test, twoMechanicalStatesInTheSameNode)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    const sofa::core::sptr<Node> root = sofa::simpleapi::createNode("root");

    const auto plugins = testing::makeScopedPlugin({Sofa.Component.StateContainer});
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"template", "Vec3"}, {"name", "A"}});

    EXPECT_MSG_EMIT(Warning);
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"template", "Vec3"}, {"name", "B"}});

    //the last added state is the one in Node
    EXPECT_EQ(root->mechanicalState->getName(), "B");

    CounterVisitor visitor(core::MechanicalParams::defaultInstance());
    root->executeVisitor(&visitor);

    //only one of the two added states is visited
    EXPECT_EQ(visitor.m_counter, 1);
}

TEST(NodeTest, objectDestruction_singleObject)
{
    EXPECT_MSG_NOEMIT(Error) ;

    Node_test_objectDestruction_singleObject<Node>();
}

TEST(NodeTest, objectDestruction_multipleObjects)
{
    EXPECT_MSG_NOEMIT(Error) ;
    Node_test_objectDestruction_multipleObjects<Node>();
}

TEST(NodeTest, objectDestruction_childNode_singleObject)
{
    EXPECT_MSG_NOEMIT(Error) ;
    Node_test_objectDestruction_childNode_singleObject<Node>();
}

TEST(NodeTest, objectDestruction_childNode_complexChild)
{
    EXPECT_MSG_NOEMIT(Error) ;
    Node_test_objectDestruction_childNode_complexChild<Node>();
}


struct Node_test_f : public BaseTest
{
    Node_test_f() {}

    void test_findCommonParent()
    {
        const Node::SPtr root = core::objectmodel::New<Node>("root");
        const Node::SPtr node1 = core::objectmodel::New<Node>("node1");
        const Node::SPtr node2 = core::objectmodel::New<Node>("node2");
        const Node::SPtr node3 = core::objectmodel::New<Node>("node3");
        const Node::SPtr node11 = core::objectmodel::New<Node>("node11");
        const Node::SPtr node12 = core::objectmodel::New<Node>("node12");
        const Node::SPtr node31 = core::objectmodel::New<Node>("node31");

        root->addChild(node1);
        root->addChild(node2);
        root->addChild(node3);

        node1->addChild(node11);
        node1->addChild(node12);

        node3->addChild(node31);

        const simulation::Node* commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node11.get()) );
        EXPECT_STREQ(node1->getName().c_str(), commonParent->getName().c_str());

        commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node31.get()));
        EXPECT_STREQ(root->getName().c_str(), commonParent->getName().c_str());

        commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node1.get()));
        EXPECT_STREQ(root->getName().c_str(), commonParent->getName().c_str());
    }

    void test_findCommonParent_MultipleParents()
    {
        const Node::SPtr root = core::objectmodel::New<Node>("root");
        const Node::SPtr node1 = core::objectmodel::New<Node>("node1");
        const Node::SPtr node2 = core::objectmodel::New<Node>("node2");
        const Node::SPtr node11 = core::objectmodel::New<Node>("node11");
        const Node::SPtr node22 = core::objectmodel::New<Node>("node22");
        const Node::SPtr node23 = core::objectmodel::New<Node>("node23");

        root->addChild(node1);
        root->addChild(node2);

        node1->addChild(node11);
        node1->addChild(node22);

        node2->addChild(node11);
        node2->addChild(node22);
        node2->addChild(node23);

        const simulation::Node* commonParent = node11->findCommonParent(static_cast<simulation::Node*>(node22.get()));

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

TEST_F(Node_test_f, test_findCommonParent) { test_findCommonParent(); }
TEST_F(Node_test_f, test_findCommonParent_MultipleParents) { test_findCommonParent_MultipleParents(); }


/** Check the traversal of a Directed Acyclic Graph.
 * The traversal order is recorded in a string, and compared with an expected one.
 * @author Francois Faure, Matthieu Nesme @date 2014
 */
struct DAG_test : public BaseTest
{
    DAG_test()
    {
    }


    /**
     * The TestVisitor struct records the name of the traversed nodes in a string.
     * The string can be analyzed as a trace of the traversal.
     */
    struct TestVisitor: public sofa::simulation::Visitor
    {

        std::string visited, topdown, bottomup;
        bool tree; // enforce tree traversal
        TreeTraversalRepetition repeat; // repeat callbacks

        TestVisitor()
            : Visitor(sofa::core::execparams::defaultInstance() )
            , tree( false )
            , repeat( NO_REPETITION )
        {
            clear();
        }

        void clear()
        {
            visited.clear();
            topdown.clear();
            bottomup.clear();
        }

        Result processNodeTopDown(simulation::Node* node) override
        {
            visited += node->getName();
            topdown += node->getName();
            return RESULT_CONTINUE;
        }

        void processNodeBottomUp(simulation::Node* node) override
        {
            visited += node->getName();
            bottomup += node->getName();
        }

        bool treeTraversal(TreeTraversalRepetition& r) override { r=repeat; return tree; }

    };



    /// utility function testing a scene graph traversals with given expected results
    static void traverse_test( Node::SPtr node, std::string treeTraverse, std::string treeTraverseRepeatAll, std::string treeTraverseRepeatOnce, std::string dagTopDown )
    {
        // dagBottumUp must be the exact inverse of dagTopDown
        std::string dagBottomUp(dagTopDown);
        std::reverse(dagBottomUp.begin(), dagBottomUp.end());

        TestVisitor t;

        t.tree = true; // visitor as TREE traversal w/o repetition
        t.execute( node.get() );
        //        cout<<"traverse_simple_tree: visited = " << t.visited << endl;
        if( t.visited != treeTraverse ){
            ADD_FAILURE() << "Dag_test::traverse_test treeTraverse: wrong traversal order, expected "<<treeTraverse<<", got " << t.visited;
        }
        t.clear();
        t.repeat = simulation::Visitor::REPEAT_ALL; // visitor as TREE traversal with repetitions
        t.execute( node.get() );
        //        cout<<"traverse_simple_tree: visited = " << t.visited << endl;
        if( t.visited != treeTraverseRepeatAll ){
            ADD_FAILURE() << "Dag_test::traverse_test treeTraverseRepeatAll: wrong traversal order, expected "<<treeTraverseRepeatAll<<", got " << t.visited;
        }
        t.clear();
        t.repeat = simulation::Visitor::REPEAT_ONCE; // visitor as TREE traversal with first repetition
        t.execute( node.get() );
        //        cout<<"traverse_simple_tree: visited = " << t.visited << endl;
        if( t.visited != treeTraverseRepeatOnce ){
            ADD_FAILURE() << "Dag_test::traverse_test treeTraverseRepeatOnce: wrong traversal order, expected "<<treeTraverseRepeatOnce<<", got " << t.visited;
        }

        t.clear();
        t.tree = false; // visitor as DAG traversal
        t.execute(node.get());
        //        cout<<"traverse_test: visited = " << t.visited << endl;
        if( t.topdown != dagTopDown ){
            ADD_FAILURE() << "Dag_test::traverse_test dagTopDown: wrong traversal order, expected "<<dagTopDown<<", got " << t.topdown;
        }
        if( t.bottomup != dagBottomUp ){
            ADD_FAILURE() << "Dag_test::traverse_test dagBottomUp: wrong traversal order, expected "<<dagBottomUp<<", got " << t.bottomup;
        }


//        sofa::simulation::getSimulation()->print(node.get());
    }


    /**
     * @brief The root and two children:
\f$
\begin{array}{ccc}
 & R & \\
 \diagup & & \diagdown \\
 A & & B
\end{array}
\f$
Expected output: RAABBR
     */
    void traverse_simple_tree()
    {
        const Node::SPtr root = sofa::simulation::getSimulation()->createNewGraph("");

        root->setName("R");
        root->createChild("A");
        root->createChild("B");

        traverse_test( root, "RAABBR", "RAABBR", "RAABBR", "RAB" );
    }


    /**
     * @brief Diamond-shaped graph:
\f$
\begin{array}{ccc}
 & R & \\
 \diagup & & \diagdown \\
 A & & B \\
 \diagdown & & \diagup \\
 & C
\end{array}
\f$
Expected output: RABCCBAR
     */
    void traverse_simple_diamond()
    {
        const Node::SPtr root = sofa::simulation::getSimulation()->createNewGraph("");

        root->setName("R");
        const Node::SPtr A = root->createChild("A");
        const Node::SPtr B = root->createChild("B");
        const Node::SPtr C = A->createChild("C");
        B->addChild(C);

        traverse_test( root, "RACCABBR", "RACCABCCBR", "RACCABCCBR", "RABC" );
    }


/**
  * @brief More complex graph:

  R__
 / \ \
 A B |
 \ / |
  C  /
  \ /
   D
   |
   E

Expected output: RABCDEEDCBAR
     */
    void traverse_complex()
    {
        const Node::SPtr root = sofa::simulation::getSimulation()->createNewGraph("");
        root->setName("R");
        const Node::SPtr A = root->createChild("A");
        const Node::SPtr B = root->createChild("B");
        const Node::SPtr C = A->createChild("C");
        B->addChild(C);
        const Node::SPtr D = C->createChild("D");
        root->addChild(D);
        Node::SPtr E = D->createChild("E");

        traverse_test( root, "RACDEEDCABBR", "RACDEEDCABCDEEDCBDEEDR", "RACDEEDCABCCBDDR", "RABCDE" );
    }


/**
  * @brief Even more complex graph:

  R__
 / \ \
 A B C
 \/ \|
  D  E
  |  |
  F  G

     */
    void traverse_morecomplex()
    {
        const Node::SPtr root = sofa::simulation::getSimulation()->createNewGraph("");
        root->setName("R");
        const Node::SPtr A = root->createChild("A");
        const Node::SPtr B = root->createChild("B");
        const Node::SPtr C = root->createChild("C");
        const Node::SPtr D = A->createChild("D");
        B->addChild(D);
        const Node::SPtr E = B->createChild("E");
        C->addChild(E);
        Node::SPtr F = D->createChild("F");
        Node::SPtr G = E->createChild("G");

        traverse_test( root, "RADFFDABEGGEBCCR", "RADFFDABDFFDEGGEBCEGGECR", "RADFFDABDDEGGEBCEECR", "RABDFCEG" );
    }


/**
  * @brief another complex case

  R______
 / \ \ \ \
 A B C D E
 \/__/_/_/
  F
  |\
  G |
  |/
  H

     */
    void traverse_morecomplex2()
    {
        const Node::SPtr root = sofa::simulation::getSimulation()->createNewGraph("");
        root->setName("R");
        const Node::SPtr A = root->createChild("A");
        const Node::SPtr B = root->createChild("B");
        const Node::SPtr C = root->createChild("C");
        const Node::SPtr D = root->createChild("D");
        const Node::SPtr E = root->createChild("E");
        const Node::SPtr F = A->createChild("F");
        B->addChild(F);
        C->addChild(F);
        D->addChild(F);
        E->addChild(F);
        const Node::SPtr G = F->createChild("G");
        const Node::SPtr H = G->createChild("H");
        F->addChild(H);

        traverse_test( root, "RAFGHHGFABBCCDDEER",
                       "RAFGHHGHHFABFGHHGHHFBCFGHHGHHFCDFGHHGHHFDEFGHHGHHFER",
                       "RAFGHHGHHFABFFBCFFCDFFDEFFER",
                       "RABCDEFGH" );
    }



    static void getObjectByPath( Node::SPtr node, const std::string& searchpath, const std::string& objpath )
    {
        void *foundObj = node->getObject(classid(Dummy), searchpath);
        ASSERT_TRUE( foundObj!=nullptr );
        const Dummy* dummyObj = reinterpret_cast<Dummy*>(foundObj);
        ASSERT_TRUE( dummyObj!=nullptr );
        EXPECT_STREQ( objpath.c_str(), dummyObj->getPathName().c_str() );
    }



    void getObject()
    {
        const Node::SPtr A = sofa::simulation::getSimulation()->createNewGraph("");
        A->setName("A");

        const Node::SPtr B = A->createChild("B");
        const Node::SPtr C = A->createChild("C");
        const Node::SPtr D = B->createChild("D");
        C->addChild(D);
        const Node::SPtr E = D->createChild("E");

/**
        A
       / \
       B C
       \ /
        D
        |
        E
*/

        const Dummy::SPtr dummyA = sofa::core::objectmodel::New<Dummy>("obj");
        A->addObject(dummyA);
        const Dummy::SPtr dummyA2 = sofa::core::objectmodel::New<Dummy>("obj2");
        A->addObject(dummyA2);
        const Dummy::SPtr dummyB = sofa::core::objectmodel::New<Dummy>("obj");
        B->addObject(dummyB);
        const Dummy::SPtr dummyC = sofa::core::objectmodel::New<Dummy>("obj");
        C->addObject(dummyC);
        const Dummy::SPtr dummyD = sofa::core::objectmodel::New<Dummy>("obj");
        D->addObject(dummyD);
        const Dummy::SPtr dummyE = sofa::core::objectmodel::New<Dummy>("obj");
        E->addObject(dummyE);



        // by path
        {
            const void* foundObj = A->getObject(classid(Dummy), "/inexisting");
        ASSERT_TRUE( foundObj==nullptr );
        }

        getObjectByPath( A, "/obj", "/obj" );
        getObjectByPath( A, "obj", "/obj" );
        getObjectByPath( A, "/B/obj", "/B/obj" );
        getObjectByPath( A, "C/obj", "/C/obj" );
        getObjectByPath( A, "/B/D/obj", "/B/D/obj" );
        getObjectByPath( A, "C/D/obj", "/B/D/obj" );
        getObjectByPath( A, "/B/D/E/obj", "/B/D/E/obj" );
        getObjectByPath( A, "C/D/E/obj", "/B/D/E/obj" );
        getObjectByPath( B, "obj", "/B/obj" );
        getObjectByPath( C, "D/E/obj", "/B/D/E/obj" );
        getObjectByPath( A, "/obj2", "/obj2" );
        getObjectByPath( A, "obj2", "/obj2" );


        // TODO test other getObject{s} functions



    }
};

TEST_F( DAG_test, traverse )
{
    EXPECT_MSG_NOEMIT(Error) ;
    traverse_simple_tree();
    traverse_simple_diamond();
    traverse_complex();
    traverse_morecomplex();
    traverse_morecomplex2();
}

TEST_F(DAG_test, getObject)
{
    EXPECT_MSG_NOEMIT(Error) ;
    getObject();
}


}// namespace sofa







