/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "../Node_test.h"
#include <SofaTest/Sofa_test.h>
#include <SceneCreator/SceneCreator.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/graph/DAGNode.h>
#include <sofa/simulation/graph/DAGSimulation.h>

using sofa::simulation::graph::DAGNode;

namespace sofa {

using namespace modeling;
using namespace simulation;


/** Check the traversal of a Directed Acyclic Graph.
 * The traversal order is recorded in a string, and compared with an expected one.
 * @author Francois Faure, Matthieu Nesme @date 2014
 */
struct DAG_test : public Sofa_test<>
{
    DAG_test()
    {
        sofa::simulation::setSimulation(new simulation::graph::DAGSimulation());
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
            : Visitor(sofa::core::ExecParams::defaultInstance() )
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

        Result processNodeTopDown(simulation::Node* node)
        {
            visited += node->getName();
            topdown += node->getName();
            return RESULT_CONTINUE;
        }

        void processNodeBottomUp(simulation::Node* node)
        {
            visited += node->getName();
            bottomup += node->getName();
        }

        bool treeTraversal(TreeTraversalRepetition& r) { r=repeat; return tree; }

    };



    /// utility function testing a scene graph traversals with given expected results
    void traverse_test( Node::SPtr node, std::string treeTraverse, std::string treeTraverseRepeatAll, std::string treeTraverseRepeatOnce, std::string dagTopDown )
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
        Node::SPtr root = clearScene();
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
        Node::SPtr root = clearScene();
        root->setName("R");
        Node::SPtr A = root->createChild("A");
        Node::SPtr B = root->createChild("B");
        Node::SPtr C = A->createChild("C");
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
        Node::SPtr root = clearScene();
        root->setName("R");
        Node::SPtr A = root->createChild("A");
        Node::SPtr B = root->createChild("B");
        Node::SPtr C = A->createChild("C");
        B->addChild(C);
        Node::SPtr D = C->createChild("D");
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
        Node::SPtr root = clearScene();
        root->setName("R");
        Node::SPtr A = root->createChild("A");
        Node::SPtr B = root->createChild("B");
        Node::SPtr C = root->createChild("C");
        Node::SPtr D = A->createChild("D");
        B->addChild(D);
        Node::SPtr E = B->createChild("E");
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
        Node::SPtr root = clearScene();
        root->setName("R");
        Node::SPtr A = root->createChild("A");
        Node::SPtr B = root->createChild("B");
        Node::SPtr C = root->createChild("C");
        Node::SPtr D = root->createChild("D");
        Node::SPtr E = root->createChild("E");
        Node::SPtr F = A->createChild("F");
        B->addChild(F);
        C->addChild(F);
        D->addChild(F);
        E->addChild(F);
        Node::SPtr G = F->createChild("G");
        Node::SPtr H = G->createChild("H");
        F->addChild(H);

        traverse_test( root, "RAFGHHGFABBCCDDEER",
                       "RAFGHHGHHFABFGHHGHHFBCFGHHGHHFCDFGHHGHHFDEFGHHGHHFER",
                       "RAFGHHGHHFABFFBCFFCDFFDEFFER",
                       "RABCDEFGH" );
    }

};



TEST_F( DAG_test, traverse )
{
    traverse_simple_tree();
    traverse_simple_diamond();
    traverse_complex();
    traverse_morecomplex();
    traverse_morecomplex2();
}

TEST(DAGNodeTest, objectDestruction_singleObject)
{
    Node_test_objectDestruction_singleObject<DAGNode>();
}

TEST(DAGNodeTest, objectDestruction_multipleObjects)
{
    Node_test_objectDestruction_multipleObjects<DAGNode>();
}

TEST(DAGNodeTest, objectDestruction_childNode_singleObject)
{
    Node_test_objectDestruction_childNode_singleObject<DAGNode>();
}

TEST(DAGNodeTest, objectDestruction_childNode_complexChild)
{
    Node_test_objectDestruction_childNode_complexChild<DAGNode>();
}


}// namespace sofa
