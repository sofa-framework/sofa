/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/


#include "Solver_test.h"
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/tree/TreeSimulation.h>

namespace sofa {

using namespace modeling;


/** Check the traversal of a Directed Acyclic Graph.
 * @warning Derived from a Solver_test since it requires the ability to create a scene and traverse them with visitors.
 * @todo generalize the concept of Solver_test
 * Francois Faure, 2014
 */
struct DAG_test : public Solver_test
{
    DAG_test()
    {
        sofa::simulation::setSimulation(new simulation::graph::DAGSimulation());
//        sofa::simulation::setSimulation(new simulation::tree::TreeSimulation());
    }

    struct TestVisitor: public sofa::simulation::Visitor
    {

        std::string visited;

        TestVisitor():Visitor(sofa::core::ExecParams::defaultInstance() )
        {
            visited.clear();
        }

        Result processNodeTopDown(simulation::Node* node)
        {
            visited += node->getName();
            return RESULT_CONTINUE;
        }

        void processNodeBottomUp(simulation::Node* node)
        {
            visited += node->getName();
        }

    };

    void traverse_simple_tree()
    {
        Node::SPtr root = clearScene();
        root->setName("R");
        root->createChild("A");
        root->createChild("B");

        TestVisitor t;
        t.execute(root.get());
//        cout<<"traverse_simple_tree: visited = " << t.visited << endl;
        if( t.visited != "RAABBR"){
            ADD_FAILURE() << "Dag_test::traverse_simple_tree : wrong traversal order, expected RAABBR, got " << t.visited;
        }
//        sofa::simulation::getSimulation()->print(root.get());
    }



};

TEST_F( DAG_test,  )
{
    traverse_simple_tree();
}

}// namespace sofa







