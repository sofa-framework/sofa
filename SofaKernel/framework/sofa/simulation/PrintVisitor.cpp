/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/PrintVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace simulation
{


template<class T>
void PrintVisitor::processObject(T obj)
{
    std::cout << ' ' << obj->getName() << '(' << sofa::helper::gettypename(typeid(*obj)) << ')';
}

template<class Seq>
void PrintVisitor::processObjects(Seq& list, const char* name)
{
    if (list.empty()) return;
    for (int i=0; i<=level; i++)
        std::cout << "| ";
    std::cout << name << ":";
    for (typename Seq::iterator it = list.begin(); it != list.end(); ++it)
    {
        typename Seq::value_type obj = *it;
        this->processObject<typename Seq::value_type>(obj);
    }
    std::cout << std::endl;
}

Visitor::Result PrintVisitor::processNodeTopDown(simulation::Node* node)
{
    for (int i=0; i<level; i++)
        std::cout << "| ";
    std::cout << "+-";
    std::cout << node->getName()<<std::endl;
    ++level;
    processObjects(node->mechanicalState,"MechanicalState");
    processObjects(node->mechanicalMapping,"MechanicalMapping");
    processObjects(node->solver,"Solver");
    processObjects(node->linearSolver,"LinearSolver");
    processObjects(node->mass,"Mass");
    processObjects(node->topology,"Topology");
    processObjects(node->forceField,"ForceField");
    processObjects(node->interactionForceField,"InteractionForceField");
    processObjects(node->projectiveConstraintSet,"ProjectiveConstraintSet");
    processObjects(node->constraintSet,"ConstraintSet");
    processObjects(node->contextObject,"ContextObject");

    processObjects(node->mapping,"Mapping");
    processObjects(node->behaviorModel,"BehaviorModel");
    processObjects(node->visualModel,"VisualModel");
    processObjects(node->collisionModel,"CollisionModel");
    processObjects(node->collisionPipeline,"CollisionPipeline");
    processObjects(node->unsorted,"unsorted");

    return RESULT_CONTINUE;
}

void PrintVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
    --level;
}

} // namespace simulation

} // namespace sofa

