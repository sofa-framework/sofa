/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/simulation/tree/PrintVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{

namespace tree
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
    // the following line breaks the compilator on Visual2003
    //for_each<PrintVisitor, Seq, typename Seq::value_type>(this, list, &PrintVisitor::processObject<typename Seq::value_type>);
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
    processObjects(node->mass,"Mass");
    processObjects(node->topology,"Topology");
    processObjects(node->forceField,"ForceField");
    processObjects(node->interactionForceField,"InteractionForceField");
    processObjects(node->constraint,"Constraint");
    processObjects(node->contextObject,"ContextObject");

    processObjects(node->mapping,"Mapping");
    processObjects(node->behaviorModel,"BehaviorModel");
    processObjects(node->visualModel,"VisualModel");
    processObjects(node->collisionModel,"CollisionModel");

    return RESULT_CONTINUE;
}

void PrintVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
    --level;
}

} // namespace tree

} // namespace simulation

} // namespace sofa

