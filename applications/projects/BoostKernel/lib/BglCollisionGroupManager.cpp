/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "BglCollisionGroupManager.h"
#include "BglSolverMerger.h"
#include "BglSimulation.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CollisionModel.h>
// #include <sofa/helper/system/config.h>
// #include <string.h>
#include <sofa/simulation/common/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::componentmodel::behavior;
using namespace core::componentmodel::collision;

SOFA_DECL_CLASS(BglCollisionGroupManager);

int BglCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration in Simulations ruled by the Boost Graph Library")
        .add< BglCollisionGroupManager >()
        ;


BglCollisionGroupManager::BglCollisionGroupManager()
{
}

BglCollisionGroupManager::~BglCollisionGroupManager()
{
}

simulation::Node* BglCollisionGroupManager::buildCollisionGroup()
{
    return simulation::getSimulation()->newNode("CollisionGroup");
}

void BglCollisionGroupManager::createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<Contact*>& contacts)
{
    if (!contacts.size()) return;
//         std::cerr << "create Groups in " << scene->getName() << "\n";

    simulation::bgl::BglSimulation *simu = dynamic_cast<simulation::bgl::BglSimulation*>(simulation::getSimulation());
    assert(simu);

    std::map< Contact*, simulation::Node* > contactGroup;
    std::vector< simulation::Node **> psolver; psolver.resize(contacts.size());

    for (unsigned int i=0; i<contacts.size(); ++i)
    {
        Contact* contact = contacts[i];
//             std::cerr << "\tbetween : " << contact->getCollisionModels().first->getName()  << "@" << contact->getCollisionModels().first->getContext()
//                       << " and "        << contact->getCollisionModels().second->getName() << "@" << contact->getCollisionModels().second->getContext()
//                       << std::endl;

        simulation::bgl::BglNode *group1=dynamic_cast<simulation::bgl::BglNode *>(getIntegrationNode(contact->getCollisionModels().first));
        simulation::bgl::BglNode *group2=dynamic_cast<simulation::bgl::BglNode *>(getIntegrationNode(contact->getCollisionModels().second));
//             simulation::bgl::BglNode *group=NULL;

        if (group1== NULL || group2 == NULL)
        {
            //One of the interaction concern a not simulated object: nothing todo
//                 std::cerr<< "NULL : " << group1 << " : " << group2 << "\n";
        }
        else if (group1 == group2)
        {
            std::cerr << "Equal : " << group1->getName() << "\n";
            //Two collision models using the same solver are colliding
//                 groupSet[group1] = group1->solver[0];
// group = group1;

        }
        else
        {
//                 std::cerr << "Test : " << group1->getName() << " : " << group2->getName() << "\n";
            simulation::Node **s1=groupSet[group1];
            simulation::Node **s2=groupSet[group2];
            if (!s1) s1 = (simulation::Node**)(&group1);
            if (!s2) s2 = (simulation::Node**)(&group2);

            // We can merge the groups
            // if solvers are compatible...
            psolver[i] = new simulation::Node*();
            (*psolver[i]) = solvermerger::BglSolverMerger::merge((*s1)->solver[0], (*s2)->solver[0]);
            groupSet[group1] = psolver[i];
            groupSet[group2] = psolver[i];
            contactGroup[contact] = group1;
        }


    }

    ///Setting in the BglSimulation the solver
    GroupSet::iterator it;
    for (it=groupSet.begin(); it!=groupSet.end(); it++)
    {
//             std::cerr << "Setting : " << it->first->getName()  << " with " << (*(it->second))->getName() << "\n";
        simu->graphManager.setSolverOfCollisionGroup(it->first,*(it->second));
    }

    // now that the groups are final, attach contacts' response
    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact* contact = contacts[i];

        if (contactGroup.find(contact) != contactGroup.end())
            contact->createResponse(*groupSet[contactGroup[contact]]);
        else
            contact->createResponse(scene);
    }
    for (unsigned int i=0; i<psolver.size(); ++i) delete psolver[i];
}

void BglCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
//         std::cerr << "clearGroups \n";
    groupSet.clear();
}

simulation::Node* BglCollisionGroupManager::getIntegrationNode(core::CollisionModel* model)
{

    simulation::Node* node = static_cast<simulation::Node*>(model->getContext());
    helper::vector< core::componentmodel::behavior::OdeSolver *> listSolver;
    node->get< core::componentmodel::behavior::OdeSolver >(&listSolver);

    if (!listSolver.empty()) return static_cast<simulation::Node*>(listSolver.back()->getContext());
    else                     return NULL;
}


}// namespace collision

} // namespace component

} // namespace Sofa
