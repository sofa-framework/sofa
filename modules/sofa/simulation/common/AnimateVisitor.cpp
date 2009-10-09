/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
//#include "MechanicalIntegration.h"

namespace sofa
{

namespace simulation
{


void AnimateVisitor::processMasterSolver(simulation::Node*, core::componentmodel::behavior::MasterSolver* obj)
{
    obj->step(getDt());
}

void AnimateVisitor::processBehaviorModel(simulation::Node*, core::BehaviorModel* obj)
{
    obj->updatePosition(getDt());
}

void AnimateVisitor::fwdInteractionForceField(simulation::Node*, core::componentmodel::behavior::InteractionForceField* obj)
{
    obj->addForce();
}

void AnimateVisitor::processCollisionPipeline(simulation::Node* node, core::componentmodel::collision::Pipeline*)
{
    CollisionVisitor act;
    node->execute(&act);
}

void AnimateVisitor::processOdeSolver(simulation::Node* /*node*/, core::componentmodel::behavior::OdeSolver* solver)
{
    /*    MechanicalIntegrationVisitor act(getDt());
        node->execute(&act);*/
    cerr<<"AnimateVisitor::processOdeSolver "<<solver->getName()<<endl;
    solver->solve(getDt());
}

Visitor::Result AnimateVisitor::processNodeTopDown(simulation::Node* node)
{
    //cerr<<"AnimateVisitor::process Node  "<<node->getName()<<endl;
    if (!node->is_activated.getValue()) return Visitor::RESULT_PRUNE;
    if (dt == 0) setDt(node->getDt());
// 	for_each(this, node, node->behaviorModel, &AnimateVisitor::processBehaviorModel);
    if (node->masterSolver != NULL)
    {
        ctime_t t0 = begin(node, node->masterSolver);
        processMasterSolver(node, node->masterSolver);
        end(node, node->masterSolver, t0);
        return RESULT_PRUNE;
    }
    if (node->collisionPipeline != NULL)
    {
        //ctime_t t0 = begin(node, node->collisionPipeline);
        processCollisionPipeline(node, node->collisionPipeline);
        //end(node, node->collisionPipeline, t0);
    }
    /*	if (node->solver != NULL)
    	{
    		ctime_t t0 = begin(node, node->solver);
    		processOdeSolver(node, node->solver);
    		end(node, node->solver, t0);
    		return RESULT_PRUNE;
            }*/
    if (!node->solver.empty() )
    {
        double nextTime = node->getTime() + dt;

        MechanicalBeginIntegrationVisitor beginVisitor(dt);
        node->execute(&beginVisitor);

#ifdef SOFA_HAVE_EIGEN2
        MechanicalExpressJacobianVisitor JacobianVisitor;
        node->execute(&JacobianVisitor);
#endif

        for( unsigned i=0; i<node->solver.size(); i++ )
        {
            ctime_t t0 = begin(node, node->solver[i]);
            //cerr<<"AnimateVisitor::processNodeTpDown  solver  "<<node->solver[i]->getName()<<endl;
            node->solver[i]->solve(getDt());
            end(node, node->solver[i], t0);
        }

        MechanicalPropagatePositionAndVelocityVisitor(nextTime,core::componentmodel::behavior::OdeSolver::VecId::position(),core::componentmodel::behavior::OdeSolver::VecId::velocity()).execute( node );

#ifdef SOFA_HAVE_EIGEN2
        MechanicalResetConstraintVisitor resetConstraint;
        node->execute(&resetConstraint);
#endif

        MechanicalEndIntegrationVisitor endVisitor(dt);
        node->execute(&endVisitor);
        return RESULT_PRUNE;
    }
    /*
    if (node->mechanicalModel != NULL)
    {
    	std::cerr << "Graph Error: MechanicalState without solver." << std::endl;
    	return RESULT_PRUNE;
    }
    */
    {
        // process InteractionForceFields
        for_each(this, node, node->interactionForceField, &AnimateVisitor::fwdInteractionForceField);
        return RESULT_CONTINUE;
    }
}

// void AnimateVisitor::processNodeBottomUp(simulation::Node* node)
// {
//     node->setTime( node->getTime() + node->getDt() );
// }


} // namespace simulation

} // namespace sofa

