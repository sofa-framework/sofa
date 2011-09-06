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
#include <sofa/component/mastersolver/MultiTagMasterSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <math.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mastersolver
{

int MultiTagMasterSolverClass = core::RegisterObject("Simple master solver that given a list of tags, animate the graph one tag after another.")
        .add< MultiTagMasterSolver >()
        ;

SOFA_DECL_CLASS(MultiTagMasterSolver);

MultiTagMasterSolver::MultiTagMasterSolver(simulation::Node* gnode)
    : Inherit(gnode)
{
}

MultiTagMasterSolver::~MultiTagMasterSolver()
{
}

void MultiTagMasterSolver::init()
{
    tagList = this->getTags();
    sofa::core::objectmodel::TagSet::iterator it;

    for (it = tagList.begin(); it != tagList.end(); ++it)
        this->removeTag (*it);
}



void MultiTagMasterSolver::step(const sofa::core::ExecParams* params /* PARAMS FIRST */, double dt)
{
    sofa::helper::AdvancedTimer::stepBegin("MasterSolverStep");

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }


    double startTime = this->gnode->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();
    AnimateVisitor act(params);
    act.setDt ( mechanicalDt );
    BehaviorUpdatePositionVisitor beh(params , this->gnode->getDt());
    for( unsigned i=0; i<numMechSteps.getValue(); i++ )
    {
        this->gnode->execute ( beh );

        sofa::core::objectmodel::TagSet::iterator it;

        for (it = tagList.begin(); it != tagList.end(); ++it)
        {
            this->addTag (*it);

            if (this->f_printLog.getValue()) sout << "MultiTagMasterSolver::step, begin constraints reset" << sendl;
            sofa::simulation::MechanicalResetConstraintVisitor(params).execute(this->getContext());
            if (this->f_printLog.getValue()) sout << "MultiTagMasterSolver::step, end constraints reset" << sendl;
            if (this->f_printLog.getValue()) sout << "MultiTagMasterSolver::step, begin collision for tag: "<< *it << sendl;
            computeCollision(params);
            if (this->f_printLog.getValue()) sout << "MultiTagMasterSolver::step, end collision" << sendl;
            if (this->f_printLog.getValue()) sout << "MultiTagMasterSolver::step, begin integration  for tag: "<< *it << sendl;
            integrate(params /* PARAMS FIRST */, dt);
            if (this->f_printLog.getValue()) sout << "MultiTagMasterSolver::step, end integration" << sendl;

            this->removeTag (*it);
        }

        this->gnode->setTime ( startTime + (i+1)* act.getDt() );
        this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time
        nbMechSteps.setValue(nbMechSteps.getValue() + 1);
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    this->gnode->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        this->gnode->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

#ifndef SOFA_NO_UPDATE_BBOX
    sofa::helper::AdvancedTimer::stepBegin("UpdateBBox");
    this->gnode->execute<UpdateBoundingBoxVisitor>(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateBBox");
#endif
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode(std::string("Step"));
#endif
    nbSteps.setValue(nbSteps.getValue() + 1);

    sofa::helper::AdvancedTimer::stepEnd("MasterSolverStep");
}

void MultiTagMasterSolver::clear()
{
    if (!tagList.empty())
    {
        sofa::core::objectmodel::TagSet::iterator it;
        for (it = tagList.begin(); it != tagList.end(); ++it)
            this->addTag (*it);

        tagList.clear();
    }
}



} // namespace mastersolver

} // namespace component

} // namespace sofa

