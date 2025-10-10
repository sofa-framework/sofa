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
#include "SceneCheckCollisionResponse.h"

#include <sofa/simulation/Node.h>
#include <sofa/component/collision/response/contact/CollisionResponse.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>

namespace sofa::_scenechecking_
{

const bool SceneCheckCollisionResponseRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckCollisionResponse::newSPtr());

using sofa::simulation::Node;

const std::string SceneCheckCollisionResponse::getName()
{
    return "SceneCheckCollisionResponse";
}

const std::string SceneCheckCollisionResponse::getDesc()
{
    return "Check that the appropriate components are in the scene to compute the desired collision response";
}

void SceneCheckCollisionResponse::doInit(Node* node)
{
    SOFA_UNUSED(node);
    m_message.str("");
}

void SceneCheckCollisionResponse::doCheckOn(Node* node)
{
    if(m_checkDone)
        return;

    const sofa::core::objectmodel::BaseContext* root = node->getContext()->getRootContext();
    std::vector<sofa::component::collision::response::contact::CollisionResponse*> contactManager;
    root->get<sofa::component::collision::response::contact::CollisionResponse>(&contactManager, sofa::core::objectmodel::BaseContext::SearchDown);
    m_checkDone=true;
    const sofa::Size nbContactManager = contactManager.size();
    if( nbContactManager  > 0 )
    {
        if( nbContactManager!= 1 )
        {
            m_message << "Only one CollisionResponse is allowed in the scene."<< msgendl;
        }
        else
        {
            const std::string response = contactManager[0]->d_response.getValue().getSelectedItem();

            /// If StickContactConstraint is chosen, make sure the scene includes a FreeMotionAnimationLoop and a GenericConstraintSolver (specifically)
            if ( response == "StickContactConstraint" )
            {
                sofa::core::behavior::BaseAnimationLoop* animationLoop;
                root->get(animationLoop, sofa::core::objectmodel::BaseContext::SearchRoot);
                if (!animationLoop || ( animationLoop && ( animationLoop->getClassName() != "FreeMotionAnimationLoop" )) )
                {
                    m_message <<"A FreeMotionAnimationLoop must be in the scene to solve StickContactConstraint" << msgendl;
                }

                sofa::core::behavior::ConstraintSolver* constraintSolver;
                root->get(constraintSolver, sofa::core::objectmodel::BaseContext::SearchRoot);
                if (!constraintSolver || ( constraintSolver && ( constraintSolver->getClassName() != "ProjectedGaussSeidelConstraintSolver" )) )
                {
                    m_message <<"A ProjectedGaussSeidelConstraintSolver must be in the scene to solve StickContactConstraint" << msgendl;
                }
            }
            /// If FrictionContactConstraint is chosen, make sure the scene includes a FreeMotionAnimationLoop
            else if ( response == "FrictionContactConstraint")
            {
                sofa::core::behavior::BaseAnimationLoop* animationLoop;
                root->get(animationLoop, sofa::core::objectmodel::BaseContext::SearchRoot);
                if (!animationLoop || ( animationLoop && ( animationLoop->getClassName() != "FreeMotionAnimationLoop" )) )
                {
                    m_message <<"A FreeMotionAnimationLoop must be in the scene to solve FrictionContactConstraint" << msgendl;
                }
                else
                {
                    checkIfContactStiffnessIsSet(root);
                }
            }
            /// If PenalityContactForceField make sure that contactStiffness is defined
            else if ( response == "PenalityContactForceField")
            {
                checkIfContactStiffnessIsNotSet(root);
            }
        }
    }
}

void SceneCheckCollisionResponse::checkIfContactStiffnessIsSet(const sofa::core::objectmodel::BaseContext* root)
{
    type::vector<core::CollisionModel*> colModels;
    root->get<core::CollisionModel>(&colModels, core::objectmodel::BaseContext::SearchDown);
    for (const auto model : colModels)
    {
        if(model->isContactStiffnessSet())
        {
            m_message <<"The data \"contactStiffness\" is set in the component " << model->getClassName() <<", named \"" << model->getName() << "\"";
            m_message <<"This data is not used when using a FrictionContactConstraint collision response." << msgendl;
            m_message <<"Remove the data \"contactStiffness\" to remove this warning" << msgendl;
            break;
        }
    }
}

void SceneCheckCollisionResponse::checkIfContactStiffnessIsNotSet(const sofa::core::objectmodel::BaseContext* root)
{
    type::vector<core::CollisionModel*> colModels;
    root->get<core::CollisionModel>(&colModels, core::objectmodel::BaseContext::SearchDown);
    for (const auto model : colModels)
    {
        if(!model->isContactStiffnessSet())
        {
            m_message <<"Using PenalityContactForceField, the contactStiffness should be defined for each CollisionModel" << msgendl;
            break;
        }
    }
}

void SceneCheckCollisionResponse::doPrintSummary()
{
    if(m_checkDone && m_message.str()!= "")
    {
        msg_warning(this->getName()) << m_message.str();
    }
}


} // namespace sofa::_scenechecking_
