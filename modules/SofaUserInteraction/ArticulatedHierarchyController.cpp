/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
//
// C++ Implementation: ArticulatedHierarchyController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <SofaUserInteraction/ArticulatedHierarchyController.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>

#include <sofa/simulation/Node.h>

// #include <cctype>

namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::helper;
using sofa::core::behavior::MechanicalState;
using sofa::component::controller::ArticulatedHierarchyContainer;


ArticulatedHierarchyController::ArticulatedHierarchyController()
    : articulationsIndices( initData(&articulationsIndices, "articulationsIndices", "Indices of articulations controlled by the keyboard") )
    , bindingKeys( initData(&bindingKeys, "bindingKeys", "Keys to press to control the articulations" ) )
    , angleDelta( initData(&angleDelta, 0.01, "angleDelta", "Angle incrementation due to each user interaction" ) )
    , propagateUserInteraction( initData(&propagateUserInteraction, false, "propagateUserInteraction", "Says wether or not the user interaction is local on the articulations, or must be propagated to children recursively" ) )
{
    m_artCenterVec.resize(0);
}



void ArticulatedHierarchyController::init()
{
    dumpArticulationsAndBindingKeys();

    activeArticulations.resize(bindingKeys.getValue().size());
    propagationChain = false;

    sofa::simulation::Node* curNode = dynamic_cast<sofa::simulation::Node*>(this->getContext());
    if (curNode)
        curNode->getTreeObjects<ArticulationCenter, ArtCenterVec >(&m_artCenterVec);

    resetControler();
}



void ArticulatedHierarchyController::buildPropagationArticulationsChain(void)
{
    ArtCenterVecIt artCenterIt;
    ArtCenterVecIt artCenterItEnd;

    ArtVecIt artIt;
    ArtVecIt artItEnd;

    ArticulationCenter *activeArticulationCenter = NULL;
    Articulation *activeArticulation = NULL;

    for (unsigned int i=0; i<articulationsIndices.getValue().size(); i++)
    {
        int activeArticulationIndex = articulationsIndices.getValue()[i];

        artCenterIt = m_artCenterVec.begin();
        artCenterItEnd = m_artCenterVec.end();

        while ((artCenterIt != artCenterItEnd) && (activeArticulationCenter == NULL))
        {
            artIt = (*artCenterIt)->articulations.begin();
            artItEnd = (*artCenterIt)->articulations.end();
            while (artIt != artItEnd)
            {
                if ((*artIt)->articulationIndex.getValue() == activeArticulationIndex)
                {
                    activeArticulationCenter = *artCenterIt;
                    activeArticulation = *artIt;
                    break;
                }
                ++artIt;
            }
            ++artCenterIt;
        }

        std::vector< int > propagationArticulationsArray;

        if ((activeArticulation != NULL) && (activeArticulationCenter != NULL))
        {
            buildArray(propagationArticulationsArray, activeArticulation, activeArticulationCenter);
        }

        articulationsPropagationChains.insert(std::make_pair(activeArticulationIndex, propagationArticulationsArray));

        activeArticulation = NULL;
        activeArticulationCenter = NULL;
    }
}



void ArticulatedHierarchyController::buildArray(std::vector< int > &artIndices, Articulation *artRef, ArticulationCenter *artCenterParent)
{
    ArtCenterVecIt artCenterIt = m_artCenterVec.begin();
    ArtCenterVecIt artCenterItEnd = m_artCenterVec.end();

    bool childFound = false;
    while (artCenterIt != artCenterItEnd)
    {
        if ((*artCenterIt)->parentIndex.getValue() == artCenterParent->childIndex.getValue())
        {
            ArtVecIt artIt = (*artCenterIt)->articulations.begin();
            ArtVecIt artItEnd = (*artCenterIt)->articulations.end();

            while (artIt != artItEnd)
            {
                if (((*artIt)->rotation.getValue() == artRef->rotation.getValue())
                    && ((*artIt)->translation.getValue() == artRef->translation.getValue())
                    && ((*artIt)->axis.getValue() == artRef->axis.getValue()))
                {
                    artIndices.push_back((*artIt)->articulationIndex.getValue());
                    childFound = true;
                    buildArray(artIndices,artRef,*artCenterIt);
                    break;
                }
                ++artIt;
            }
        }

        if (childFound) break;
        ++artCenterIt;
    }
}



void ArticulatedHierarchyController::dumpActiveArticulations(void) const
{
    vector<bool>::const_iterator it = activeArticulations.begin();
    vector<bool>::const_iterator itEnd = activeArticulations.end();
    int i=0;
    while (it != itEnd)
    {
        if (*it){
            msg_info() << "-------------> Articulation " << articulationsIndices.getValue()[i] << " active" ;
        }else{
            msg_info() << "-------------> Articulation " << articulationsIndices.getValue()[i] << " inactive" ;
        }
        ++it;
        i++;
    }
}



void ArticulatedHierarchyController::dumpArticulationsAndBindingKeys(void) const
{
    msg_info() << "ARTICULATIONS_KEYBOARD_CONTROLER : Controled Articulations & Binding Keys" ;

    vector<int>::const_iterator articulationsIndicesIt = articulationsIndices.getValue().begin();
    vector<int>::const_iterator articulationsIndicesItEnd = articulationsIndices.getValue().end();

    vector<char>::const_iterator bindinKeysIt = bindingKeys.getValue().begin();
    vector<char>::const_iterator bindinKeysItEnd = bindingKeys.getValue().end();

    while (articulationsIndicesIt != articulationsIndicesItEnd)
    {
        msg_info() << "Articulation " << *articulationsIndicesIt << " controlled with key " << *bindinKeysIt ;
        ++articulationsIndicesIt;
        ++bindinKeysIt;
        if (bindinKeysIt == bindinKeysItEnd)
            break;
    }
}



void ArticulatedHierarchyController::updateActiveArticulationsIndices(const char keyChar)
{
    unsigned int numKeys = bindingKeys.getValue().size();

    if (numKeys != 0)
    {
        unsigned int i = 0;
        for (; i < numKeys; i++)
        {
            if (bindingKeys.getValue()[i] == (isupper(bindingKeys.getValue()[i]) ? toupper(keyChar) : tolower(keyChar)))
                break;
        }

        if ((i < numKeys)&&(i < articulationsIndices.getValue().size()))
        {
            // If the selected articulation is the current activated one it must be disabled
            if (activeArticulations[i])
            {
                activeArticulations[i] = false;
            }
            else
            {
                // Set all but the new select one articulations as inactive
                vector<bool>::iterator it = activeArticulations.begin();
                vector<bool>::iterator itEnd = activeArticulations.end();
                while (it != itEnd)
                {
                    *it = false;
                    ++it;
                }

                activeArticulations[i] = true;
            }
        }
    }

    dumpActiveArticulations();
}



void ArticulatedHierarchyController::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kev)
{
    updateActiveArticulationsIndices(kev->getKey());
}



void ArticulatedHierarchyController::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
    switch (mev->getState())
    {
    case sofa::core::objectmodel::MouseEvent::LeftPressed :
        signFactor = 1;
        mouseMode = BtLeft;
        break;

    case sofa::core::objectmodel::MouseEvent::LeftReleased :
        mouseMode = None;
        break;

    case sofa::core::objectmodel::MouseEvent::RightPressed :
        signFactor = -1;
        mouseMode = BtRight;
        break;

    case sofa::core::objectmodel::MouseEvent::RightReleased :
        mouseMode = None;
        break;

    case sofa::core::objectmodel::MouseEvent::Wheel :
        signFactor = 2 * abs(mev->getWheelDelta()) / mev->getWheelDelta();
        mouseMode = Wheel;
        break;

    case  sofa::core::objectmodel::MouseEvent::Reset :
        resetControler();
        break;

    default :
        break;
    }
}



void ArticulatedHierarchyController::onBeginAnimationStep(const double /*dt*/)
{
    applyController();
}



void ArticulatedHierarchyController::resetControler(void)
{
    vector<bool>::iterator it = activeArticulations.begin();
    vector<bool>::iterator itEnd = activeArticulations.end();
    while (it != itEnd)
    {
        *it = false;
        ++it;
    }

    mouseMode = None;
}



void ArticulatedHierarchyController::applyController(void)
{
    if (mouseMode != None)
    {
        if ((!propagationChain) && (propagateUserInteraction.getValue()))
        {
            buildPropagationArticulationsChain();
            propagationChain = true;
        }

        // MouseWheel event won't be stopped by any "release" event, we stop it manually
        if (mouseMode == Wheel) mouseMode=None;

        int articulationIndex;
        unsigned int i = 0;
        for (; i<activeArticulations.size(); i++)
        {
            if (activeArticulations[i])
            {
                articulationIndex = articulationsIndices.getValue()[i];
                break;
            }
        }

        if (i < activeArticulations.size())
        {
            std::vector< int > articulationPropagationChain;
            std::map< int, sofa::helper::vector< int > >::iterator iter = articulationsPropagationChains.find(articulationIndex);
            if( iter != articulationsPropagationChains.end())
                articulationPropagationChain = iter->second;

            double distributedAngleDelta = angleDelta.getValue() / (double)(articulationPropagationChain.size() + 1);

            for (unsigned int j=0; j<articulationPropagationChain.size()+1; j++)
            {
                ArtCenterVecIt artCenterIt = m_artCenterVec.begin();
                ArtCenterVecIt artCenterItEnd = m_artCenterVec.end();

                bool articulationFound =  false;

                while ((artCenterIt != artCenterItEnd) && (!articulationFound))
                {
                    ArtVecIt it = (*artCenterIt)->articulations.begin();
                    ArtVecIt itEnd = (*artCenterIt)->articulations.end();
                    while (it != itEnd)
                    {
                        if ((*it)->articulationIndex.getValue() == articulationIndex)
                        {
                            std::vector< MechanicalState<sofa::defaulttype::Vec1Types>* > articulatedObjects;

                            sofa::simulation::Node* curNode = dynamic_cast<sofa::simulation::Node*>(this->getContext());
                            if (curNode)
                                curNode->getTreeObjects<MechanicalState<sofa::defaulttype::Vec1Types>, std::vector< MechanicalState<sofa::defaulttype::Vec1Types>* > >(&articulatedObjects);

                            if (!articulatedObjects.empty())
                            {
                                // Reference potential initial articulations value for interaction springs
                                // and Current articulation value at the coresponding artculation

                                std::vector< MechanicalState<sofa::defaulttype::Vec1Types>* >::iterator articulatedObjIt = articulatedObjects.begin();
//								std::vector< MechanicalState<sofa::defaulttype::Vec1dTypes>* >::iterator articulatedObjItEnd = articulatedObjects.end();

                                //	while (articulatedObjIt != articulatedObjItEnd)
                                {
                                    helper::WriteAccessor<Data<sofa::defaulttype::Vec1Types::VecCoord> > x = *(*articulatedObjIt)->write(sofa::core::VecCoordId::position());
                                    helper::WriteAccessor<Data<sofa::defaulttype::Vec1Types::VecCoord> > xfree = *(*articulatedObjIt)->write(sofa::core::VecCoordId::freePosition());
                                    x[(*it)->articulationIndex.getValue()].x() += signFactor * distributedAngleDelta;
                                    xfree[(*it)->articulationIndex.getValue()].x() += signFactor * distributedAngleDelta;
                                    ++articulatedObjIt;
                                }
                            }

                            articulationFound = true;
                            break;
                        }
                        ++it;
                    }
                    ++artCenterIt;
                }

                if (j < articulationPropagationChain.size())
                    articulationIndex = articulationPropagationChain[j];
            }

            static_cast<sofa::simulation::Node*>(this->getContext())->execute<sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor>(sofa::core::MechanicalParams::defaultInstance());
            static_cast<sofa::simulation::Node*>(this->getContext())->execute<sofa::simulation::UpdateMappingVisitor>(sofa::core::ExecParams::defaultInstance());
        }
    }
}

SOFA_DECL_CLASS(ArticulatedHierarchyController)

// Register in the Factory
int ArticulatedHierarchyControllerClass = core::RegisterObject("Implements an user interaction handler that controls the values of the articulations of an articulated hierarchy container.")
        .add< ArticulatedHierarchyController >()
        ;

} // namespace controller

} // namespace component

} // namespace sofa
