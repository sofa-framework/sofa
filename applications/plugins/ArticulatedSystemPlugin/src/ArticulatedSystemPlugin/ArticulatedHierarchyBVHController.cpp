/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
// C++ Implementation: ArticulatedHierarchyBVHController
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <ArticulatedSystemPlugin/ArticulatedHierarchyBVHController.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/simulation/Node.h>

namespace sofa::component::controller
{

void ArticulatedHierarchyBVHController::init()
{
    sofa::simulation::Node* curNode = dynamic_cast<sofa::simulation::Node*>(this->getContext());
    if (curNode)
    {
        curNode->getTreeObjects<ArticulationCenter, ArtCenterVec >(&m_artCenterVec);
        curNode->getTreeObject(ahc);
        frame = 0;
        n=0;
    }
}

void ArticulatedHierarchyBVHController::reset()
{
    frame = 0;
    n=0;
}

void ArticulatedHierarchyBVHController::applyController(void)
{
    double ndiv = 0.0;
    int frameInc = 0;
    double alpha;

    if (useExternalTime.getValue())
    {

        frame = (int)(floor(externalTime.getValue() / ahc->dtbvh));

        const double residu = (externalTime.getValue() / ahc->dtbvh) - (double) frame;

        msg_info() << "externalTime.getValue() = "<<externalTime.getValue() <<"  frame= "<<frame<<" residu = "<<residu ;

        if(frame > ahc->numOfFrames-2)
        {
            frame = ahc->numOfFrames-2;
        }

        alpha = residu;
    }
    else
    {

        ndiv = 1.0/ahc->dtbvh;

        alpha= (n/ndiv);



        frameInc = 1;
        if (ndiv < 1.0)
            frameInc = int(ahc->numOfFrames/(ahc->numOfFrames/ahc->dtbvh));
    }

    for (unsigned int i=0; i<m_artCenterVec.size(); i++)
    {
        ArtCenterVecIt artCenterIt = m_artCenterVec.begin();
        ArtCenterVecIt artCenterItEnd = m_artCenterVec.end();

        while ((artCenterIt != artCenterItEnd))
        {
            ArtVecIt it = (*artCenterIt)->articulations.begin();
            ArtVecIt itEnd = (*artCenterIt)->articulations.end();
            while (it != itEnd)
            {
                std::vector< core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>* > articulatedObjects;

                sofa::simulation::Node* curNode = dynamic_cast<sofa::simulation::Node*>(this->getContext());
                if (curNode)
                    curNode->getTreeObjects<core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>, std::vector< core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>* > >(&articulatedObjects);

                if (!articulatedObjects.empty())
                {
                    // Reference potential initial articulations value for interaction springs
                    // and Current articulation value at the corresponding artculation

                    std::vector< core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>* >::iterator articulatedObjIt = articulatedObjects.begin();
                    //std::vector< core::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>* >::iterator articulatedObjItEnd = articulatedObjects.end();

                    //  while (articulatedObjIt != articulatedObjItEnd)
                    if ((*it)->translation.getValue())
                    {
                        const double diffMotions = (*it)->motion[frame+1] - (*it)->motion[frame];
                        helper::WriteAccessor<Data<sofa::defaulttype::Vec1Types::VecCoord> > x = *(*articulatedObjIt)->write(sofa::core::vec_id::write_access::position);
                        this->getContext()->getMechanicalState()->vRealloc( sofa::core::MechanicalParams::defaultInstance(), core::vec_id::write_access::freePosition ); // freePosition is not allocated by default
                        helper::WriteAccessor<Data<sofa::defaulttype::Vec1Types::VecCoord> > xfree = *(*articulatedObjIt)->write(sofa::core::vec_id::write_access::freePosition);

                        x[(*it)->articulationIndex.getValue()] = (*it)->motion[frame] + alpha*diffMotions;
                        xfree[(*it)->articulationIndex.getValue()] = (*it)->motion[frame] + alpha*diffMotions;
                    }
                    else
                    {
                        const double diffMotions = (((*it)->motion[frame+1]/180.0)*3.14) - (((*it)->motion[frame]/180.0)*3.14);
                        helper::WriteAccessor<Data<sofa::defaulttype::Vec1Types::VecCoord> > x = *(*articulatedObjIt)->write(sofa::core::vec_id::write_access::position);
                        helper::WriteAccessor<Data<sofa::defaulttype::Vec1Types::VecCoord> > xfree = *(*articulatedObjIt)->write(sofa::core::vec_id::write_access::freePosition);
                        x[(*it)->articulationIndex.getValue()] = (((*it)->motion[frame]/180.0)*3.14) + alpha*diffMotions;
                        xfree[(*it)->articulationIndex.getValue()] = (((*it)->motion[frame]/180.0)*3.14) + alpha*diffMotions;
                    }
                }
                ++it;
            }
            ++artCenterIt;
        }
    }

    if (!useExternalTime.getValue())
    {
        if (frame<(ahc->numOfFrames-2))
        {
            if (n<ndiv-1)
            {
                n++;
            }
            else
            {
                if((frame+frameInc) <= (ahc->numOfFrames-2))
                    frame+=frameInc;
                n=0;
            }
        }
    }
}

// Register in the Factory
int ArticulatedHierarchyBVHControllerClass = core::RegisterObject("Implements a handler that controls the values of the articulations of an articulated hierarchy container using a .bvh file.")
        .add< ArticulatedHierarchyBVHController >()
        ;
} // namespace sofa::component::controller
