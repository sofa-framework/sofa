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
#ifndef SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_INL
#define SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_INL

#include <ARTrackController.h>
#include <ARTrackEvent.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
void ARTrackController<DataTypes>::init()
{
}

template <>
void ARTrackController<Vec1dTypes>::init()
{
    getContext()->get<sofa::component::container::ArticulatedHierarchyContainer::ArticulationCenter::Articulation>(&articulations);
}


template <>
void ARTrackController<RigidTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    if(mstate)
    {
        if(!(*mstate->getXfree()).empty() && !(*mstate->getX()).empty())
        {
            (*mstate->getXfree())[0].getCenter() = aev->getPosition();
            (*mstate->getX())[0].getCenter() = aev->getPosition();

            (*mstate->getXfree())[0].getOrientation() = aev->getOrientation();
            (*mstate->getX())[0].getOrientation() = aev->getOrientation();
        }
    }
}

template <>
void ARTrackController<Vec1dTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    if(mstate)
    {
        if(!(*mstate->getXfree()).empty() && !(*mstate->getX()).empty())
        {
            for (unsigned int i=6; i<9; ++i) // thumb
            {
                (*mstate->getXfree())[i] = aev->getAngles()[0];// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                if((*mstate->getXfree())[i].x()<0)
                {
                    (*mstate->getXfree())[i] = 0.0;
                    (*mstate->getX())[i] = 0.0;
                }
            }

            for (unsigned int i=9; i<12; ++i) // index
            {
                (*mstate->getXfree())[i] = aev->getAngles()[1];// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                if((*mstate->getXfree())[i].x()<0)
                {
                    (*mstate->getXfree())[i] = 0.0;
                    (*mstate->getX())[i] = 0.0;
                }
            }

            for(unsigned int i=12; i<21; ++i) // middle, ring, little.
            {
                (*mstate->getXfree())[i] = aev->getAngles()[2];// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                if((*mstate->getXfree())[i].x()<0)
                {
                    (*mstate->getXfree())[i] = 0.0;
                    (*mstate->getX())[i] = 0.0;
                }
            }
        }
    }
}

template <class DataTypes>
void ARTrackController<DataTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent* /*aev*/)
{
}

template <class DataTypes>
void ARTrackController<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::core::objectmodel::ARTrackEvent *>(event))
    {
        sofa::core::objectmodel::ARTrackEvent *aev = dynamic_cast<sofa::core::objectmodel::ARTrackEvent *>(event);
        onARTrackEvent(aev);
    }
}


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_H
