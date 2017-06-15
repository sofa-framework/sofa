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
#ifndef SOFA_COMPONENT_MISC_PAUSEANIMATION_H
#define SOFA_COMPONENT_MISC_PAUSEANIMATION_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace misc
{

/**
 * Abstract class defining how to pause the animation.
 */
class PauseAnimation: public core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(PauseAnimation, core::objectmodel::BaseObject);

protected:
    PauseAnimation ();
    virtual ~PauseAnimation ();
public:
    virtual void init();

    virtual bool isPaused() = 0;

    virtual void pause();

protected:
    sofa::core::objectmodel::BaseNode* root;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
