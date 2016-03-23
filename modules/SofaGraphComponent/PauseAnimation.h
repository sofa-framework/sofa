/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_PAUSEANIMATION_H
#define SOFA_COMPONENT_MISC_PAUSEANIMATION_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/common/Simulation.h>

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
