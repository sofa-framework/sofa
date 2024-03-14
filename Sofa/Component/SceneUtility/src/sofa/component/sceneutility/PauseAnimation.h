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
#pragma once

#include <sofa/component/sceneutility/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/type/Vec.h>

namespace sofa::component::sceneutility
{

/**
 * Abstract class defining how to pause the animation.
 */
class SOFA_COMPONENT_SCENEUTILITY_API PauseAnimation: public core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(PauseAnimation, core::objectmodel::BaseObject);

protected:
    PauseAnimation ();
    ~PauseAnimation () override;
public:
    void init() override;

    virtual bool isPaused() = 0;

    virtual void pause();

protected:
    sofa::core::objectmodel::BaseNode* root;
};

} // namespace sofa::component::sceneutility
