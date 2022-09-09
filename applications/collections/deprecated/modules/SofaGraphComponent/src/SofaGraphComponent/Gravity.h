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

#include <SofaGraphComponent/config.h>

#include <sofa/type/Vec.h>
#include <sofa/core/objectmodel/ContextObject.h>

namespace sofa::simulation
{
    class Node;
} // sofa::simulation

namespace sofa::component::contextobject
{

/** Override the default gravity */
class     SOFA_ATTRIBUTE_DISABLED("v22.12 (PR#2988)", "v23.12", "Gravity class has been removed") Gravity : public core::objectmodel::ContextObject
{
public:
    SOFA_CLASS(Gravity, core::objectmodel::ContextObject);
protected:
    Gravity() = default;
public:
    SOFA_ATTRIBUTE_DISABLED("v22.12 (PR#2988)", "v23.12", "Gravity class has been removed")
    DeprecatedAndRemoved f_gravity; ///< Gravity in the world coordinate system

    void apply() override {};
};

} // namespace sofa::component::contextobject
