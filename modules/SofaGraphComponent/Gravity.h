/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONTEXTOBJECT_GRAVITY_H
#define SOFA_COMPONENT_CONTEXTOBJECT_GRAVITY_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/SofaGeneral.h>

namespace sofa
{

namespace simulation
{
class Node;
}

namespace component
{

namespace contextobject
{

/** Override the default gravity */
class SOFA_GRAPH_COMPONENT_API Gravity : public core::objectmodel::ContextObject
{
public:
    SOFA_CLASS(Gravity, core::objectmodel::ContextObject);
protected:
    Gravity();
public:
    Data<sofa::defaulttype::Vector3> f_gravity; ///< Gravity in the world coordinate system

    /// Modify the context of the Node
    void apply();
};

} // namespace contextobject

} // namespace component

} // namespace sofa

#endif

