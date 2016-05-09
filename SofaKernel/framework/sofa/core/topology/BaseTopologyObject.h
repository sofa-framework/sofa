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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_TOPOLOGY_BASETOPOLOGYOBJECT_H
#define SOFA_CORE_TOPOLOGY_BASETOPOLOGYOBJECT_H
#include <sofa/core/objectmodel/BaseObject.h>
namespace sofa
{

namespace core
{

namespace topology
{


class SOFA_CORE_API BaseTopologyObject : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseTopologyObject, core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseTopologyObject)

protected:
    BaseTopologyObject() {}
    virtual ~BaseTopologyObject() {}

public:

    virtual bool insertInNode( objectmodel::BaseNode* node );
    virtual bool removeInNode( objectmodel::BaseNode* node );

};

} // namespace topology

} // namespace core

} // namespace sofa

#endif
