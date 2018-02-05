/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace topology

} // namespace core

} // namespace sofa

#endif
