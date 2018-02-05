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
#ifndef SOFA_CORE_OBJECTMODEL_DETACHNODEEVENT_H
#define SOFA_CORE_OBJECTMODEL_DETACHNODEEVENT_H

#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  Event indicating that a child node is being detached from the scene.
 *  Any reference to ony of its descendant (such as active contacts) should be removed.
*/
class SOFA_CORE_API DetachNodeEvent : public Event
{
public:

    SOFA_EVENT_H( DetachNodeEvent )

    DetachNodeEvent( BaseNode* n );

    ~DetachNodeEvent();

    BaseNode* getNode() const;

    bool contains(BaseNode* n) const;

    bool contains(BaseObject* o) const;

    virtual const char* getClassName() const { return "DetachNodeEvent"; }
protected:
    BaseNode* node;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
