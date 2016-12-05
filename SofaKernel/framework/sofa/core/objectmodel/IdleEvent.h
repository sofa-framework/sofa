/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: damien.marchal@univ-lille1.fr Copyright (C) CNRS                   *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_OBJECTMODEL_IDLEEVENT_H
#define SOFA_CORE_OBJECTMODEL_IDLEEVENT_H

#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
    @author Damien Marchal
    @brief This event is propagated along the objects hierarchy at regular interval.
*/
class SOFA_CORE_API IdleEvent : public Event
{
public:
    IdleEvent() {}
    virtual ~IdleEvent() {}
    SOFA_EVENT_H( IdleEvent )

    virtual const char* getClassName() const { return "IdleEvent"; }
protected:
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
