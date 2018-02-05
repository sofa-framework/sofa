/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaPhysicsAPI.h"
#include "SofaPhysicsDataMonitor_impl.h"

SofaPhysicsDataMonitor::SofaPhysicsDataMonitor()
    : impl(new Impl)
{
}

SofaPhysicsDataMonitor::~SofaPhysicsDataMonitor()
{
    delete impl;
}

const char* SofaPhysicsDataMonitor::getName() ///< (non-unique) name of this object
{
    return impl->getName();
}

ID SofaPhysicsDataMonitor::getID() ///< unique ID of this object
{
    return impl->getID();
}

const char* SofaPhysicsDataMonitor::getValue() ///< Get the value of the associated variable
{
    return impl->getValue();
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;


SofaPhysicsDataMonitor::Impl::Impl()
{
}

SofaPhysicsDataMonitor::Impl::~Impl()
{
}

const char* SofaPhysicsDataMonitor::Impl::getName() ///< (non-unique) name of this object
{
    if (!sObj) return "";
    return sObj->getName().c_str();
}

ID SofaPhysicsDataMonitor::Impl::getID() ///< unique ID of this object
{
    return sObj.get();
}

const char* SofaPhysicsDataMonitor::Impl::getValue() ///< Get the value of the associated variable
{
    if (!sObj) return 0;
    return sObj->getValue();
}
