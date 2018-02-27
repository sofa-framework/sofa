/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "SofaPhysicsDataController_impl.h"

SofaPhysicsDataController::SofaPhysicsDataController()
    : impl(new Impl)
{
}

SofaPhysicsDataController::~SofaPhysicsDataController()
{
    delete impl;
}

const char* SofaPhysicsDataController::getName() ///< (non-unique) name of this object
{
    return impl->getName();
}

ID SofaPhysicsDataController::getID() ///< unique ID of this object
{
    return impl->getID();
}

void SofaPhysicsDataController::setValue(const char* v) ///< Set the value of the associated variable
{
    return impl->setValue(v);
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;


SofaPhysicsDataController::Impl::Impl()
{
}

SofaPhysicsDataController::Impl::~Impl()
{
}

const char* SofaPhysicsDataController::Impl::getName() ///< (non-unique) name of this object
{
    if (!sObj) return "";
    return sObj->getName().c_str();
}

ID SofaPhysicsDataController::Impl::getID() ///< unique ID of this object
{
    return sObj.get();
}

void SofaPhysicsDataController::Impl::setValue(const char* v) ///< Set the value of the associated variable
{
    if (sObj)
        sObj->setValue(v);
}
