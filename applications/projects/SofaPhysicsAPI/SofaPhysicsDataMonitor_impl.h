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
#ifndef SOFAPHYSICSDATAMONITOR_IMPL_H
#define SOFAPHYSICSDATAMONITOR_IMPL_H

#include "SofaPhysicsAPI.h"
#include <SofaValidation/DataMonitor.h>

class SofaPhysicsDataMonitor::Impl
{
public:

    Impl();
    ~Impl();

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    const char* getValue();   ///< Get the value of the associated variable

    typedef sofa::component::misc::DataMonitor SofaDataMonitor;

protected:
    SofaDataMonitor::SPtr sObj;

public:
    SofaDataMonitor* getObject() { return sObj.get(); }
    void setObject(SofaDataMonitor* dm) { sObj = dm; }
};

#endif // SOFAPHYSICSDATAMONITOR_IMPL_H
