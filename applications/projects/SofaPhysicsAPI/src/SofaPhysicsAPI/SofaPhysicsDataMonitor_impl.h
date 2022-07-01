/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#if __has_include(<SofaValidation/DataMonitor.h>)
#include <SofaValidation/DataMonitor.h>
#define SOFAPHYSICSAPI_HAVE_DATAMONITOR
#endif

class SofaPhysicsDataMonitor::Impl
{
public:

    Impl();
    ~Impl();

    std::string m_internalValue; ///< Store the textual representation of the internal value.

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    const char* getValue();   ///< Get the value of the associated variable


#ifdef SOFAPHYSICSAPI_HAVE_DATAMONITOR
    typedef sofa::component::misc::DataMonitor SofaDataMonitor;

protected:
    SofaDataMonitor::SPtr sObj;

public:
    SofaDataMonitor* getObject() { return sObj.get(); }
    void setObject(SofaDataMonitor* dm) { sObj = dm; }
#endif
};

#endif // SOFAPHYSICSDATAMONITOR_IMPL_H
