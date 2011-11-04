/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
//
// C++ Interface: DevBaseMonitor
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_CORE_DEVBASEMONITOR_H
#define SOFA_CORE_DEVBASEMONITOR_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

/**
 *  \brief A basic interface to define a Monitor capable to compute metrics.
 *
 *
 *
 */
class DevBaseMonitor : public virtual objectmodel::BaseObject
{
public:
    SOFA_CLASS(DevBaseMonitor, objectmodel::BaseObject);

    /// Destructor
    virtual ~DevBaseMonitor() { };
    /// Compute metrics
    virtual void eval() = 0;
};

} // namespace core
} // namespace sofa

#endif //SOFA_CORE_DEVBASEMONITOR_H
