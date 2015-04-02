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
/*
 * DevMonitorManager.h
 *
 *  Created on: Nov 21, 2008
 *      Author: paul
 */

#ifndef DEVMONITORMANAGER_H_
#define DEVMONITORMANAGER_H_

#include <sofa/component/misc/DevMonitor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace misc
{

class DevMonitorManager : public DevMonitor<sofa::defaulttype::Vec3Types>
{
public:
    SOFA_CLASS(DevMonitorManager, SOFA_TEMPLATE(DevMonitor,sofa::defaulttype::Vec3Types));
protected:
    DevMonitorManager();
    virtual ~DevMonitorManager();
public:
    void init();
    void eval();

private:
    sofa::helper::vector<core::DevBaseMonitor*> monitors;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif /* DEVMONITORMANAGER_H_ */
