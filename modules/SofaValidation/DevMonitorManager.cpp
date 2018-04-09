/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/*
 * DevMonitorManager.cpp
 *
 *  Created on: Nov 21, 2008
 *      Author: paul
 */

#include "DevMonitorManager.h"
#include <sofa/core/ObjectFactory.h>

#include <SofaValidation/DevTensionMonitor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <algorithm>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(DevMonitorManager)

int DevMonitorManagerClass = sofa::core::RegisterObject("DevMonitorManager")
        .add< DevMonitorManager >()
        ;

using namespace sofa::defaulttype;

DevMonitorManager::DevMonitorManager()
{
    // TODO Auto-generated constructor stub

}

DevMonitorManager::~DevMonitorManager()
{
    // TODO Auto-generated destructor stub
}

void DevMonitorManager::init()
{
    sofa::helper::vector<core::DevBaseMonitor*>::iterator it;

    getContext()->get<core::DevBaseMonitor, sofa::helper::vector<core::DevBaseMonitor*> >(&monitors, core::objectmodel::BaseContext::SearchDown);

    //remove itself
    it = std::find(monitors.begin(), monitors.end(), this->toDevBaseMonitor());
    if(it != monitors.end())
        monitors.erase(it);

    sout << "Number of Monitors detected = " << monitors.size()  <<sendl;

    for (unsigned int j=0 ; j<monitors.size() ; j++)
        sout << "Monitor " << j << " " << monitors[j]->getName() << sendl;

}

void DevMonitorManager::eval()
{
    sofa::helper::vector<core::DevBaseMonitor*>::iterator it;
    sout << " Monitor Manager results :" << sendl;

    for (it = monitors.begin() ; it != monitors.end() ; ++it)
    {
        sout << "Data from Monitor " << (*it)->getName() << " : " ;

        //add cast for every monitor you want to fetch the data
        if (dynamic_cast<DevTensionMonitor<RigidTypes>*>(*it))
        {
            DevTensionMonitor<RigidTypes>* tm = dynamic_cast<DevTensionMonitor<RigidTypes>*>(*it);

            sofa::helper::vector<std::pair<Vector1, SReal> > d = tm->getData();
            for (unsigned int i=0 ; i<d.size() ; i++)
                sout << "Tension is " << d[i].first << " at " << d[i].second << sendl;
        }
        sout << sendl;
    }

}

} // namespace misc

} // namespace component

} // namespace sofa
