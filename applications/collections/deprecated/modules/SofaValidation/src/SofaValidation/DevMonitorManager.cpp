/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include "DevMonitorManager.h"
#include <sofa/core/ObjectFactory.h>

#include <SofaValidation/DevTensionMonitor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <algorithm>

namespace sofa::component::misc
{

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

void DevMonitorManager::doBaseObjectInit()
{
    sofa::type::vector<core::DevBaseMonitor*>::iterator it;

    getContext()->get<core::DevBaseMonitor, sofa::type::vector<core::DevBaseMonitor*> >(&monitors, core::objectmodel::BaseContext::SearchDown);

    //remove itself
    it = std::find(monitors.begin(), monitors.end(), this->toDevBaseMonitor());
    if(it != monitors.end())
        monitors.erase(it);

    msg_info() << "Number of Monitors detected = " << monitors.size();

    for (unsigned int j=0 ; j<monitors.size() ; j++)
        msg_info() << "Monitor " << j << " " << monitors[j]->getName();

}

void DevMonitorManager::eval()
{
    sofa::type::vector<core::DevBaseMonitor*>::iterator it;
    msg_info() << " Monitor Manager results :";

    for (it = monitors.begin() ; it != monitors.end() ; ++it)
    {
        msg_info() << "Data from Monitor " << (*it)->getName() << " : " ;

        //add cast for every monitor you want to fetch the data
        if (dynamic_cast<DevTensionMonitor<RigidTypes>*>(*it))
        {
            DevTensionMonitor<RigidTypes>* tm = dynamic_cast<DevTensionMonitor<RigidTypes>*>(*it);

            auto d = tm->getData();
            for (unsigned int i=0 ; i<d.size() ; i++)
                msg_info() << "Tension is " << d[i].first << " at " << d[i].second;
        }
        msg_info();
    }

}

} // namespace sofa::component::misc

