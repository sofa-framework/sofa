/*
 * DevMonitorManager.cpp
 *
 *  Created on: Nov 21, 2008
 *      Author: paul
 */

#include "DevMonitorManager.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/misc/DevTensionMonitor.h>
#include <sofa/defaulttype/RigidTypes.h>

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
    for (it = monitors.begin() ; (*it) != dynamic_cast<core::DevBaseMonitor*>(this) || it == monitors.end() ; it++) ;
    monitors.erase(it);

    std::cout << "Number of Monitors detected = " << monitors.size()  <<std::endl;

    for (unsigned int j=0 ; j<monitors.size() ; j++)
        std::cout << "Monitor " << j << " " << monitors[j]->getName() << std::endl;

}

void DevMonitorManager::eval()
{
    sofa::helper::vector<core::DevBaseMonitor*>::iterator it;
    std::cout << " Monitor Manager results :" << std::endl;

    for (it = monitors.begin() ; it != monitors.end() ; it++)
    {
        std::cout << "Data from Monitor " << (*it)->getName() << " : " ;

        //add cast for every monitor you want to fetch the data
        if (dynamic_cast<DevTensionMonitor<RigidTypes>*>(*it))
        {
            DevTensionMonitor<RigidTypes>* tm = dynamic_cast<DevTensionMonitor<RigidTypes>*>(*it);

            sofa::helper::vector<std::pair<Vec1d, double> > d = tm->getData();
            for (unsigned int i=0 ; i<d.size() ; i++)
                std::cout << "Tension is " << d[i].first << " at " << d[i].second << std::endl;
        }
        std::cout << std::endl;
    }

}

} // namespace misc

} // namespace component

} // namespace sofa
