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

class DevMonitorManager : public DevMonitor<sofa::defaulttype::Vec3dTypes>
{
public:
    DevMonitorManager();
    virtual ~DevMonitorManager();

    void init();
    void eval();

private:
    sofa::helper::vector<core::DevBaseMonitor*> monitors;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif /* DEVMONITORMANAGER_H_ */
