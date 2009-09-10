/*
 * VRPNDevice.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef VRPNDEVICE_H_
#define VRPNDEVICE_H_

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofavrpn
{

namespace client
{

class VRPNDevice :  public virtual sofa::core::objectmodel::BaseObject
{
public:
    Data<std::string> deviceName;
    Data<std::string> serverName;
    Data<unsigned int> serverPort;

    std::string deviceURL;

    VRPNDevice();
    virtual ~VRPNDevice();

    virtual void init();
    virtual void reinit();

    virtual bool connectToServer() = 0;
};

}

}

#endif /* VRPNDEVICE_H_ */
