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
    SOFA_CLASS(VRPNDevice,sofa::core::objectmodel::BaseObject);

private:
    bool connect();

protected:
    virtual bool connectToServer() =0;

public:
    Data<std::string> deviceName;
    Data<std::string> serverName;
    Data<unsigned int> serverPort;

    std::string deviceURL;

    VRPNDevice();
    virtual ~VRPNDevice();

    virtual void init();
    virtual void reinit();
};

}

}

#endif /* VRPNDEVICE_H_ */
