/*
 * VRPNDevice.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef VRPNDEVICE_H_
#define VRPNDEVICE_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseController.h>

namespace sofavrpn
{

namespace client
{

class VRPNDevice :  public virtual sofa::core::objectmodel::BaseObject, public sofa::core::behavior::BaseController
{
public:
    SOFA_CLASS(VRPNDevice,sofa::core::objectmodel::BaseObject);

private:
    bool connect();
    void handleEvent(sofa::core::objectmodel::Event *);

protected:
    virtual bool connectToServer() =0;
    virtual void update() =0;

public:
    Data<std::string> deviceName;
    Data<std::string> serverName;
    Data<std::string> serverPort;

    std::string deviceURL;

    VRPNDevice();
    virtual ~VRPNDevice();

    virtual void init();
    virtual void reinit();
};

}

}

#endif /* VRPNDEVICE_H_ */
