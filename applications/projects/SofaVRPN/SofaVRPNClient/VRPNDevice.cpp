/*
 * VRPNDevice.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#include "VRPNDevice.h"

namespace sofavrpn
{

namespace client
{

VRPNDevice::VRPNDevice()
    : deviceName(initData(&deviceName, std::string("Dummy"), "deviceName", "Name of this device"))
    , serverName(initData(&serverName, std::string("127.0.0.1"), "serverName", "VRPN server name"))
    , serverPort(initData(&serverPort, (unsigned int) 3883, "serverPort", "VRPN server port"))
{
    // TODO Auto-generated constructor stub

}

VRPNDevice::~VRPNDevice()
{
    // TODO Auto-generated destructor stub
}

void VRPNDevice::init()
{
    this->reinit();
}

void VRPNDevice::reinit()
{
    deviceURL = deviceName.getValue() + std::string("@") + serverName.getValue();

    bool connected = connect();
    if (!connected)
        std::cout << getName() << " : Not Connected" << std::endl;

}

bool VRPNDevice::connect()
{
    std::cout << "Opening: " << deviceURL << "." << std::endl;

    return connectToServer();
}

}

}
