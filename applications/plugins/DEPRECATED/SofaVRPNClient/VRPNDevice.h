/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
    SOFA_CLASS2(VRPNDevice,sofa::core::objectmodel::BaseObject, sofa::core::behavior::BaseController);

private:
    bool connect();
    void handleEvent(sofa::core::objectmodel::Event *);

protected:
    virtual bool connectToServer() =0;
    virtual void update() =0;

public:
    sofa::core::objectmodel::Data<std::string> deviceName;
    sofa::core::objectmodel::Data<std::string> serverName;
    sofa::core::objectmodel::Data<std::string> serverPort;

    std::string deviceURL;

    VRPNDevice();
    virtual ~VRPNDevice();

    virtual void init();
    virtual void reinit();
};

}

}

#endif /* VRPNDEVICE_H_ */
