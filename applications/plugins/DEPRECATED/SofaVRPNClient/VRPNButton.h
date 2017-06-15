/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
 * VRPNButton.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef VRPNButton_H_
#define VRPNButton_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <VRPNDevice.h>

#include <vrpn/vrpn_Button.h>

namespace sofavrpn
{

namespace client
{

class VRPNButton :  public virtual VRPNDevice
{
public:
    SOFA_CLASS(VRPNButton,sofavrpn::client::VRPNDevice);

    VRPNButton();
    virtual ~VRPNButton();

//	void init();
//	void reinit();

private:
    vrpn_Button_Remote* btn;

    bool connectToServer();
    void update();
};

}

}

#endif /* VRPNBUTTON_H_ */
