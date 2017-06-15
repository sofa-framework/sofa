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
 * VRPNButton.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#include "VRPNButton.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofavrpn
{

namespace client
{

int VRPNButtonClass = sofa::core::RegisterObject("VRPN Tracker")
        .add< VRPNButton >();

SOFA_DECL_CLASS(VRPNButton)

void handle_button(void * /*userdata*/, const vrpn_BUTTONCB b)
{
    printf("\nButton %3d is in state: %d                      \n",
            b.button, b.state);
    fflush(stdout);
}

VRPNButton::VRPNButton()
{
    // TODO Auto-generated constructor stub

}

VRPNButton::~VRPNButton()
{
    // TODO Auto-generated destructor stub
}

bool VRPNButton::connectToServer()
{
    btn = new vrpn_Button_Remote(deviceURL.c_str());
    btn->register_change_handler(NULL, handle_button);

    //main interactive loop

    while (1)
    {
        // Let the tracker do its thing
        btn->mainloop();
    }

    return true;
}




void VRPNButton::update()
{
//	if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
//	{
//		switch(ev->getKey())
//		{
//
//			case 'T':
//			case 't':
//				std::cout << "Tracker : " << std::endl;
//				break;
//		}
//	}
}

}

}
