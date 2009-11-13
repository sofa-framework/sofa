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

void handle_button(void *userdata, const vrpn_BUTTONCB b)
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
