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
