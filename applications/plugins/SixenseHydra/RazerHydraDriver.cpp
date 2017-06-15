/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "RazerHydraDriver.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>

#include <SofaBaseVisual/VisualTransform.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;


// flags that the controller manager system can set to tell the graphics system to draw the instructions
// for the player
static bool controller_manager_screen_visible = true;
std::string controller_manager_text_string;

RazerHydraDriver::RazerHydraDriver()
	: scale(initData(&scale, 1.0, "scale","Default scale applied to the Leap Motion Coordinates. "))
	, positionBase(initData(&positionBase, Vec3d(0,0,0), "positionBase","Position of the interface base in the scene world coordinates"))
    , orientationBase(initData(&orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the interface base in the scene world coordinates"))
    , positionFirstTool(initData(&positionFirstTool, Vec3d(0,0,0), "positionFirstTool","Position of the first tool"))
	, positionSecondTool(initData(&positionSecondTool, Vec3d(0,0,0), "positionSecondTool","Position of the second tool"))
    , orientationFirstTool(initData(&orientationFirstTool, Quat(0,0,0,1), "orientationFirstTool","Orientation of the first tool"))
    , orientationSecondTool(initData(&orientationSecondTool, Quat(0,0,0,1), "orientationSecondTool","Orientation of the second tool"))
	, triggerJustPressedFirstTool(initData(&triggerJustPressedFirstTool, false, "triggerJustPressedFirstTool","Boolean passing to true when the trigger of the first tool is pressed"))
	, triggerJustPressedSecondTool(initData(&triggerJustPressedSecondTool, false, "triggerJustPressedSecondTool","Boolean passing to true when the trigger of the second tool is pressed"))
	, triggerValueFirstTool(initData(&triggerValueFirstTool, float(0.0), "triggerValueFirstTool","Trigger value of the first tool (between 0 and 1.0)"))
	, triggerValueSecondTool(initData(&triggerValueSecondTool, float(0.0), "triggerValueSecondTool","Trigger value of the second tool (between 0 and 1.0)"))
	, useBothTools(initData (&useBothTools, false, "useBothTools", "If true, the two controllers are used, otherwise only one controller is used"))
	, displayTools(initData (&displayTools, false, "displayTools", "display the Razer Hydra Controller joysticks as tools"))
	
{
	this->f_listening.setValue(true);
}

RazerHydraDriver::~RazerHydraDriver() {
	sixenseExit();
}

void RazerHydraDriver::cleanup() {
    sout << "RazerHydraDriver::cleanup()" << sendl;
}


void controller_manager_setup_callback( sixenseUtils::ControllerManager::setup_step /*step*/ ) {
	if( sixenseUtils::getTheControllerManager()->isMenuVisible() ) {
		controller_manager_screen_visible = true;
		controller_manager_text_string = sixenseUtils::getTheControllerManager()->getStepString();
	} else {
		controller_manager_screen_visible = false;
	}
}


sixenseUtils::IControllerManager::setup_callback(controller_manager_setup_callback_ONE_CONTROLLER )(sixenseUtils::IControllerManager::setup_step step) {
	//std::cout << " --> step: " << sixenseUtils::getTheControllerManager()->getCurrentStep();
	if(step == sixenseUtils::IControllerManager::P1C2_AIM_P1L) {
		std::cout << sixenseUtils::getTheControllerManager()->getCurrentStep() << std::endl;
	}
	return 0;
};


void RazerHydraDriver::init()
{	
	sixenseInit();
	if(useBothTools.getValue()) {
		sixenseUtils::getTheControllerManager()->setGameType(sixenseUtils::ControllerManager::ONE_PLAYER_TWO_CONTROLLER);
		sixenseUtils::getTheControllerManager()->registerSetupCallback( controller_manager_setup_callback );
	} else {
		sixenseUtils::getTheControllerManager()->setGameType(sixenseUtils::ControllerManager::ONE_PLAYER_ONE_CONTROLLER);
		//sixenseUtils::getTheControllerManager()->registerSetupCallback(controller_manager_setup_callback(sixenseUtils::IControllerManager::setup_step::P1C2_AIM_P1L));
		//sixenseUtils::getTheControllerManager()->registerSetupCallback( sixenseUtils::IControllerManager::setup_callback(*controller_manager_setup_callback_ONE_CONTROLLER(sixenseUtils::IControllerManager::setup_step::P1C2_AIM_P1L)) );
		sixenseUtils::getTheControllerManager()->registerSetupCallback( controller_manager_setup_callback );
	}
}

void RazerHydraDriver::bwdInit()
{
    sout<<"RazerHydraDriver::bwdInit()"<<sendl;	
}

void RazerHydraDriver::reset()
{
    sout<<"RazerHydraDriver::reset()" << sendl;
	//Reset useful values
    this->reinit();
}

void RazerHydraDriver::reinit()
{
    this->bwdInit();
}

void RazerHydraDriver::draw(const sofa::core::visual::VisualParams* vparams)
{
	if (!vparams->displayFlags().getShowVisualModels()) return;

	if(displayTools.getValue()) {
		helper::gl::GlText text;
		text.setText(controller_manager_text_string);
		text.update(Vec3d(-1.8,0,0));
		text.update(0.001);
	    text.draw();
		
		if(triggerValueFirstTool.getValue()>0) {
			text.setText("PAN !");
			text.update( positionFirstTool.getValue() + Vec3d(0,50.0,0).mulscalar(scale.getValue()) );
			text.update(0.001);
			text.draw();
		}

		if(triggerValueSecondTool.getValue()>0) {
			text.setText("PAN !");
			text.update( positionSecondTool.getValue() + Vec3d(0,50.0,0).mulscalar(scale.getValue()) );
			text.update(0.001);
			text.draw();
		}

		// Draw a sphere at the origin
		glutSolidSphere( 0.1, 8, 8 );

		float rot_mat[16];
		Rigid3dTypes::Coord controller;
		
		//Draw the first controller as a sphere
		glPushMatrix();
			controller = Rigid3dTypes::Coord(positionFirstTool.getValue(), orientationFirstTool.getValue());
			glColor3d(1.0f, 1.0f, 0.3f );
			controller.writeOpenGlMatrix(rot_mat);
			glMultMatrixf( (GLfloat*)rot_mat );
			glutSolidSphere( 0.1, 5, 5 );
		glPopMatrix();

		//Draw the second controller as a sphere
		//if(useBothTools.getValue()) {
			glPushMatrix();
			controller = Rigid3dTypes::Coord(positionSecondTool.getValue(), orientationSecondTool.getValue());
				glColor3d(1.0f, 0.3f, 1.0f );
				controller.writeOpenGlMatrix(rot_mat);
				glMultMatrixf( (GLfloat*)rot_mat );
				glutSolidSphere( 0.1, 5, 5 );
			glPopMatrix();
		//}
		
	}
}

void RazerHydraDriver::handleEvent(core::objectmodel::Event *event)
{
	if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
		if (ev->getKey() == 'J' || ev->getKey() == 'j')
        {
            displayTools.setValue(!displayTools.getValue());
        }
    }

	if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {	
		first_controller_index = sixenseUtils::getTheControllerManager()->getIndex( sixenseUtils::ControllerManager::P1L );
		second_controller_index = sixenseUtils::getTheControllerManager()->getIndex( sixenseUtils::ControllerManager::P1R );

		//std::cout << this->getName() << " --> (" << first_controller_index << "," << second_controller_index << ")";
		//std::cout << this->getName() << " --> step: " << sixenseUtils::getTheControllerManager()->getCurrentStep();
		//std::cout << std::endl;


		for( int base=0; base<sixenseGetMaxBases(); base++ ) {
			if(sixenseIsBaseConnected(base)) {
				sixenseSetActiveBase(base);
		
				sixenseGetAllNewestData( &acd );
				sixenseUtils::getTheControllerManager()->update( &acd );

				// For each possible controller
				for( int cont=0; cont<sixenseGetMaxControllers(); cont++ ) {
			
					// See if it's enabled
					if( sixenseIsControllerEnabled( cont ) ) {
						if (cont == first_controller_index) {
							positionFirstTool.setValue( orientationBase.getValue().rotate( Vec3d(acd.controllers[cont].pos[0], acd.controllers[cont].pos[1], acd.controllers[cont].pos[2]).mulscalar(scale.getValue()) + positionBase.getValue() ) );
							orientationFirstTool.setValue(orientationBase.getValue() * (acd.controllers[cont].rot_quat));
						} else
						if (cont == second_controller_index) {
							positionSecondTool.setValue( orientationBase.getValue().rotate( Vec3d(acd.controllers[cont].pos[0], acd.controllers[cont].pos[1], acd.controllers[cont].pos[2]).mulscalar(scale.getValue()) + positionBase.getValue() ) );
							orientationSecondTool.setValue(orientationBase.getValue() * (acd.controllers[cont].rot_quat));
						}
					}
				}
			}
		}
		check_for_button_presses( &acd );
    } // AnimatedBeginEvent
}


void RazerHydraDriver::check_for_button_presses( sixenseAllControllerData *acd ) {
	static sixenseUtils::ButtonStates first_controller_states, second_controller_states;

	first_controller_states.update( &acd->controllers[first_controller_index] );
	second_controller_states.update( &acd->controllers[second_controller_index] );

	if(first_controller_index != -1) {
		//std::cout << fabs(acd->controllers[first_controller_index].trigger) << std::endl;
		triggerValueFirstTool.setValue( fabs(acd->controllers[first_controller_index].trigger) );
	} else { triggerValueFirstTool.setValue(0.0); }

	if(second_controller_index != -1) { //acd->controllers[second_controller_index].buttons & SIXENSE_BUTTON_BUMPER) {
		//std::cout << fabs(acd->controllers[second_controller_index].trigger) << std::endl;
		triggerValueSecondTool.setValue( fabs(acd->controllers[second_controller_index].trigger) );
	} else { triggerValueSecondTool.setValue(0.0); }

	if( acd->controllers[second_controller_index].trigger != 0) {//first_controller_states.triggerJustPressed() ) {
		//std::cout<< "TRIGGER of first tool pressed" << std::endl;
		triggerJustPressedFirstTool.setValue(true);
	} else { triggerJustPressedFirstTool.setValue(false);	}

	if( second_controller_states.triggerJustPressed() ) {
		//std::cout<< "TRIGGER of second tool pressed" << std::endl;
		triggerJustPressedSecondTool.setValue(true);
	} else { triggerJustPressedSecondTool.setValue(false); }
}

int RazerHydraDriverClass = core::RegisterObject("Sixense Razer Hydra controller driver")
       .add< RazerHydraDriver >();

SOFA_DECL_CLASS(RazerHydraDriver)


} // namespace controller

} // namespace component

} // namespace sofa

