/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

//Sixense includes
#include <sixense.h>
#include <sixense_math.hpp>
#ifdef WIN32
#include <sixense_utils/mouse_pointer.hpp>
#endif
#include <sixense_utils/derivatives.hpp>
#include <sixense_utils/button_states.hpp>
#include <sixense_utils/event_triggers.hpp>
#include <sixense_utils/controller_manager/controller_manager.hpp>


#include <sofa/core/behavior/BaseController.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaUserInteraction/Controller.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/helper/gl/glText.inl>

#include <deque>


namespace sofa
{
namespace simulation { class Node; }

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;

class RazerHydraDriver : public Controller
{

public:
    SOFA_CLASS(RazerHydraDriver, Controller);	
	Data<double> scale; ///< Default scale applied to the Leap Motion Coordinates. 
	Data<Vec3d> positionBase; ///< Position of the interface base in the scene world coordinates
    Data<Quat> orientationBase; ///< Orientation of the interface base in the scene world coordinates
    Data<Vec3d> positionFirstTool; ///< Position of the first tool
    Data<Vec3d> positionSecondTool; ///< Position of the second tool
    Data<Quat> orientationFirstTool; ///< Orientation of the first tool
    Data<Quat> orientationSecondTool; ///< Orientation of the second tool
	Data< bool > triggerJustPressedFirstTool; ///< Boolean passing to true when the trigger of the first tool is pressed
	Data< bool > triggerJustPressedSecondTool; ///< Boolean passing to true when the trigger of the second tool is pressed
	Data< float > triggerValueFirstTool; ///< Trigger value of the first tool (between 0 and 1.0)
	Data< float > triggerValueSecondTool; ///< Trigger value of the second tool (between 0 and 1.0)
	Data< bool > useBothTools; ///< If true, the two controllers are used, otherwise only one controller is used
	Data< bool > displayTools; ///< display the Razer Hydra Controller joysticks as tools

    RazerHydraDriver();
    virtual ~RazerHydraDriver();

    void init();
    void bwdInit();
    void reset();
    void reinit();

    void cleanup();
	void draw(const sofa::core::visual::VisualParams* vparams);

private:
	int first_controller_index, second_controller_index;
	sixenseAllControllerData acd;

    void handleEvent(core::objectmodel::Event *);
	void check_for_button_presses( sixenseAllControllerData *acd );
	void applyRotation (Vec3d* positionToRotate, Quat* orientationToRotate);

};

} // namespace controller

} // namespace component

} // namespace sofa
