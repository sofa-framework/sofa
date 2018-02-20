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
#ifndef SOFAHAPI_SOFAHAPIHAPTICSDEVICE_H
#define SOFAHAPI_SOFAHAPIHAPTICSDEVICE_H

#include <SofaHAPI/config.h>

#include <cstddef>
//HAPI include
#include <H3DUtil/AutoRef.h>
#include <HAPI/HAPIHapticsDevice.h>
#include <HAPI/HapticSpring.h>

#include "SofaHAPIForceFeedbackEffect.h"

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/BaseController.h>
#include <SofaUserInteraction/Controller.h>
#include <SofaHaptics/ForceFeedback.h>

namespace sofa
{

	namespace component
	{

		using sofa::helper::vector;
		using sofa::defaulttype::Vec3d;
		using sofa::defaulttype::Quat;
		using sofa::defaulttype::Rigid3dTypes;
		typedef sofa::defaulttype::SolidTypes<double>::Transform Transform;
		using sofa::core::objectmodel::Data;
		using sofa::core::objectmodel::BaseLink;
		using sofa::core::objectmodel::MultiLink;
		using sofa::core::objectmodel::KeypressedEvent;
		using sofa::core::objectmodel::KeyreleasedEvent;
		using sofa::core::behavior::MechanicalState;
		using sofa::component::controller::Controller;
		using sofa::component::controller::ForceFeedback;

		/**
		* HAPI Haptics Device
		*/
		class SOFA_SOFAHAPI_API SofaHAPIHapticsDevice : public Controller
		{
		public:
			SOFA_CLASS(SofaHAPIHapticsDevice, Controller);
			Data<double> scale; ///< Default scale applied to the Phantom Coordinates. 
			Data<double> forceScale; ///< Default forceScale applied to the force feedback. 
			Data<Vec3d> positionBase; ///< Position of the interface base in the scene world coordinates
			Data<Quat> orientationBase; ///< Orientation of the interface base in the scene world coordinates
			Data<Vec3d> positionTool; ///< Position of the tool in the device end effector frame
			Data<Quat> orientationTool; ///< Orientation of the tool in the device end effector frame
			Data<bool> permanent; ///< Apply the force feedback permanently
			Data<bool> toolSelector; ///< Switch tools with 2nd button
			Data<int> toolCount; ///< Number of tools to switch between
			Data<int> toolIndex; ///< Current tool index
			Data<double> toolTransitionSpringStiffness; ///< Stiffness of haptic springs when switching instruments (0 to disable)
			Data<std::string> driverName; ///< Name of the HAPI device driver
			Data<bool> drawDevice; ///< Visualize the position of the interface in the virtual scene
			Data<float> drawHandleSize; ///< Visualize the handle direction of the interface in the virtual scene
			Data<float> drawForceScale; ///< Visualize the haptics force in the virtual scene

			//SofaHAPIHapticsDeviceData data;
			ForceFeedbackTransform data;

			SofaHAPIHapticsDevice();
			virtual ~SofaHAPIHapticsDevice();

			virtual void init();
			virtual void bwdInit();
			virtual void reset();
			void reinit();

			bool initDevice();
			void releaseDevice();

			void cleanup();
			virtual void draw(const sofa::core::visual::VisualParams* vparams);

			void setForceFeedbacks(vector<ForceFeedback*> ffs);

			void onKeyPressedEvent(KeypressedEvent *);
			void onKeyReleasedEvent(KeyreleasedEvent *);
			void onBeginAnimationStep(const double /*dt*/);
			void onEndAnimationStep(const double /*dt*/);

			void setDataValue();

		protected:

			std::auto_ptr<HAPI::HAPIHapticsDevice> device;
			MultiLink<SofaHAPIHapticsDevice, SofaHAPIForceFeedbackEffect, BaseLink::FLAG_STRONGLINK> feedbackEffects;
			H3DUtil::AutoRef<HAPI::HapticSpring> transitionEffect;

			sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> *mState; ///< Controlled MechanicalState.

			bool isToolControlled;

			int fakeButtonState;
			int lastButtonState;
			Transform lastToolPosition;

			void setToolFeedback(int indice, bool enable = true, bool transfer = true);

			void sendHapticDeviceEvent();
		};

	} // namespace SofaHAPI
}
#endif // SOFAHAPI_SOFAHAPIHAPTICSDEVICE_H
