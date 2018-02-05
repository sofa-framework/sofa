/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include "SofaHAPIForceFeedbackEffect.h"
#include "conv.h"

#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/BackTrace.h>

// activate to check for NaN from device data
#define SOFAHAPI_CHECK_NAN

namespace sofa
{

	namespace component
	{

		using sofa::defaulttype::SolidTypes;

		ForceFeedbackEffect::ForceFeedbackEffect(ForceFeedback* forceFeedback)
			: forceFeedback(forceFeedback), permanent_feedback(true)
		{
		}

		ForceFeedbackEffect::~ForceFeedbackEffect()
		{
		}

		ForceFeedbackEffect::EffectOutput ForceFeedbackEffect::calculateForces( const EffectInput &input )
		{
			ForceFeedbackEffect::EffectOutput res;
		#ifdef SOFAHAPI_CHECK_NAN
			if (input.position.x != input.position.x ||
				input.position.y != input.position.y ||
				input.position.z != input.position.z ||
				input.orientation.axis.x != input.orientation.axis.x ||
				input.orientation.axis.y != input.orientation.axis.y ||
				input.orientation.axis.z != input.orientation.axis.z )
			{
				//std::cerr << "Invalid data received from haptic device" << std::endl;
				res.force.x = 0.0;
				res.force.y = 0.0;
				res.force.z = 0.0;
				return res;
			}
		#endif
			/// COMPUTATION OF THE virtualTool 6D POSITION IN THE World COORDINATES
			Vec3d pos = conv(input.position);
			Quat quat = conv(input.orientation);
			Transform baseDevice_H_endDevice(pos*data.scale, quat);
			Transform world_H_virtualTool = data.world_H_baseDevice * baseDevice_H_endDevice * data.endDevice_H_virtualTool;

			Vec3d world_pos_tool = world_H_virtualTool.getOrigin();
			Quat world_quat_tool = world_H_virtualTool.getOrientation();

			SolidTypes<double>::SpatialVector Twist_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0)); // Todo: compute a velocity !!
			SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0));

			ForceFeedback* ff = forceFeedback;

			///////////////// 3D rendering ////////////////

			//if (ff != NULL)
			//	ff->computeForce(world_pos_tool[0], world_pos_tool[1], world_pos_tool[2], world_quat_tool[0], world_quat_tool[1], world_quat_tool[2], world_quat_tool[3], Wrench_tool_inWorld.getForce()[0], Wrench_tool_inWorld.getForce()[1], Wrench_tool_inWorld.getForce()[2]);

			///////////////// 6D rendering ////////////////

			if (ff != NULL)
				ff->computeWrench(world_H_virtualTool,Twist_tool_inWorld,Wrench_tool_inWorld );

			// we compute its value in the current Tool frame:
			SolidTypes<double>::SpatialVector Wrench_tool_inTool(world_quat_tool.inverseRotate(Wrench_tool_inWorld.getForce()),  world_quat_tool.inverseRotate(Wrench_tool_inWorld.getTorque())  );
			// we transport (change of application point) its value to the endDevice frame
			SolidTypes<double>::SpatialVector Wrench_endDevice_inEndDevice = data.endDevice_H_virtualTool * Wrench_tool_inTool;
			// we compute its value in the baseDevice frame
			SolidTypes<double>::SpatialVector Wrench_endDevice_inBaseDevice( baseDevice_H_endDevice.projectVector(Wrench_endDevice_inEndDevice.getForce()), baseDevice_H_endDevice.projectVector(Wrench_endDevice_inEndDevice.getTorque()) );

			res.force.x = Wrench_endDevice_inBaseDevice.getForce()[0] * data.forceScale;
			res.force.y = Wrench_endDevice_inBaseDevice.getForce()[1] * data.forceScale;
			res.force.z = Wrench_endDevice_inBaseDevice.getForce()[2] * data.forceScale;
			return res;
		}

		SofaHAPIForceFeedbackEffect::SofaHAPIForceFeedbackEffect()
			: forceFeedback(initLink("forceFeedback", "the component used to compute the applied forces"))
		{
		}

		SofaHAPIForceFeedbackEffect::~SofaHAPIForceFeedbackEffect()
		{
		}

		ForceFeedbackEffect* SofaHAPIForceFeedbackEffect::getEffect()
		{
			if (!data.get() && forceFeedback.get())
			{
				data.reset(new ForceFeedbackEffect(forceFeedback.get()));
			}
			return data.get();
		}

		void SofaHAPIForceFeedbackEffect::setForceFeedback(ForceFeedback* ffb)
		{
			if (ffb != forceFeedback.get())
				data.reset(); // unref old effect if we just changed the feedback component
			forceFeedback.set(ffb);
		}

		ForceFeedback* SofaHAPIForceFeedbackEffect::getForceFeedback()
		{
			return forceFeedback.get();
		}

		int SofaHAPIForceFeedbackEffect::getIndice()
		{
			return getForceFeedback()->indice.getValue();
		}

		int SofaHAPIForceFeedbackEffectClass = sofa::core::RegisterObject("Implement HAPIForceEffect using a Sofa ForceFeedback component")
				.add< SofaHAPIForceFeedbackEffect >()
				;

		SOFA_DECL_CLASS(SofaHAPIForceFeedbackEffect)

	} // namespace SofaHAPI
}