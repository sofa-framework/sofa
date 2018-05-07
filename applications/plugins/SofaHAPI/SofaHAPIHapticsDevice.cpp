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

#include "SofaHAPIHapticsDevice.h"
#include "conv.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/BackTrace.h>

#include <sstream>

namespace sofa
{

	namespace component
	{

		using sofa::core::objectmodel::HapticDeviceEvent;


		bool SofaHAPIHapticsDevice::initDevice()
		{
			sout << "Available device drivers :";
			for( std::list< HAPI::HAPIHapticsDevice::HapticsDeviceRegistration >::iterator i =
					HAPI::HAPIHapticsDevice::registered_devices->begin();
					i != HAPI::HAPIHapticsDevice::registered_devices->end(); ++i )
			{
				sout << " " << i->name;
			}
			sout << sendl;

			std::string name = this->driverName.getValue();
			for( std::list< HAPI::HAPIHapticsDevice::HapticsDeviceRegistration >::iterator i =
					HAPI::HAPIHapticsDevice::registered_devices->begin();
					i != HAPI::HAPIHapticsDevice::registered_devices->end(); ++i )
			{
				if( i->name == name )
				{
					sout << "Instantiating device driver " << i->name << sendl;
					HAPI::HAPIHapticsDevice* d = (i->create_func)();
					device.reset(d);
					break;
				}
			}
			if (!device.get())
			{
				serr << "Device driver " << name << " not found" << sendl;
				return false;
			}

			if( device->initDevice() != HAPI::HAPIHapticsDevice::SUCCESS )
			{
				serr << device->getLastErrorMsg() << sendl;
				device->releaseDevice();
				device.reset();
				return false;
			}

			sout << "Enabling device" << sendl;
			device->enableDevice();

			return true;
		}

		SofaHAPIHapticsDevice::SofaHAPIHapticsDevice()
			: scale(initData(&scale, 1.0, "scale","Default scale applied to the Phantom Coordinates. "))
			, forceScale(initData(&forceScale, 1.0, "forceScale","Default forceScale applied to the force feedback. "))
			, positionBase(initData(&positionBase, Vec3d(0,0,0), "positionBase","Position of the interface base in the scene world coordinates"))
			, orientationBase(initData(&orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the interface base in the scene world coordinates"))
			, positionTool(initData(&positionTool, Vec3d(0,0,0), "positionTool","Position of the tool in the device end effector frame"))
			, orientationTool(initData(&orientationTool, Quat(0,0,0,1), "orientationTool","Orientation of the tool in the device end effector frame"))
			, permanent(initData(&permanent, false, "permanent" , "Apply the force feedback permanently"))
			, toolSelector(initData(&toolSelector, false, "toolSelector", "Switch tools with 2nd button"))
			, toolCount(initData(&toolCount, 1, "toolCount", "Number of tools to switch between"))
			, toolIndex(initData(&toolIndex, 0, "toolIndex", "Current tool index"))
			, toolTransitionSpringStiffness(initData(&toolTransitionSpringStiffness, 0.0, "toolTransitionSpringStiffness", "Stiffness of haptic springs when switching instruments (0 to disable)"))
			, driverName(initData(&driverName, std::string("Any"), "driverName", "Name of the HAPI device driver"))
			, drawDevice(initData(&drawDevice, false, "drawDevice", "Visualize the position of the interface in the virtual scene"))
			, drawHandleSize(initData(&drawHandleSize, 0.0f, "drawHandleSize", "Visualize the handle direction of the interface in the virtual scene"))
			, drawForceScale(initData(&drawForceScale, 0.0f, "drawForceScale", "Visualize the haptics force in the virtual scene"))
			, feedbackEffects(initLink("feedbackEffects", "Force feedback effects list"))
			, mState(NULL)
			, isToolControlled(true)
			, fakeButtonState(0)
			, lastButtonState(0)
		{
			this->f_listening.setValue(true);
		}

		SofaHAPIHapticsDevice::~SofaHAPIHapticsDevice()
		{
			releaseDevice();
		}

		void SofaHAPIHapticsDevice::cleanup()
		{
		}

		void SofaHAPIHapticsDevice::releaseDevice()
		{
			bool xfer = false;
			for (int i=feedbackEffects.size()-1; i>=0; --i)
			{
				SofaHAPIForceFeedbackEffect::SPtr ffe = feedbackEffects[i];
				removeSlave(ffe.get());
				if (device.get() && ffe->getEffect())
				{
					device->removeEffect(ffe->getEffect());
					xfer = true;
				}
				feedbackEffects.remove(ffe);
			}
			//feedbackEffects.clear();
			if (device.get())
			{
				if (xfer)
					device->transferObjects();
				device->disableDevice();
				device->releaseDevice();
				device.reset();
			}
		}

		void SofaHAPIHapticsDevice::setForceFeedbacks(vector<ForceFeedback*> ffs)
		{
			bool xfer = false;
			for (int i=feedbackEffects.size()-1; i>=0; --i)
			{
				SofaHAPIForceFeedbackEffect::SPtr ffe = feedbackEffects[i];
				removeSlave(ffe.get());
				if (device.get() && ffe->getEffect())
				{
					device->removeEffect(ffe->getEffect());
					xfer = true;
				}
				feedbackEffects.remove(ffe);
			}
			//feedbackEffects.clear();
			if (xfer)
				device->transferObjects();
			for (unsigned int i=0; i<ffs.size(); ++i)
			{
				SofaHAPIForceFeedbackEffect::SPtr ffe = sofa::core::objectmodel::New<SofaHAPIForceFeedbackEffect>();
				ffe->setForceFeedback(ffs[i]);
				std::ostringstream name;
				name << "Tool"<<ffs[i]->indice.getValue() <<"-" << ffs[i]->getName();
				ffe->setName(name.str());
				ForceFeedbackEffect* e = ffe->getEffect();
				e->setTransform(data);
				e->permanent_feedback = permanent.getValue();
				addSlave(ffe.get());
				feedbackEffects.add(ffe);
			}
		}

		void SofaHAPIHapticsDevice::init()
		{
			mState = dynamic_cast<MechanicalState<sofa::defaulttype::Rigid3dTypes> *> (this->getContext()->getMechanicalState());
			if (!mState) serr << "SofaHAPIHapticsDevice has no binding MechanicalState" << sendl;
			else sout << "[Device] init" << sendl;

			if(mState && mState->getSize()<toolCount.getValue())
				mState->resize(toolCount.getValue());
		}

		void SofaHAPIHapticsDevice::bwdInit()
		{
			sofa::core::objectmodel::BaseContext* context = this->getContext();
			vector<ForceFeedback*> ffs;
			context->get<ForceFeedback>(&ffs, sofa::core::objectmodel::BaseContext::SearchRoot);
			sout << ffs.size()<<" ForceFeedback objects found:";
			for ( size_t i= 0; i < ffs.size(); ++i ) {
				sout << " " << ffs[i]->getContext()->getName()<<"/"<<ffs[i]->getName();
			}
			sout << sendl;

			setDataValue();
			setForceFeedbacks(ffs);

			if(!device.get() && !initDevice())
			{
				serr<<"NO DEVICE"<<sendl;
				releaseDevice();
			}
			else
			{
				if (isToolControlled)
					setToolFeedback(toolIndex.getValue());
			}
		}

		void SofaHAPIHapticsDevice::setDataValue()
		{
			data.scale = scale.getValue();
			data.forceScale = forceScale.getValue();
			Quat q = orientationBase.getValue();
			q.normalize();
			orientationBase.setValue(q);
			data.world_H_baseDevice.set( positionBase.getValue(), q);
			q=orientationTool.getValue();
			q.normalize();
			orientationTool.setValue(q);
			data.endDevice_H_virtualTool.set(positionTool.getValue(), q);
		}

		void SofaHAPIHapticsDevice::reset()
		{
			this->reinit();
		}

		void SofaHAPIHapticsDevice::reinit()
		{
			this->bwdInit();
		}

		void SofaHAPIHapticsDevice::draw(const sofa::core::visual::VisualParams* vparams)
		{
			if (!device.get()) return;
			if(drawDevice.getValue())
			{
				// TODO
				HAPI::HAPIHapticsDevice::DeviceValues dv = device->getDeviceValues();
				/// COMPUTATION OF THE virtualTool 6D POSITION IN THE World COORDINATES
				Vec3d pos = conv(dv.position);
				Vec3d force = conv(dv.force);
				Quat quat = conv(dv.orientation);
				quat.normalize();
				Transform baseDevice_H_endDevice(pos*data.scale, quat);
				Transform world_H_virtualTool = data.world_H_baseDevice * baseDevice_H_endDevice * data.endDevice_H_virtualTool;
				Vec3d wpos = world_H_virtualTool.getOrigin();

				vparams->drawTool()->setLightingEnabled(true); //Enable lightning
				if (drawHandleSize.getValue() == 0.0f)
				{
					std::vector<Vec3d> points;
					points.push_back(wpos);
					vparams->drawTool()->drawSpheres(points, 1.0f, sofa::defaulttype::Vec<4,float>(0,0,1,1));
				}
				else
				{
					Vec3d wposH = wpos + data.world_H_baseDevice.projectVector( baseDevice_H_endDevice.projectVector(Vec3d(0.0,0.0,drawHandleSize.getValue())));
					vparams->drawTool()->drawArrow(wposH, wpos, drawHandleSize.getValue()*0.05f, sofa::defaulttype::Vec<4,float>(0,0,1,1));
				}
				if (force.norm() > 0 && drawForceScale.getValue() != 0.0f)
				{
					//std::cout << "F = " << force << std::endl;
					Vec3d fscaled = force*(drawForceScale.getValue()*data.scale);
					Transform baseDevice_H_endDeviceF(pos*data.scale+fscaled, quat);
					Transform world_H_virtualToolF = data.world_H_baseDevice * baseDevice_H_endDeviceF * data.endDevice_H_virtualTool;
					Vec3d wposF = world_H_virtualToolF.getOrigin();
					vparams->drawTool()->drawArrow(wpos, wposF, 0.1f, sofa::defaulttype::Vec<4,float>(1,0,0,1));
				}

				vparams->drawTool()->setLightingEnabled(false); //Disable lightning
			}
		}

		void SofaHAPIHapticsDevice::sendHapticDeviceEvent()
		{
			HapticDeviceEvent event(
				toolIndex.getValue(),
				lastToolPosition.getOrigin(),
				lastToolPosition.getOrientation(),
				lastButtonState);
			getContext()->getRootContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &event);
		}

		void SofaHAPIHapticsDevice::setToolFeedback(int indice, bool enable, bool transfer)
		{
			sout << (enable ? "Enable" : "Disable") << " FFB on tool " << indice << sendl;
			if (!device.get()) return;
			for (unsigned int i=0; i < feedbackEffects.size(); ++i)
			{
				SofaHAPIForceFeedbackEffect::SPtr ffe = feedbackEffects[i];
				if (ffe->getIndice() != indice) continue;
				if (enable)
				{
					if (ffe->getForceFeedback())
						ffe->getForceFeedback()->setReferencePosition(lastToolPosition);
					sout << "Enable effect " << ffe->getName() << sendl;
					device->addEffect(ffe->getEffect());
				}
				else
				{
					sout << "Disable effect " << ffe->getName() << sendl;
					device->removeEffect(ffe->getEffect());
				}
			}
			if (transfer)
				device->transferObjects();
		}


		void SofaHAPIHapticsDevice::onBeginAnimationStep(const double /*dt*/)
		{
			if (!device.get()) return;
			sofa::helper::AdvancedTimer::stepBegin("SofaHAPIHapticsDevice");

			sofa::helper::AdvancedTimer::stepBegin("DeviceValues");
			HAPI::HAPIHapticsDevice::DeviceValues dv = device->getDeviceValues();
			sofa::helper::AdvancedTimer::stepEnd("DeviceValues");

			sofa::helper::AdvancedTimer::stepBegin("XForm");
			/// COMPUTATION OF THE virtualTool 6D POSITION IN THE World COORDINATES
			Vec3d pos = conv(dv.position);
			Quat quat = conv(dv.orientation);
			Transform baseDevice_H_endDevice(pos*data.scale, quat);
			Transform world_H_virtualTool = data.world_H_baseDevice * baseDevice_H_endDevice * data.endDevice_H_virtualTool;
			lastToolPosition = world_H_virtualTool;
			/*
				Transform baseDevice_H_endDevice2 = data.world_H_baseDevice.inversed() * world_H_virtualTool * data.endDevice_H_virtualTool.inversed();
				sout << "bHe = " << baseDevice_H_endDevice << sendl;
				sout << "wHb = " << data.world_H_baseDevice << sendl;
				sout << "dHt = " << data.endDevice_H_virtualTool << sendl;
				sout << "wHv = " << world_H_virtualTool << sendl;
				sout << "bHe2 = " << baseDevice_H_endDevice2 << sendl;
				sout << sendl;
			*/
			sofa::helper::AdvancedTimer::stepEnd("XForm");


			sofa::helper::AdvancedTimer::stepBegin("Button");

			int buttonState = fakeButtonState | device->getButtonStatus();

			int buttonChanged = buttonState ^ lastButtonState;
			// special case: btn2 is mapped to tool selection if "toolSelector" is used
			if (toolSelector.getValue() && (buttonChanged & HapticDeviceEvent::Button2StateMask))
			{
				if ((buttonState & HapticDeviceEvent::Button2StateMask) != 0)
				{
					// start tool switch : disable feedback on previous instrument
					int currentToolIndex = toolIndex.getValue();
					int newToolIndex = ((currentToolIndex+1)%toolCount.getValue());
					toolIndex.setValue(newToolIndex);
					if (toolTransitionSpringStiffness.getValue() != 0.0 && mState)
					{
						sout << "Enabling tool transition spring" << sendl;

						sofa::helper::ReadAccessor<Data<sofa::helper::vector<sofa::defaulttype::RigidCoord<3,double> > > > x = *this->mState->read(sofa::core::VecCoordId::position());
						Transform world_H_virtualTool(x[newToolIndex].getCenter(), x[newToolIndex].getOrientation());
						Transform baseDevice_H_endDevice2 = data.world_H_baseDevice.inversed() * world_H_virtualTool * data.endDevice_H_virtualTool.inversed();
						transitionEffect.reset(
							new HAPI::HapticSpring( conv(baseDevice_H_endDevice2.getOrigin()*(1/data.scale)),
									toolTransitionSpringStiffness.getValue()));
						device->addEffect(transitionEffect.get(), 1.0);
					}
					setToolFeedback(currentToolIndex, false);
					isToolControlled = false; // we disable update of the tool position and feedback until the button is released
				}
				else
				{
					if (transitionEffect.get())
					{
						sout << "Disabling tool transition spring" << sendl;
						device->removeEffect(transitionEffect.get());
						transitionEffect.reset();
					}
					setToolFeedback(toolIndex.getValue(), true);
					isToolControlled = true;
				}
			}
			sofa::helper::AdvancedTimer::stepNext("Button", "Event");

			if (buttonState != lastButtonState)
			{
				lastButtonState = buttonState;
				sendHapticDeviceEvent();
			}
			sofa::helper::AdvancedTimer::stepEnd("Event");

			if (isToolControlled) // ignore haptic device if tool is unselected
			{
				const int currentToolIndex = toolIndex.getValue();
				sofa::helper::AdvancedTimer::stepBegin("FFB");
				// store actual position of interface for the forcefeedback (as it will be used as soon as new LCP will be computed)
				for (unsigned int i=0; i < feedbackEffects.size(); ++i)
				{
					SofaHAPIForceFeedbackEffect::SPtr ffe = feedbackEffects[i];
					if (ffe->getIndice() != currentToolIndex) continue;
					if (ffe->getForceFeedback())
						ffe->getForceFeedback()->setReferencePosition(world_H_virtualTool);
				}
				sofa::helper::AdvancedTimer::stepEnd("FFB");
				if (mState)
				{
					sofa::helper::AdvancedTimer::stepBegin("SetState");
					/// TODO : SHOULD INCLUDE VELOCITY !!

					sofa::helper::WriteAccessor<Data<sofa::helper::vector<sofa::defaulttype::RigidCoord<3,double> > > > x = *this->mState->write(sofa::core::VecCoordId::position());
					sofa::helper::WriteAccessor<Data<sofa::helper::vector<sofa::defaulttype::RigidCoord<3,double> > > > xfree = *this->mState->write(sofa::core::VecCoordId::freePosition());

					xfree[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();
					x[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();

					//      std::cout << world_H_virtualTool << std::endl;

					xfree[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();
					x[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();

					sofa::helper::AdvancedTimer::stepEnd("SetState", "UpdateMapping");
					sofa::simulation::Node *node = dynamic_cast<sofa::simulation::Node*> (this->getContext());
					if (node)
					{
						sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
						sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
						sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(node);
						sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");
					}
				}
			}
			else
			{
			}

			sofa::helper::AdvancedTimer::stepEnd("SofaHAPIHapticsDevice");
		}

		void SofaHAPIHapticsDevice::onEndAnimationStep(const double /*dt*/)
		{
			if (!device.get()) return;
		}

		void SofaHAPIHapticsDevice::onKeyPressedEvent(KeypressedEvent *kpe)
		{
			if (!device.get()) return;
			switch (kpe->getKey())
			{
			case 'H': case 'h':
			{
				sout << "emulated button 1 pressed" << sendl;
				fakeButtonState |= (1<<0); //HapticDeviceEvent::Button1StateMask;
				sendHapticDeviceEvent();
				break;
			}
			case 'J': case 'j':
			{
				sout << "emulated button 2 pressed" << sendl;
				fakeButtonState |= (1<<1); //HapticDeviceEvent::Button2StateMask;
				sendHapticDeviceEvent();
				break;
			}
			default: break;
			}
		}

		void SofaHAPIHapticsDevice::onKeyReleasedEvent(KeyreleasedEvent *kre)
		{
			if (!device.get()) return;
			switch (kre->getKey())
			{
			case 'H': case 'h':
			{
				sout << "emulated button 1 released" << sendl;
				fakeButtonState &= ~(1<<0); //HapticDeviceEvent::Button1StateMask;
				sendHapticDeviceEvent();
				break;
			}
			case 'J': case 'j':
			{
				sout << "emulated button 2 released" << sendl;
				fakeButtonState &= ~(1<<1); //HapticDeviceEvent::Button2StateMask;
				sendHapticDeviceEvent();
				break;
			}
			default: break;
			}
		}

		int SofaHAPIHapticsDeviceClass = sofa::core::RegisterObject("HAPI-based Haptics Device")
				.add< SofaHAPIHapticsDevice >()
				.addAlias("HAPIHapticsDevice")
				.addAlias("DefaultHapticsDevice")
				;

		SOFA_DECL_CLASS(SofaHAPIHapticsDevice)

	} // namespace SofaHAPI
}
