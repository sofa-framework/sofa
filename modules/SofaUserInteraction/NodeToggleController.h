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
//
// C++ Interface: NodeToggleController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTROLLER_NODETOGGLECONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_NODETOGGLECONTROLLER_H
#include "config.h"

#include <SofaUserInteraction/Controller.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace controller
{

/**
 * @brief NodeToggleController Class
 *
 * Provides a Mouse & Keyboard user control on a Mechanical State.
 * On a Rigid Particle, relative and absolute control is available.
 */
class NodeToggleController : public Controller
{
public:
    SOFA_CLASS(NodeToggleController, Controller);
protected:
    /**
     * @brief Default Constructor.
     */
    NodeToggleController();

    /**
     * @brief Default Destructor.
     */
    virtual ~NodeToggleController() {};
public:
    /**
     * @brief SceneGraph callback initialization method.
     */
    void init() override;

    /**
     * @brief Switch between subnodes
     */
    void toggle();


    /**
     * @name Controller Interface
     */
    //@{

    /**
     * @brief HapticDevice event callback.
     */
    void onHapticDeviceEvent(core::objectmodel::HapticDeviceEvent *mev) override;

	    /**
    * @brief Key Press event callback.
    */
    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *oev) override;
	
    /**
     * @brief Mouse event callback.
     */
//	void onMouseEvent(core::objectmodel::MouseEvent *mev);

    /**
     * @brief HapticDevice event callback.
     */
//    void onHapticDeviceEvent(core::objectmodel::HapticDeviceEvent *mev);

    /**
     * @brief Begin Animation event callback.
     */
    void onBeginAnimationStep(const double /*dt*/) override;

    //@}

    /**
     * @name Accessors
     */
    //@{


    //@}

    Data<char> d_key; ///< Key chosen for toggling the node(s)
    Data<std::string> d_nameNode; ///< Name of a specific node to toggle
    Data<bool> d_initStatus; ///< If one node is chosen, this gives the initial status of the node
    Data<bool> d_firstFrame; ///< Toggle the node at first step

protected:
    sofa::simulation::Node * specificNode;
    bool nodeFound;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_NODETOGGLECONTROLLER_H
