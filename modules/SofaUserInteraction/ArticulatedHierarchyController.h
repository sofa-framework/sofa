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
//
// C++ Interface: ArticulatedHierarchyController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTROLLER_ARTICULATEDHIERARCHYCONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_ARTICULATEDHIERARCHYCONTROLLER_H
#include "config.h"

#include <SofaUserInteraction/Controller.h>

#include <SofaGeneralRigid/ArticulatedHierarchyContainer.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/vector.h>

#include <map>

namespace sofa
{

namespace component
{

namespace controller
{

using sofa::component::container::ArticulatedHierarchyContainer;
using sofa::component::container::ArticulationCenter;
using sofa::component::container::Articulation;

/**
 * @brief ArticulatedHierarchyController Class.
 *
 * Implements an user interaction handler that controls the values of the
 * articulations of an articulated hierarchy container.
 * Mouse Buttons and Wheel are controlling the value.
 * Keyboard is used to select the controlled articulation.
 */
class SOFA_USER_INTERACTION_API ArticulatedHierarchyController : public Controller
{
public:
    SOFA_CLASS(ArticulatedHierarchyController,Controller);
    typedef std::vector< ArticulationCenter * > ArtCenterVec;
    typedef ArtCenterVec::iterator ArtCenterVecIt;
    typedef std::vector< Articulation * > ArtVec;
    typedef ArtVec::iterator ArtVecIt;
protected:
    /**
     * @brief Default Constructor.
     */
    ArticulatedHierarchyController();

    /**
     * @brief Default Destructor.
     */
    virtual ~ArticulatedHierarchyController() {};
public:
    /**
     * @brief Init method called during the scene graph initialization.
     */
    virtual void init();

    /**
     * @name Controller Interface
     */
    //@{

    /**
     * @brief Mouse event callback.
     */
    void onMouseEvent(core::objectmodel::MouseEvent *);

    /**
     * @brief Key press event callback.
     */
    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *);

    /**
     * @brief begin animation callback. Called at the beginning of each time step.
     */
    void onBeginAnimationStep(const double dt);
    //@}

    /**
     * @brief Apply the controller current modifications to its controled component.
     */
    virtual void applyController(void);

protected:
    Data<sofa::helper::vector< int > > articulationsIndices; ///< Stores controlled articulations indices.
    Data<sofa::helper::vector< char > > bindingKeys; ///< Stores controlled articulations keyboard keys.
    Data< double > angleDelta; ///< Angle step added at each event reception.
    Data< bool > propagateUserInteraction; ///< Says wether or not to apportion the articulation modification to its children in the hierarchy.

    sofa::helper::vector< bool > activeArticulations; ///< Stores activated articulations information.
    std::map<int, sofa::helper::vector< int > > articulationsPropagationChains;

    /**
     * @brief Build the articulations list related to each controlled articulation.
     */
    void buildPropagationArticulationsChain(void);

    /**
     * @brief Build the articulations indices list according to an Articulation and its ArticulationCenter.
     */
    void buildArray(std::vector< int > &, Articulation* , ArticulationCenter* );

    /**
     * @brief Set the active articulation from a Key Input.
     */
    void updateActiveArticulationsIndices(const char);

    /**
     * @name Debugging methods.
     */
    //@{
    void dumpActiveArticulations(void) const ;
    void dumpArticulationsAndBindingKeys(void) const;
    //}@

    /**
     * @brief Set the controller in its initial state.
     */
    void resetControler(void);


    /**
     * Current MouseMode buffered.
     */
    enum MouseMode { None=0, BtLeft, BtRight, BtMiddle, Wheel };
    MouseMode mouseMode;

    double signFactor;
    bool propagationChain;

    ArtCenterVec m_artCenterVec; ///< List of ArticulationCenters controlled by the controller.
};


} // namespace controller

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_CONTROLLER_ARTICULATEDHIERARCHYCONTROLLER_H
