/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Interface: MechanicalStateController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_H

#include <sofa/component/controller/BaseController.h>

namespace sofa { namespace core { namespace componentmodel { namespace behavior { template<class DataTypes> class MechanicalState; } } } }


namespace sofa
{

namespace component
{

namespace controller
{

/**
 * @brief MechanicalStateController Class
 *
 * Provides a Mouse & Keyboard user control on a Mechanical State.
 * On a Rigid Particle, relative and absolute control is available.
 */
template<class DataTypes>
class MechanicalStateController : public BaseController
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    /**
     * @brief Default Constructor.
     */
    MechanicalStateController();

    /**
     * @brief Default Destructor.
     */
    virtual ~MechanicalStateController() {};

    /**
     * @brief SceneGraph callback initialization method.
     */
    void init();

    /**
     * @name BaseController Interface
     */
    //@{

    /**
     * @brief Mouse event callback.
     */
    void onMouseEvent(core::objectmodel::MouseEvent *mev);

    /**
     * @brief Begin Animation event callback.
     */
    void onBeginAnimationStep();

    //@}

    /**
     * @name Accessors
     */
    //@{

    /**
     * @brief Return the controlled MechanicalState.
     */
    core::componentmodel::behavior::MechanicalState<DataTypes> *getMechanicalState(void) const;

    /**
     * @brief Set a MechanicalState to the controller.
     */
    void setMechanicalState(core::componentmodel::behavior::MechanicalState<DataTypes> *);

    /**
     * @brief Return the index of the controlled DOF of the MechanicalState.
     */
    unsigned int getIndex(void) const;

    /**
     * @brief Set the index of the controlled DOF of the MechanicalState.
     */
    void setIndex(const unsigned int);

    /**
     * @brief Set the direction of the controlled DOF corresponding to the Mouse vertical axis.
     */
    void setMainDirection(const sofa::defaulttype::Vec<3,Real>);

    /**
     * @brief Return the direction of the controlled DOF corresponding to the Mouse vertical axis.
     */
    const sofa::defaulttype::Vec<3,Real> &getMainDirection() const;

    //@}

    /**
     * @brief Apply the controller modifications to the controlled MechanicalState.
     */
    void applyController(void);

protected:

    Data< unsigned int > index; ///< Controlled DOF index.
    core::componentmodel::behavior::MechanicalState<DataTypes> *mState; ///< Controlled MechanicalState.

    sofa::defaulttype::Vec<3,Real> mainDirection; ///< Direction corresponding to the Mouse vertical axis. Default value is (0.0,0.0,-1.0), Z axis.
    DataPtr< sofa::defaulttype::Vec<3,Real> > mainDirectionPtr; ///< Warning ! Only 3d Rigid DOFs can use this mainDirection.

    enum MouseMode {	None=0, BtLeft, BtRight, BtMiddle, Wheel }; ///< Mouse current mode.
    MouseMode mouseMode;

    int eventX, eventY; ///< Mouse current position in pixel
    int mouseSavedPosX, mouseSavedPosY; ///< Last recorded mouse position
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_H
