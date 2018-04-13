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
#include "config.h"

#include <SofaUserInteraction/Controller.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/MechanicalState.h>

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
class MechanicalStateController : public Controller
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MechanicalStateController,DataTypes),Controller);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
protected:
    /**
     * @brief Default Constructor.
     */
    MechanicalStateController();

    /**
     * @brief Default Destructor.
     */
    virtual ~MechanicalStateController() {};
public:
    /**
     * @brief SceneGraph callback initialization method.
     */
    void init() override;

    /**
     * @name Controller Interface
     */
    //@{

    /**
     * @brief Mouse event callback.
     */
    void onMouseEvent(core::objectmodel::MouseEvent *mev) override;

    /**
     * @brief HapticDevice event callback.
     */
    //void onHapticDeviceEvent(core::objectmodel::HapticDeviceEvent *mev);

    /**
     * @brief Begin Animation event callback.
     */
    void onBeginAnimationStep(const double dt) override;

    //@}

    /**
     * @name Accessors
     */
    //@{

    /**
     * @brief Return the controlled MechanicalState.
     */
    core::behavior::MechanicalState<DataTypes> *getMechanicalState(void) const;

    /**
     * @brief Set a MechanicalState to the controller.
     */
    void setMechanicalState(core::behavior::MechanicalState<DataTypes> *);

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

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const MechanicalStateController<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
protected:

    Data< unsigned int > index; ///< Controlled DOF index.
    Data< bool > onlyTranslation; ///< Controlling the DOF only in translation
    Data< bool > buttonDeviceState; ///< state of ths device button

    core::behavior::MechanicalState<DataTypes> *mState; ///< Controlled MechanicalState.

    Data< sofa::defaulttype::Vec<3,Real> > mainDirection; ///< Direction corresponding to the Mouse vertical axis. Default value is (0.0,0.0,-1.0), Z axis.

    enum MouseMode { None=0, BtLeft, BtRight, BtMiddle, Wheel }; ///< Mouse current mode.
    bool device;
    MouseMode mouseMode;

    int eventX, eventY; ///< Mouse current position in pixel
    double deviceX, deviceY, deviceZ;
    int mouseSavedPosX, mouseSavedPosY; ///< Last recorded mouse position
    sofa::defaulttype::Vector3 position;
    sofa::defaulttype::Quat orientation;
    bool buttonDevice;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_CPP)
#ifndef SOFA_FLOAT
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec3dTypes>;
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec2dTypes>;
extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec1dTypes>;
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec6dTypes>;
extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Rigid3dTypes>;
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec3fTypes>;
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec2fTypes>;
extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec1fTypes>;
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Vec6fTypes>;
extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Rigid3fTypes>;
//extern template class SOFA_USER_INTERACTION_API MechanicalStateController<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_H
