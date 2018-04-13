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
#ifndef SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_H

#include <SofaUserInteraction/Controller.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaGeneralRigid/ArticulatedHierarchyContainer.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include "ARTrackEvent.h"

namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;


class ARTrackVirtualTimeController : public Controller
{
public:
	SOFA_CLASS(ARTrackVirtualTimeController, Controller);

    /**
     * @brief Default Constructor.
     */
    ARTrackVirtualTimeController();

    /**
     * @brief Default Destructor.
     */
    virtual ~ARTrackVirtualTimeController () {}

    /**
     * @brief SceneGraph callback initialization method.
     */
    void init();

    /**
     * @brief SceneGraph callback re-initialization method.
     */
    void reinit();

    virtual void reset() {init();}

    /**
     * @brief Mouse event callback.
     */


    void handleEvent(core::objectmodel::Event *);

    void onMouseEvent(core::objectmodel::MouseEvent *mev);

    void onARTrackEvent(core::objectmodel::ARTrackEvent *aev);

    /**
     * @brief Apply the controller modifications to the controlled MechanicalState.
     */
    void applyController(void);

private:
    Data< double > virtualTime; ///< Time found for the BVH
    Data< double > step1; ///< time at initial position
    Data< double > step2; ///< time at intermediate position
    Data< double > step3; ///< time at final position
    Data< double > maxMotion; ///< Displacement amplitude
    int mousePosX, mousePosY; ///< Last recorded mouse position
    int mouseWheel;
    double ARTrackMotion;
    double ARTrackResetPos;
    double ARTrackIntermediatePos; /// pos when changing from forward to backward
    double TotalMouseDisplacement;

    bool resetBool;

    bool backward;

};

template<class DataTypes>
class ARTrackController : public virtual component::controller::Controller
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(ARTrackController, DataTypes), component::controller::Controller);

    typedef typename DataTypes::VecCoord VecCoord;
    /**
     * @brief Default Constructor.
     */
    ARTrackController() {};

    /**
     * @brief Default Destructor.
     */
    virtual ~ARTrackController() {};

    void init();

    void onARTrackEvent(core::objectmodel::ARTrackEvent *aev);

    void onMouseEvent(core::objectmodel::MouseEvent *mev);

    void handleEvent(core::objectmodel::Event *);

    static std::string templateName(const ARTrackController<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* p0, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = Inherit1::create(p0, context, arg);
        if (context)
        {
            obj->mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState());
        }
        return obj;
    }

protected:
    core::behavior::MechanicalState<DataTypes> *mstate; ///< Controlled MechanicalState.
    helper::vector<sofa::component::container::Articulation*> articulations;
    bool leftPressed, rightPressed, wheel;
    Vec3d beginLocalPosition,endLocalPosition;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class ARTrackController<defaulttype::Vec1dTypes>;
extern template class ARTrackController<defaulttype::Vec3dTypes>;
extern template class ARTrackController<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class ARTrackController<defaulttype::Vec1fTypes>;
extern template class ARTrackController<defaulttype::Vec3fTypes>;
extern template class ARTrackController<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_H
