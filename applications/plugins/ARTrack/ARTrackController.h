/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_H

#include <sofa/core/componentmodel/behavior/BaseController.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <ARTrackEvent.h>

namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;

template<class DataTypes>
class ARTrackController : public virtual core::componentmodel::behavior::BaseController
{
public:
    /**
     * @brief Default Constructor.
     */
    ARTrackController() {};

    /**
     * @brief Default Destructor.
     */
    virtual ~ARTrackController() {};

    void onARTrackEvent(core::objectmodel::ARTrackEvent *aev);

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
        if (dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        sofa::core::objectmodel::BaseObject::create(obj, context, arg);
        if (context)
        {
            obj->mstate = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState());
        }
    }

protected:
    core::componentmodel::behavior::MechanicalState<DataTypes> *mstate; ///< Controlled MechanicalState.
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class ARTrackController<defaulttype::Vec1dTypes>;
extern template class ARTrackController<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class ARTrackController<defaulttype::Vec1fTypes>;
extern template class ARTrackController<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_H
