/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#pragma once
#include <sofa/component/linearsolver/ordering/AMDOrderingMethod.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>


namespace sofa::component::linearsolver::ordering
{

template<class TBase>
class OrderingMethodAccessor : public TBase
{
public:
    SOFA_CLASS(OrderingMethodAccessor, TBase);

    OrderingMethodAccessor()
        : l_orderingMethod(initLink("orderingMethod", "Ordering method used by this component"))
    {}

    ~OrderingMethodAccessor() override = default;
    SingleLink<OrderingMethodAccessor, core::behavior::BaseOrderingMethod, BaseLink::FLAG_STRONGLINK> l_orderingMethod;

    void init() override
    {
        Inherit1::init();

        if (!l_orderingMethod.get())
        {
            auto* orderingMethod = this->getContext()->template get<core::behavior::BaseOrderingMethod>(sofa::core::objectmodel::BaseContext::Local);
            l_orderingMethod.set(orderingMethod);

            if (orderingMethod)
            {
                msg_info() << "Ordering method link is set to " << orderingMethod->getPathName();
            }
            else
            {
                // METIS is the preferred ordering method to keep the same
                // behavior compared to the time when ordering method was not
                // an option, and METIS was systematically used.
                // The METIS ordering method is in another module and its
                // C++ type is not accessible here. That is why the object
                // factory is used.
                const std::string preferredClass = "MetisOrderingMethod";
                core::objectmodel::BaseObjectDescription description(preferredClass.c_str(), preferredClass.c_str());
                const core::objectmodel::BaseObject::SPtr baseObject = core::ObjectFactory::getInstance()->createObject(this->getContext(), &description);
                if (auto* metisOrderingMethod = dynamic_cast<core::behavior::BaseOrderingMethod*>(baseObject.get()))
                {
                    setupCreatedOrderingMethod(metisOrderingMethod);
                }
                else
                {
                    using DefaultOrderingMethod = AMDOrderingMethod;
                    if (const auto createdOrderingMethod = sofa::core::objectmodel::New<DefaultOrderingMethod>())
                    {
                        setupCreatedOrderingMethod(createdOrderingMethod.get());
                    }
                    else
                    {
                        msg_fatal() << "An OrderingMethod is required by " << this->getClassName() << " but has not been found:"
                            " a default " << DefaultOrderingMethod::GetClass()->className << " could not be automatically added in the scene. To remove this error, add"
                            " an OrderingMethod in the scene. The list of available OrderingMethod is: "
                            << core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::BaseOrderingMethod>();
                        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
                    }
                }
            }

        }
    }

private:

    void setupCreatedOrderingMethod(core::behavior::BaseOrderingMethod* createdOrderingMethod)
    {
        createdOrderingMethod->setName(this->getContext()->getNameHelper().resolveName(createdOrderingMethod->getClassName(), sofa::core::ComponentNameHelper::Convention::python));
        this->addSlave(createdOrderingMethod);
        l_orderingMethod.set(createdOrderingMethod);

        msg_info() << "An OrderingMethod is required by " << this->getClassName() << " but has not been found:"
            " a default " << createdOrderingMethod->getClassName() << " is automatically added in the scene for you. To remove this info message, add"
            " an OrderingMethod in the scene. The list of available OrderingMethod is: "
            << core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::BaseOrderingMethod>();
    }
};

}
