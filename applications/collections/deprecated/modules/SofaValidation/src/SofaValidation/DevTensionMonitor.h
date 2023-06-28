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
#include <SofaValidation/config.h>

#include <sofa/type/vector.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <SofaValidation/DevMonitor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactoryTemplateDeductionRules.h>

namespace sofa::component::misc
{

template <class TDataTypes>
class DevTensionMonitor: public virtual DevMonitor<sofa::defaulttype::Vec1Types>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DevTensionMonitor,TDataTypes), SOFA_TEMPLATE(DevMonitor,sofa::defaulttype::Vec1Types));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
protected:
    DevTensionMonitor() { };
    virtual ~DevTensionMonitor() { };
public:
    void init() override;
    void eval() override;

    /// Deduce type from contexte, this method is called by ObjectFactory.
    static std::string TemplateDeductionMethod(sofa::core::objectmodel::BaseContext* context,
                                               sofa::core::objectmodel::BaseObjectDescription* description)
    {
        return sofa::core::getTemplateFromLinkedMechanicalState("object", context, description);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* tObj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = core::objectmodel::BaseObject::create(tObj, context, arg);

        if (arg && (arg->getAttribute("object")))
        {
            obj->mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(arg->findObject(arg->getAttribute("object","..")));
        }

        return obj;
    }

protected:
    core::behavior::MechanicalState<DataTypes> *mstate;
};


#if  !defined(SOFA_COMPONENT_MISC_DEVTENSIONMONITOR_CPP)
extern template class SOFA_SOFAVALIDATION_API DevTensionMonitor<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::misc
