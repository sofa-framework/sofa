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
#ifndef SOFA_COMPONENT_MISC_DEVTENSIONMONITOR_H
#define SOFA_COMPONENT_MISC_DEVTENSIONMONITOR_H
#include "config.h"

#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <SofaValidation/DevMonitor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace misc
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

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("object"))
        {
            if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(arg->findObject(arg->getAttribute("object",".."))) == NULL)
                return false;
        }
        else
        {
            if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                return false;
        }
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
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

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const DevTensionMonitor<TDataTypes>* = NULL)
    {
        return TDataTypes::Name();
    }
protected:

    core::behavior::MechanicalState<DataTypes> *mstate;

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MISC_DEVTENSIONMONITOR_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_VALIDATION_API DevTensionMonitor<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_VALIDATION_API DevTensionMonitor<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace misc

} // namespace component

} // namespace sofa

#endif
