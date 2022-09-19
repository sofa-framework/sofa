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
    DevTensionMonitor();
    virtual ~DevTensionMonitor() { };
public:
    void init() override;
    void eval() override;

    static std::string TemplateDeductionMethod(sofa::core::objectmodel::BaseContext* context,
                                               sofa::core::objectmodel::BaseObjectDescription* args);

protected:
    SingleLink<DevTensionMonitor<DataTypes>, core::behavior::MechanicalState<DataTypes>, BaseLink::FLAG_STOREPATH> mstate;
};


#if  !defined(SOFA_COMPONENT_MISC_DEVTENSIONMONITOR_CPP)
extern template class SOFA_SOFAVALIDATION_API DevTensionMonitor<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::misc
