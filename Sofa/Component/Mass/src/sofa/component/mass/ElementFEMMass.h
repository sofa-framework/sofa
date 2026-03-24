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

#include <sofa/component/mass/config.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/TopologyAccessor.h>

namespace sofa::component::mass
{

template<class TDataTypes, class TElementType>
class ElementFEMMass :
    public core::behavior::Mass<TDataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    SOFA_CLASS2(SOFA_TEMPLATE2(ElementFEMMass, DataTypes, ElementType),
        core::behavior::Mass<TDataTypes>,
        sofa::core::behavior::TopologyAccessor);

    /**
     * The purpose of this function is to register the name of this class according to the provided
     * pattern.
     *
     * Example: ElementFEMMass<Vec3Types, sofa::geometry::Edge> will produce
     * the class name "EdgeFEMMass".
     */
    static const std::string GetCustomClassName()
    {
        return std::string(sofa::geometry::elementTypeToString(ElementType::Element_type)) +
               "FEMMass";
    }

    static const std::string GetCustomTemplateName() { return DataTypes::Name(); }

    void init() final;

    bool isDiagonal() const override { return false; }

    void addForce(const core::MechanicalParams*,
                  sofa::DataVecDeriv_t<DataTypes>& f,
                  const sofa::DataVecCoord_t<DataTypes>& x,
                  const sofa::DataVecDeriv_t<DataTypes>& v) override;

protected:

    void elementFEMMass_init();
};

}  // namespace sofa::component::mass
