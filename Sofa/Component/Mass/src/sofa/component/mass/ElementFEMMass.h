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

#include <sofa/component/mass/NodalMassDensity.h>
#include <sofa/component/mass/config.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>

#if !defined(SOFA_COMPONENT_MASS_ELEMENTFEMMASS_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

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

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size NumberOfDofsInElement = NumberOfNodesInElement * spatial_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    using ElementMassMatrix = sofa::type::Mat<NumberOfNodesInElement, NumberOfNodesInElement, sofa::Real_t<DataTypes>>;

    using NodalMassDensity = ::sofa::component::mass::NodalMassDensity<sofa::Real_t<DataTypes>>;
    using GlobalMassMatrixType = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<Real_t<DataTypes>>;

public:

    /**
     * The purpose of this function is to register the name of this class according to the provided
     * pattern.
     *
     * Example: ElementFEMMass<Vec3Types, sofa::geometry::Edge> will produce
     * the class name "EdgeFEMMass".
     */
    static const std::string GetCustomClassName()
    {
        return std::string(sofa::geometry::elementTypeToString(ElementType::Element_type)) + "FEMMass";
    }

    static const std::string GetCustomTemplateName() { return DataTypes::Name(); }

    sofa::SingleLink<ElementFEMMass, NodalMassDensity,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nodalMassDensity;

    void init() final;

    bool isDiagonal() const override { return false; }

    void addForce(const core::MechanicalParams*,
                  sofa::DataVecDeriv_t<DataTypes>& f,
                  const sofa::DataVecCoord_t<DataTypes>& x,
                  const sofa::DataVecDeriv_t<DataTypes>& v) override;

    void buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices) override;

    using Inherit1::addMDx;
    void addMDx(const core::MechanicalParams*, DataVecDeriv_t<DataTypes>& f, const DataVecDeriv_t<DataTypes>& dx, SReal factor) override;

protected:

    ElementFEMMass();

    void elementFEMMass_init();

    void validateNodalMassDensity();

    GlobalMassMatrixType m_globalMassMatrix;
};

#if !defined(SOFA_COMPONENT_MASS_ELEMENTFEMMASS_CPP)
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::mass
