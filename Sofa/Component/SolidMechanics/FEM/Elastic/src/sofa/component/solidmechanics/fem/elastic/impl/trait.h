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

#include <sofa/core/trait/DataTypes.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/type/FullySymmetric4Tensor.h>

#include <sofa/component/solidmechanics/fem/elastic/impl/StrainDisplacement.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class SpatialTypes, class ElementType>
struct trait
{
    // ---- Types of the field (displacement, temperature, concentration, ...) ----
    using DataVecCoord = sofa::DataVecCoord_t<DataTypes>;
    using DataVecDeriv = sofa::DataVecDeriv_t<DataTypes>;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using Real = sofa::Real_t<DataTypes>;

    // ---- Types of the geometry ----
    using GeometryCoord = sofa::Coord_t<SpatialTypes>;
    using GeometryReal = sofa::Real_t<SpatialTypes>;

    using FiniteElement = sofa::fem::FiniteElement<ElementType, SpatialTypes>;
    using TopologyElement = typename FiniteElement::TopologyElement;
    using ReferenceCoord = typename FiniteElement::ReferenceCoord;

    /// Dimension of the space in which the mesh lives
    static constexpr sofa::Size spatial_dimensions = SpatialTypes::coord_total_size;
    /// Number of components of the field at each node (1 for a scalar field)
    static constexpr sofa::Size field_dimensions = DataTypes::coord_total_size;

    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size NumberOfDofsInElement = NumberOfNodesInElement * field_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;
    static constexpr sofa::Size NbQuadraturePoints = FiniteElement::quadraturePoints().size();

    static_assert(field_dimensions >= 1, "The field must have at least one component");
    static_assert(TopologicalDimension <= spatial_dimensions,
                  "The element cannot have a higher dimension than the space it is embedded in");

    /// 4th-order tensor for isotropic elasticity: lives in physical space
    using ElasticityTensor = sofa::type::FullySymmetric4Tensor<spatial_dimensions, Real>;

    /// the type of B in e = B d, if e is the strain, and d is the displacement
    using StrainDisplacement = sofa::component::solidmechanics::fem::elastic::StrainDisplacement<SpatialTypes, ElementType>;

    /// the concatenation of the nodal values (displacement, temperature, ...) of the element nodes in a single vector
    using ElementDisplacement = sofa::type::Vec<NumberOfDofsInElement, Real>;

    /// the type of the element hessian matrix
    using ElementHessian = sofa::type::Mat<
        NumberOfDofsInElement,
        NumberOfDofsInElement,
        sofa::Real_t<DataTypes>>;

    using ElementGradient = sofa::type::Vec<NumberOfDofsInElement, sofa::Real_t<DataTypes>>;

    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real_t<DataTypes>>;

    using StressVoigtVector = sofa::type::Vec<sofa::type::NumberOfIndependentElements<spatial_dimensions>, sofa::Real_t<DataTypes>>;
};

}
