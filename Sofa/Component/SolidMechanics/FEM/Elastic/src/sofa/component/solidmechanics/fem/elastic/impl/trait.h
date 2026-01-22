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

#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/ElementStiffnessMatrix.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/FullySymmetric4Tensor.h>
#include <sofa/core/trait/DataTypes.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
struct trait
{
    using DataVecCoord = sofa::DataVecDeriv_t<DataTypes>;
    using DataVecDeriv = sofa::DataVecDeriv_t<DataTypes>;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using Real = sofa::Real_t<DataTypes>;

    using FiniteElement = sofa::component::solidmechanics::fem::elastic::FiniteElement<ElementType, DataTypes>;
    using TopologyElement = typename FiniteElement::TopologyElement;
    using ReferenceCoord = typename FiniteElement::ReferenceCoord;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size NumberOfDofsInElement = NumberOfNodesInElement * spatial_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;
    static constexpr sofa::Size NbQuadraturePoints = FiniteElement::quadraturePoints().size();

    /// type of 2nd-order tensor for the elasticity tensor for isotropic materials
    using ElasticityTensor = FullySymmetric4Tensor<DataTypes>;

    /// the type of B in e = B d, if e is the strain, and d is the displacement
    using StrainDisplacement = sofa::component::solidmechanics::fem::elastic::StrainDisplacement<DataTypes, ElementType>;

    /// the concatenation of the displacement of the element nodes in a single vector
    using ElementDisplacement = sofa::type::Vec<NumberOfDofsInElement, Real>;

    /// tells how to compute the matrix-vector product of the stiffness matrix with a displacement
    /// vector. It does not change the result, but it can have an impact on performances.
    static constexpr MatrixVectorProductType matrixVectorProductType =
        NbQuadraturePoints > 1
            ? MatrixVectorProductType::Dense
            : MatrixVectorProductType::Factorization;

    /// the type of the element stiffness matrix
    using ElementStiffness = sofa::component::solidmechanics::fem::elastic::FactorizedElementStiffness<DataTypes, ElementType, matrixVectorProductType>;

    using ElementForce = sofa::type::Vec<trait::NumberOfDofsInElement, sofa::Real_t<DataTypes>>;
};

}
