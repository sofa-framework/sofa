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
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @struct QuadratureContext
 * @brief Everything the integrator knows at one quadrature point.
 *
 * Built once per quadrature point and handed to every integrated term. 
 * A source term reads from it and returns an integrand.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template <class TDataTypes, class TElementType>
struct QuadratureContext
{
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;

    using Real = sofa::Real_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;

    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    using Element = typename FiniteElement::TopologyElement;
    using ShapeFunctions = sofa::type::Vec<NumberOfNodesInElement, Real>;
    using GradientShapeFunctions = sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>;
    using Jacobian = sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>;

    /// Node indices of the element being integrated, with which a term gathers its own nodal
    /// degrees of freedom.
    const Element& element;

    /// Shape function value at this quadrature point.
    ShapeFunctions N;

    /// Reference-space gradients of the shape functions at this quadrature point.
    GradientShapeFunctions gradientShapeFunctions;

    /// dx/dq of the reference-to-physical mapping, on the configuration the integrator chose.
    Jacobian jacobian;

    /// \f$ |\det J| \f$, for information only: the integrator applies it, a term must not.
    Real measure;

    /// Interpolated rest position at this quadrature point.
    Coord restPosition;

    /// Interpolated displacement at this quadrature point.
    Deriv displacement;
};

}  // namespace sofa::component::solidmechanics::fem::elastic
