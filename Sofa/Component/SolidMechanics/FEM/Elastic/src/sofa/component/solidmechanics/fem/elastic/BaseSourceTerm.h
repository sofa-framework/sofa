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

#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/core/BaseNodalProperty.h>
#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/core/trait/DataTypes.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/helper/accessor.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <array>
#include <limits>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_BASE_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @struct QuadratureContext
 * @brief Geometry of one quadrature point.
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

    /// Node indices of the element.
    const Element& element;

    /// Shape function values.
    ShapeFunctions N;

    /// Reference-space gradients of the shape functions.
    GradientShapeFunctions gradientShapeFunctions;

    /// dx/dq of the reference-to-physical mapping.
    Jacobian jacobian;

    /// \f$ |\det J| \f$
    Real measure;

    /// Interpolated rest position.
    Coord restPosition;

    /// Interpolated displacement.
    Deriv displacement;
};

/**
 * @brief Unit normal of an element of codimension 1, oriented by its node ordering.
 *
 * @param jacobian dx/dq of the reference-to-physical mapping.
 */
template <sofa::Size spatial_dimensions, sofa::Size TopologicalDimension, class Real>
sofa::type::Vec<spatial_dimensions, Real> elementNormal(
    const sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>& jacobian)
{
    static_assert(TopologicalDimension + 1 == spatial_dimensions,
        "A normal is only defined for an element of codimension 1.");

    sofa::type::Vec<spatial_dimensions, Real> normal;
    if constexpr (spatial_dimensions == 3)
    {
        normal = jacobian.col(0).cross(jacobian.col(1));
    }
    else
    {
        const sofa::type::Vec<2, Real> tangent = jacobian.col(0);
        normal = sofa::type::Vec<2, Real>(tangent[1], -tangent[0]);
    }

    normal.normalize(sofa::type::Vec<spatial_dimensions, Real>{}, std::numeric_limits<Real>::min());
    return normal;
}

/**
 * @brief Derivative of the area normal of an element with respect to a node position:
 *
 * \f[ \frac{\partial (|\det J| \, n)}{\partial x}
 *     = \sum_i \frac{\partial (|\det J| \, n)}{\partial t_i} \frac{\partial N}{\partial q_i}, \f]
 *
 * with \f$ t_i \f$ the columns of J and N the shape function of the node.
 *
 * @param jacobian dx/dq of the reference-to-physical mapping.
 * @param gradientOfShapeFunction dN/dq of the node.
 */
template <sofa::Size spatial_dimensions, sofa::Size TopologicalDimension, class Real>
sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real> elementAreaNormalDerivative(
    const sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>& jacobian,
    const sofa::type::Vec<TopologicalDimension, Real>& gradientOfShapeFunction)
{
    static_assert(TopologicalDimension + 1 == spatial_dimensions,
        "An area normal is only defined for an element of codimension 1.");

    using Derivative = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    if constexpr (spatial_dimensions == 3)
    {
        const auto tangent0 = jacobian.col(0);
        const auto tangent1 = jacobian.col(1);

        const Derivative skewTangent0 {
            {Real{0}, -tangent0[2], tangent0[1]},
            {tangent0[2], Real{0}, -tangent0[0]},
            {-tangent0[1], tangent0[0], Real{0}}};

        const Derivative skewTangent1 {
            {Real{0}, -tangent1[2], tangent1[1]},
            {tangent1[2], Real{0}, -tangent1[0]},
            {-tangent1[1], tangent1[0], Real{0}}};

        return skewTangent0 * gradientOfShapeFunction[1]
             - skewTangent1 * gradientOfShapeFunction[0];
    }
    else
    {
        const Derivative rotation {{Real{0}, Real{1}}, {Real{-1}, Real{0}}};

        return rotation * gradientOfShapeFunction[0];
    }
}

/**
 * @class BaseSourceTerm
 * @brief A source term of the weak form.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template <class TDataTypes, class TElementType>
class BaseSourceTerm : public sofa::core::objectmodel::BaseComponent
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;

    SOFA_CLASS(SOFA_TEMPLATE2(BaseSourceTerm, DataTypes, ElementType),
        sofa::core::objectmodel::BaseComponent);

    using Real = sofa::Real_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using QuadratureContext_t = QuadratureContext<DataTypes, ElementType>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

    /// Derivative of a nodal force with respect to a node position.
    using SourceDerivative = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    /**
     * @brief Source density at one quadrature point, per unit physical measure.
     *
     * @param context Geometry of the quadrature point.
     */
    virtual Deriv evaluate(const QuadratureContext_t& context) const = 0;

    /**
     * @brief Data the integrand depends on; used for callbacks.
     */
    virtual sofa::type::vector<const sofa::core::objectmodel::BaseData*> integrandInputs() const
    {
        return {};
    }

    /**
     * @brief Derivative of the measure-weighted density with respect to a node position,
     * \f$ \partial (|\det J| \, r) / \partial x_b \f$, with r the source density.
     *
     * @param context Geometry of the quadrature point.
     * @param node Index of the node in the element.
     */
    virtual SourceDerivative evaluateStiffness(const QuadratureContext_t& context,
        sofa::Size node) const = 0;

protected:

    BaseSourceTerm() = default;

    /**
     * @brief Nodal property interpolated at the quadrature point.
     *
     * @param property The nodal values.
     * @param context Geometry of the quadrature point.
     */
    template <class PropertyType>
    static PropertyType interpolateProperty(
        const sofa::core::BaseNodalProperty<PropertyType>& property,
        const QuadratureContext_t& context)
    {
        static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;

        sofa::helper::ReadAccessor<sofa::Data<sofa::type::vector<PropertyType>>> propertyAccessor {
            property.d_property};

        std::array<PropertyType, NumberOfNodesInElement> elementNodesProperty;
        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
        {
            elementNodesProperty[i] = property.getNodeProperty(context.element[i], propertyAccessor);
        }

        return QuadratureContext_t::FiniteElement::Helper::evaluateValueInElement(
            elementNodesProperty, context.N);
    }
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_BASE_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
