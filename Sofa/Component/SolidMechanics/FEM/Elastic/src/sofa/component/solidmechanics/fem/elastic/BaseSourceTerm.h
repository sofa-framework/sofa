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
#include <sofa/component/solidmechanics/fem/elastic/QuadratureContext.h>
#include <sofa/core/BaseNodalProperty.h>
#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/core/trait/DataTypes.h>
#include <sofa/helper/accessor.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <array>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_BASE_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @brief Unit normal of a codimension-1 element, from the jacobian of its mapping.
 *
 * Defined only where the element spans one dimension less than the space it lives in: a surface
 * element in 3D, an edge in 2D. Its orientation follows the node ordering of the element.
 *
 * @param jacobian dx/dq of the reference-to-physical mapping at the point of interest.
 */
template <sofa::Size spatial_dimensions, sofa::Size TopologicalDimension, class Real>
sofa::type::Vec<spatial_dimensions, Real> elementNormal(
    const sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>& jacobian)
{
    static_assert(TopologicalDimension + 1 == spatial_dimensions,
        "A normal is only defined for an element of codimension 1.");

    if constexpr (spatial_dimensions == 3)
    {
        return jacobian.col(0).cross(jacobian.col(1)).normalized();
    }
    else
    {
        const sofa::type::Vec<2, Real> tangent = jacobian.col(0);
        return sofa::type::Vec<2, Real>(tangent[1], -tangent[0]).normalized();
    }
}

/**
 * @class BaseSourceTerm
 * @brief A source density whose value is determined by the geometry, not by the solution.
 *
 * The component calculates the integrand in evaluate() given a QuadratureContext.
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

    using Deriv = sofa::Deriv_t<DataTypes>;
    using QuadratureContext = QuadratureContext<DataTypes, ElementType>;

    /**
     * @brief Source density at one quadrature point, per unit physical measure.
     *
     * @param context Geometry of the quadrature point.
     */
    virtual Deriv evaluate(const QuadratureContext& context) const = 0;

protected:

    BaseSourceTerm() = default;

    /**
     * @brief Value of a nodal property interpolated at the quadrature point.
     *
     * The property is gathered at the nodes of the element being integrated and combined with the
     * shape functions evaluated at that point.
     *
     * @param property The component holding the nodal values.
     * @param context Geometry of the quadrature point.
     */
    template <class PropertyType>
    static PropertyType interpolateProperty(
        const sofa::core::BaseNodalProperty<PropertyType>& property,
        const QuadratureContext& context)
    {
        static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;

        sofa::helper::ReadAccessor<sofa::Data<sofa::type::vector<PropertyType>>> propertyAccessor {
            property.d_property};

        std::array<PropertyType, NumberOfNodesInElement> elementNodesProperty;
        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
        {
            elementNodesProperty[i] = property.getNodeProperty(context.element[i], propertyAccessor);
        }

        return QuadratureContext::FiniteElement::Helper::evaluateValueInElement(
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
