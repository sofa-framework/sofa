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
#include <sofa/core/trait/DataTypes.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/type/Mat.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NON_CONSTANT_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#include <sofa/geometry/Edge.h>
#include <sofa/geometry/Hexahedron.h>
#include <sofa/geometry/Quad.h>
#include <sofa/geometry/Tetrahedron.h>
#include <sofa/geometry/Triangle.h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class NonConstantSourceTerm
 * @brief A source density prescribed at the nodes, depending on the current displacement.
 *
 * evaluate() defaults to zero. This class contributes nothing until subclassed. Link it (or a
 * subclass) to a FEMSourceTermIntegrator through l_nonConstantSources.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 */
template <class TDataTypes, class TElementType>
class NonConstantSourceTerm : public sofa::core::BaseNodalProperty<sofa::Deriv_t<TDataTypes>>
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;

    SOFA_CLASS(SOFA_TEMPLATE2(NonConstantSourceTerm, DataTypes, ElementType),
        SOFA_TEMPLATE(sofa::core::BaseNodalProperty, sofa::Deriv_t<DataTypes>));

    using Real = sofa::Real_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    /// Jacobian of the reference-to-physical mapping, evaluated where evaluate() is called.
    using Jacobian = sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>;

    /// d(nodal force)/d(node j position), for one test node, at one integration point.
    using SourceDerivative = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    /**
     * @brief Source density at one integration point. Defaults to zero.
     */
    virtual Deriv evaluate(const Coord& restPosition, const Deriv& displacement, const Jacobian& jacobian) const
    {
        SOFA_UNUSED(restPosition);
        SOFA_UNUSED(displacement);
        SOFA_UNUSED(jacobian);
        return Deriv{};
    }

    /**
     * @brief Stiffness contribution of node j at one integration point
     * Defaults to zero.
     */
    virtual SourceDerivative evaluateStiffness(const Jacobian& jacobian,
        const sofa::type::Vec<TopologicalDimension, Real>& gradientOfShapeFunction) const
    {
        SOFA_UNUSED(jacobian);
        SOFA_UNUSED(gradientOfShapeFunction);
        return SourceDerivative{};
    }

protected:

    NonConstantSourceTerm() : sofa::core::BaseNodalProperty<Deriv>(Deriv{}) {}
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NON_CONSTANT_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
