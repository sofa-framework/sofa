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
#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/core/trait/DataTypes.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_BASE_GEOMETRIC_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class BaseGeometricSourceTerm
 * @brief A source density whose value is determined by the geometry, not by the solution.
 *
 * The component calculates the integrand in evaluate() given a QuadratureContext.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template <class TDataTypes, class TElementType>
class BaseGeometricSourceTerm : public sofa::core::objectmodel::BaseComponent
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;

    SOFA_CLASS(SOFA_TEMPLATE2(BaseGeometricSourceTerm, DataTypes, ElementType),
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

    BaseGeometricSourceTerm() = default;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_BASE_GEOMETRIC_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseGeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
