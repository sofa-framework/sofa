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
#include <sofa/component/solidmechanics/fem/elastic/BaseGeometricSourceTerm.h>
#include <sofa/core/BaseNodalProperty.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/fem/FiniteElement.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_GEOMETRIC_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class GeometricSourceTerm
 * @brief A source term built from a nodal property.
 *
 * The property is provided by a linked BaseNodalProperty component. A derived class interpolates
 * it with interpolateProperty() and turns it into a source density in evaluate().
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 * @tparam TPropertyType The type of the nodal property (e.g., Deriv).
 */
template <class TDataTypes, class TElementType, class TPropertyType>
class GeometricSourceTerm : public BaseGeometricSourceTerm<TDataTypes, TElementType>
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    using PropertyType = TPropertyType;

    SOFA_CLASS(SOFA_TEMPLATE3(GeometricSourceTerm, DataTypes, ElementType, PropertyType),
        SOFA_TEMPLATE2(BaseGeometricSourceTerm, DataTypes, ElementType));

    using Deriv = sofa::Deriv_t<DataTypes>;
    using QuadratureContext = QuadratureContext<DataTypes, ElementType>;
    using NodalProperty = sofa::core::BaseNodalProperty<PropertyType>;

    /**
     * @brief Nodal values of the property this term is built from.
     */
    sofa::SingleLink<GeometricSourceTerm<DataTypes, ElementType, PropertyType>, NodalProperty,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nodalProperty;

    /**
     * @brief Initializes the component and validates the linked nodal property.
     */
    void init() override;

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;

    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;

    GeometricSourceTerm();

    /**
     * @brief Value of the linked nodal property interpolated at the quadrature point.
     */
    PropertyType interpolateProperty(const QuadratureContext& context) const;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_GEOMETRIC_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge, sofa::Deriv_t<sofa::defaulttype::Vec1Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge, sofa::Deriv_t<sofa::defaulttype::Vec2Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge, sofa::Deriv_t<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle, sofa::Deriv_t<sofa::defaulttype::Vec2Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle, sofa::Deriv_t<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad, sofa::Deriv_t<sofa::defaulttype::Vec2Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad, sofa::Deriv_t<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron, sofa::Deriv_t<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API GeometricSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron, sofa::Deriv_t<sofa::defaulttype::Vec3Types>>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
