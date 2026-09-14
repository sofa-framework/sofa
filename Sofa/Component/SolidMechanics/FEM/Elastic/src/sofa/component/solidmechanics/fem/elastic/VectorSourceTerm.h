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
#include <sofa/component/solidmechanics/fem/elastic/BaseSourceTerm.h>
#include <sofa/core/BaseNodalProperty.h>
#include <sofa/core/objectmodel/Link.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_VECTOR_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class NodalSourceDensity
 * @brief A source density prescribed at the nodes, one vector per node.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 */
template <class TDataTypes>
class NodalSourceDensity : public sofa::core::BaseNodalProperty<sofa::Deriv_t<TDataTypes>>
{
public:
    using DataTypes = TDataTypes;
    using Deriv = sofa::Deriv_t<DataTypes>;

    SOFA_CLASS(SOFA_TEMPLATE(NodalSourceDensity, DataTypes),
        SOFA_TEMPLATE(sofa::core::BaseNodalProperty, sofa::Deriv_t<DataTypes>));

protected:

    NodalSourceDensity() : sofa::core::BaseNodalProperty<Deriv>(Deriv{}) {}
};

/**
 * @class VectorSourceTerm
 * @brief A source density given directly as a vector at each node.
 *
 * The linked NodalSourceDensity is interpolated at the quadrature point and returned unchanged.
 * The measure the integrator divides it by is the one of the element it is attached to: a body
 * force on a volume element, a traction on a boundary one.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template <class TDataTypes, class TElementType>
class VectorSourceTerm : public BaseSourceTerm<TDataTypes, TElementType>
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;

    SOFA_CLASS(SOFA_TEMPLATE2(VectorSourceTerm, DataTypes, ElementType),
        SOFA_TEMPLATE2(BaseSourceTerm, DataTypes, ElementType));

    using Deriv = sofa::Deriv_t<DataTypes>;
    using QuadratureContext_t = QuadratureContext<DataTypes, ElementType>;
    using NodalSourceDensity =
        ::sofa::component::solidmechanics::fem::elastic::NodalSourceDensity<DataTypes>;

    /**
     * @brief Nodal values of the source density this term integrates.
     */
    sofa::SingleLink<VectorSourceTerm<DataTypes, ElementType>, NodalSourceDensity,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_sourceDensity;

    /**
     * @brief Initializes the component and checks that a source density is linked.
     */
    void init() override;

    /**
     * @brief The linked source density interpolated at the quadrature point.
     */
    Deriv evaluate(const QuadratureContext_t& context) const override;

protected:

    VectorSourceTerm();
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_VECTOR_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec3Types>;

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
