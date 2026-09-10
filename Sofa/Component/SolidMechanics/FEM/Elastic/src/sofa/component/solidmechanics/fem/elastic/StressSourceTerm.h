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
#include <sofa/type/MatSym.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_STRESS_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/// The symmetric stress tensor prescribed at one node.
template <class DataTypes>
using StressTensor =
    sofa::type::MatSym<DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;

/**
 * @class NodalStress
 * @brief A symmetric stress tensor prescribed at the nodes, one tensor per node.
 *
 * A tensor is read from a scene as a full matrix, row by row: nine components in 3D, four in 2D,
 * of which MatSym keeps the independent ones.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 */
template <class TDataTypes>
class NodalStress : public sofa::core::BaseNodalProperty<StressTensor<TDataTypes>>
{
public:
    using DataTypes = TDataTypes;
    using Tensor = StressTensor<DataTypes>;

    SOFA_CLASS(SOFA_TEMPLATE(NodalStress, DataTypes),
        SOFA_TEMPLATE(sofa::core::BaseNodalProperty, StressTensor<DataTypes>));

protected:

    NodalStress() : sofa::core::BaseNodalProperty<Tensor>(Tensor{}) {}
};

/**
 * @class StressSourceTerm
 * @brief A traction \f$ \sigma \, n \f$ built from a stress tensor prescribed at the nodes.
 *
 * The linked NodalStress is interpolated at the quadrature point and contracted with the unit
 * normal of the element, whose orientation follows the node ordering.
 *
 * The tensor must be symmetric, which excludes the first Piola-Kirchhoff stress.
 *
 * Only available on elements of codimension 1, the ones that have a normal.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Triangle).
 */
template <class TDataTypes, class TElementType>
class StressSourceTerm : public BaseSourceTerm<TDataTypes, TElementType>
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;

    SOFA_CLASS(SOFA_TEMPLATE2(StressSourceTerm, DataTypes, ElementType),
        SOFA_TEMPLATE2(BaseSourceTerm, DataTypes, ElementType));

    using Deriv = sofa::Deriv_t<DataTypes>;
    using QuadratureContext = QuadratureContext<DataTypes, ElementType>;
    using NodalStress = ::sofa::component::solidmechanics::fem::elastic::NodalStress<DataTypes>;

    /**
     * @brief Nodal values of the stress tensor this term integrates.
     */
    sofa::SingleLink<StressSourceTerm<DataTypes, ElementType>, NodalStress,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_stress;

    /**
     * @brief Initializes the component and checks that a stress is linked.
     */
    void init() override;

    /**
     * @brief The linked stress interpolated at the quadrature point, contracted with the normal.
     */
    Deriv evaluate(const QuadratureContext& context) const override;

protected:

    StressSourceTerm();
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_STRESS_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalStress<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalStress<sofa::defaulttype::Vec3Types>;

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API StressSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API StressSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API StressSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
