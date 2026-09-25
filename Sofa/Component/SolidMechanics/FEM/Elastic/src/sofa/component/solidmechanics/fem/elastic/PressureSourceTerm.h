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

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_PRESSURE_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class NodalPressure
 * @brief A pressure prescribed at the nodes, one scalar per node.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 */
template <class TDataTypes>
class NodalPressure : public sofa::core::BaseNodalProperty<sofa::Real_t<TDataTypes>>
{
public:
    using DataTypes = TDataTypes;
    using Real = sofa::Real_t<DataTypes>;

    SOFA_CLASS(SOFA_TEMPLATE(NodalPressure, DataTypes),
        SOFA_TEMPLATE(sofa::core::BaseNodalProperty, sofa::Real_t<DataTypes>));

protected:

    NodalPressure() : sofa::core::BaseNodalProperty<Real>(Real{}) {}
};

/**
 * @class PressureSourceTerm
 * @brief A traction \f$ p \, n \f$ built from a pressure prescribed at the nodes.
 *
 * The linked NodalPressure is interpolated at the quadrature point and multiplied by the unit
 * normal of the element. A positive pressure acts along the normal, whose orientation follows the
 * node ordering of the element.
 *
 * Only available on elements of codimension 1, the ones that have a normal.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Triangle).
 */
template <class TDataTypes, class TElementType>
class PressureSourceTerm : public BaseSourceTerm<TDataTypes, TElementType>
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;

    SOFA_CLASS(SOFA_TEMPLATE2(PressureSourceTerm, DataTypes, ElementType),
        SOFA_TEMPLATE2(BaseSourceTerm, DataTypes, ElementType));

    using Deriv = sofa::Deriv_t<DataTypes>;
    using QuadratureContext_t = QuadratureContext<DataTypes, ElementType>;
    using NodalPressure = ::sofa::component::solidmechanics::fem::elastic::NodalPressure<DataTypes>;

    /**
     * @brief Nodal values of the pressure this term integrates.
     */
    sofa::SingleLink<PressureSourceTerm<DataTypes, ElementType>, NodalPressure,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_pressure;

    /**
     * @brief Initializes the component and checks that a pressure is linked.
     */
    void init() override;

    /**
     * @brief The linked pressure interpolated at the quadrature point, times the unit normal.
     */
    Deriv evaluate(const QuadratureContext_t& context) const override;

    /**
     * @brief The nodal values this term interpolates, for FEMSourceTermIntegrator to track.
     */
    sofa::type::vector<const sofa::core::objectmodel::BaseData*> integrandInputs() const override;

protected:

    PressureSourceTerm();
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_PRESSURE_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalPressure<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalPressure<sofa::defaulttype::Vec3Types>;

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API PressureSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API PressureSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API PressureSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
