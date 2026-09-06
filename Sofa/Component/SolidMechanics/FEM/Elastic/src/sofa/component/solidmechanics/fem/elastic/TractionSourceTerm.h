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
#include <sofa/component/solidmechanics/fem/elastic/NonConstantSourceTerm.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_TRACTION_SOURCE_TERM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/geometry/Quad.h>
#include <sofa/geometry/Triangle.h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class TractionSourceTerm
 * @brief A pressure load, following the current-configuration normal direction.
 *
 * Restricted to Triangle and Quad, the same element types SurfacePressureForceField supports.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The boundary element type (Triangle or Quad).
 */
template <class TDataTypes, class TElementType>
class TractionSourceTerm : public NonConstantSourceTerm<TDataTypes, TElementType>
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;

    SOFA_CLASS(SOFA_TEMPLATE2(TractionSourceTerm, DataTypes, ElementType),
        SOFA_TEMPLATE2(NonConstantSourceTerm, DataTypes, ElementType));

    using Real = sofa::Real_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Jacobian = typename NonConstantSourceTerm<DataTypes, ElementType>::Jacobian;
    using SourceDerivative = typename NonConstantSourceTerm<DataTypes, ElementType>::SourceDerivative;
    static constexpr sofa::Size TopologicalDimension = NonConstantSourceTerm<DataTypes, ElementType>::TopologicalDimension;

    /**
     * @brief Pressure per unit area, following the current-configuration normal direction.
     */
    sofa::Data<Real> d_pressure;

    /**
     * @brief pressure * unit normal, jacobian.col(0) x jacobian.col(1) normalized — same tangent
     * convention as SurfacePressureForceField, generalized through the Jacobian so Triangle and
     * Quad share one formula.
     */
    Deriv evaluate(const Coord& restPosition, const Deriv& displacement, const Jacobian& jacobian) const override
    {
        SOFA_UNUSED(restPosition);
        SOFA_UNUSED(displacement);
        return jacobian.col(0).cross(jacobian.col(1)).normalized() * d_pressure.getValue();
    }

    /**
     * @brief d(pressure * jacobian.col(0) x jacobian.col(1))/d(node j position).
     */
    SourceDerivative evaluateStiffness(const Jacobian& jacobian,
        const sofa::type::Vec<TopologicalDimension, Real>& gradientOfShapeFunction) const override
    {
        const auto t0 = jacobian.col(0);
        const auto t1 = jacobian.col(1);

        SourceDerivative skewT0{};
        skewT0(0,1) = -t0[2]; skewT0(0,2) =  t0[1];
        skewT0(1,0) =  t0[2]; skewT0(1,2) = -t0[0];
        skewT0(2,0) = -t0[1]; skewT0(2,1) =  t0[0];

        SourceDerivative skewT1{};
        skewT1(0,1) = -t1[2]; skewT1(0,2) =  t1[1];
        skewT1(1,0) =  t1[2]; skewT1(1,2) = -t1[0];
        skewT1(2,0) = -t1[1]; skewT1(2,1) =  t1[0];

        return (skewT0 * gradientOfShapeFunction[1] - skewT1 * gradientOfShapeFunction[0]) * d_pressure.getValue();
    }

protected:

    TractionSourceTerm()
        : d_pressure(initData(&d_pressure, Real{0}, "pressure",
              "Pressure per unit area, following the current-configuration normal direction."))
    {}
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_TRACTION_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TractionSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TractionSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
