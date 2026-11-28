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

#include <sofa/component/mass/config.h>
#include <sofa/component/mass/BaseFEMMassMatrix.h>
#include <sofa/component/mass/NodalMassDensity.h>

#if !defined(SOFA_COMPONENT_MASS_FEMMASS_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::mass
{

/**
 * @class FEMMass
 * @brief Computes and stores the mass matrix for a finite element model (solid mechanics).
 *
 * The mass matrix is computed from a nodal mass density (see NodalMassDensity), interpolated with
 * the element shape functions:
 * \f$ M_{ij} = \int_{\Omega} \rho(\mathbf{x}) N_i(\mathbf{x}) N_j(\mathbf{x}) d\Omega \f$
 *
 * The degrees of freedom are the positions of the mesh nodes, and the gravity is applied as a body
 * force (f = M g).
 *
 * For other fields (e.g. temperature in a diffusion problem), use FEMCapacity instead.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template<class TDataTypes, class TElementType>
class FEMMass : public BaseFEMMassMatrix<TDataTypes, TDataTypes, TElementType>
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    SOFA_CLASS(SOFA_TEMPLATE2(FEMMass, DataTypes, ElementType),
        SOFA_TEMPLATE3(BaseFEMMassMatrix, DataTypes, DataTypes, ElementType));

protected:
    using Real = sofa::Real_t<DataTypes>;
    using NodalMassDensity = ::sofa::component::mass::NodalMassDensity<Real>;
    using typename Inherit1::NodalCoefficient;
    using typename Inherit1::GlobalMatrixType;

public:

    /**
     * @brief Link to the nodal mass density component.
     *
     * This component provides the mass density at each node of the mesh.
     * If not set, the first one found in the current context is used.
     */
    sofa::SingleLink<FEMMass, NodalMassDensity,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nodalMassDensity;

    /**
     * @brief Adds the gravity force (f = M * g) to the force vector.
     */
    void addForce(const core::MechanicalParams* mparams,
                  sofa::DataVecDeriv_t<DataTypes>& f,
                  const sofa::DataVecCoord_t<DataTypes>& x,
                  const sofa::DataVecDeriv_t<DataTypes>& v) override;

    using Inherit1::getPotentialEnergy;
    /**
     * @brief Computes the total gravitational potential energy of the system.
     *
     * \f$ V = - \mathbf{x}^T \mathbf{M} \mathbf{g} \f$
     */
    SReal getPotentialEnergy(
        const core::MechanicalParams* mparams,
        const DataVecCoord_t<DataTypes>& x) const override;

protected:

    FEMMass();

    const NodalCoefficient* getNodalCoefficient() const override;
    void validateNodalCoefficient() override;
};

#if !defined(SOFA_COMPONENT_MASS_FEMMASS_CPP)
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
#endif

}  // namespace sofa::component::mass
