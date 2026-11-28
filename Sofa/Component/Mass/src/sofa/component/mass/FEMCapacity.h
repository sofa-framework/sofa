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
#include <sofa/component/mass/NodalCapacity.h>

#if !defined(SOFA_COMPONENT_MASS_FEMCAPACITY_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::mass
{

/**
 * @class FEMCapacity
 * @brief Computes and stores the capacity matrix of a field defined on a finite element mesh.
 *
 * In a diffusion problem \f$ \mathbf{C} \dot{\mathbf{u}} + \mathbf{K} \mathbf{u} = \mathbf{f} \f$
 * (e.g. heat transfer), the capacity matrix C is computed from a nodal capacity (see NodalCapacity),
 * interpolated with the element shape functions:
 * \f$ C_{ij} = \int_{\Omega} c(\mathbf{x}) N_i(\mathbf{x}) N_j(\mathbf{x}) d\Omega \f$
 *
 * The field (DataTypes, e.g. a scalar temperature) is defined on a mesh whose coordinates
 * (SpatialTypes) are read from the rest positions of the state linked by `spatialState`.
 * Unlike FEMMass, no body force (gravity) is applied.
 *
 * In a scene, the template is "<field type>,<spatial type>,<element type>", for example
 * "Vec1,Vec3,Tetrahedron" for a scalar field on a tetrahedral mesh in 3D.
 *
 * @tparam TDataTypes The type of the field (e.g., Vec1Types).
 * @tparam TSpatialTypes The type of the spatial coordinates of the mesh (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template<class TDataTypes, class TSpatialTypes, class TElementType>
class FEMCapacity : public BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>
{
public:
    using DataTypes = TDataTypes;
    using SpatialTypes = TSpatialTypes;
    using ElementType = TElementType;
    SOFA_CLASS(SOFA_TEMPLATE3(FEMCapacity, DataTypes, SpatialTypes, ElementType),
        SOFA_TEMPLATE3(BaseFEMMassMatrix, DataTypes, SpatialTypes, ElementType));

protected:
    using Real = sofa::Real_t<DataTypes>;
    using NodalCapacity = ::sofa::component::mass::NodalCapacity<Real>;
    using typename Inherit1::NodalCoefficient;

public:

    /**
     * @brief Link to the nodal capacity component.
     *
     * This component provides the capacity at each node of the mesh (e.g. the volumetric heat
     * capacity in heat transfer). If not set, the first one found in the current context is used.
     */
    sofa::SingleLink<FEMCapacity, NodalCapacity,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nodalCapacity;

    /// No body force is applied on a field which is not a position
    void addForce(const core::MechanicalParams* mparams,
                  sofa::DataVecDeriv_t<DataTypes>& f,
                  const sofa::DataVecCoord_t<DataTypes>& x,
                  const sofa::DataVecDeriv_t<DataTypes>& v) override;

    using Inherit1::getPotentialEnergy;
    /// No body force is applied, hence no potential energy
    SReal getPotentialEnergy(
        const core::MechanicalParams* mparams,
        const DataVecCoord_t<DataTypes>& x) const override;

protected:

    FEMCapacity();

    const NodalCoefficient* getNodalCoefficient() const override;
    void validateNodalCoefficient() override;
};

#if !defined(SOFA_COMPONENT_MASS_FEMCAPACITY_CPP)
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;

extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMCapacity<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
#endif

}  // namespace sofa::component::mass
