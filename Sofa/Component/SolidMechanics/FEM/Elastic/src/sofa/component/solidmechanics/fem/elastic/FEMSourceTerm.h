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
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_FEM_SOURCE_TERM_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class FEMSourceTerm
 * @brief Computes nodal source terms by integrating a given source density field stored at the nodes.
 *
 * This class assembles and stores a geometric matrix M using a quadrature rule over the element domain. 
 * The matrix is then used to multiply the source density b to compute a nodal source term F that 
 * contributes to the RHS of the weak form through the addForce function.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template <class TDataTypes, class TElementType>
class FEMSourceTerm :
    public sofa::core::behavior::ForceField<TDataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    SOFA_CLASS2(SOFA_TEMPLATE2(FEMSourceTerm, DataTypes, ElementType),
        sofa::core::behavior::ForceField<DataTypes>,
        sofa::core::behavior::TopologyAccessor);

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;

    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;

    using GlobalMatrix = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<sofa::Real_t<DataTypes>>;

public:

    /**
     * @brief Initializes the component.
     *
     * This method performs several initialization steps:
     * 1. Initializes the base force field.
     * 2. Initializes the topology accessor.
     * 3. Resizes the nodal source density.
     * 4. Assembles the global matrix M.
     */
    void init() override;

    /**
     * @brief Adds the force (f = M * b) to the RHS vector.
     *
     * This method computes the product of the geometric matrix and the source density vector,
     * adding the result to the force vector `f`.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param f The force vector to which the source term will be added.
     * @param x The current positions (unused in this implementation).
     * @param v The current velocities (unused in this implementation).
     */
    void addForce(
        const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& f,
        const sofa::DataVecCoord_t<DataTypes>& x,
        const sofa::DataVecDeriv_t<DataTypes>& v) override;

    /**
     * @brief A no-op as the load is prescribed on the rest configuration
     */ 
    void addDForce(const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& df,
        const sofa::DataVecDeriv_t<DataTypes>& dx) override;

    /**
     * @brief A no-op as the load is prescribed on the rest configuration
     */ 
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    using sofa::core::behavior::ForceField<DataTypes>::getPotentialEnergy;
    /**
     * @brief Not implemented, returns 0.
     */
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
        const sofa::DataVecCoord_t<DataTypes>& x) const override;

    /**
     * @brief Source term (per unit volume) sampled at each node.
     */
    sofa::Data<sofa::VecDeriv_t<DataTypes> > d_nodalSourceDensity;

protected:

    /**
     * @brief Default constructor.
     */
    FEMSourceTerm();

    /**
     * @brief Resizes the source density to the size of the mechanical state
     */
    void resizeNodalSourceDensity(const std::size_t size);

    /**
     * @brief Assembles and stores the geometry-only matrix \f$ M_{ij} = \int_{\Omega} N_i N_j \, d\Omega \f$ over each element on the rest configuration.
     */
    void assembleGlobalMatrix();

    /**
     * @brief Geometry-only matrix \f$ M_{ij} = \int_{\Omega} N_i N_j \, d\Omega \f$ of the system.
     *
     * Stored in compressed sparse row format. Assembled once in init on the rest configuration.
     */
    GlobalMatrix m_globalMatrix;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_FEM_SOURCE_TERM_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
