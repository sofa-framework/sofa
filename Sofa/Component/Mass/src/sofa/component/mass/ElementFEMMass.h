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

#include <sofa/component/mass/NodalMassDensity.h>
#include <sofa/component/mass/config.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>

#if !defined(SOFA_COMPONENT_MASS_ELEMENTFEMMASS_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::mass
{

/**
 * @class ElementFEMMass
 * @brief Computes and stores the mass matrix for a finite element model.
 *
 * This class calculates the mass matrix for a given set of finite elements based on a nodal mass density field.
 * The mass matrix is computed using numerical integration (quadrature) over the element domain.
 * The mass density is interpolated from the nodes using the element's shape functions.
 *
 * Mathematically, the mass matrix M is defined as:
 * \f$ M_{ij} = \int_{\Omega} \rho(\mathbf{x}) N_i(\mathbf{x}) N_j(\mathbf{x}) d\Omega \f$
 * where \f$ \rho(\mathbf{x}) \f$ is the mass density and \f$ N_i(\mathbf{x}) \f$ are the shape functions.
 *
 * The resulting matrix is a block-diagonal-like matrix where each block is a scalar multiple of the identity,
 * representing the contribution to each degree of freedom. This allows for efficient storage and computation.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template<class TDataTypes, class TElementType>
class ElementFEMMass :
    public core::behavior::Mass<TDataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    SOFA_CLASS2(SOFA_TEMPLATE2(ElementFEMMass, DataTypes, ElementType),
        core::behavior::Mass<TDataTypes>,
        sofa::core::behavior::TopologyAccessor);

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size NumberOfDofsInElement = NumberOfNodesInElement * spatial_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    using ElementMassMatrix = sofa::type::Mat<NumberOfNodesInElement, NumberOfNodesInElement, sofa::Real_t<DataTypes>>;

    using NodalMassDensity = ::sofa::component::mass::NodalMassDensity<sofa::Real_t<DataTypes>>;
    using GlobalMassMatrixType = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<Real_t<DataTypes>>;

public:

    /**
     * @brief Gets the class name according to the provided template parameters.
     *
     * For example, `ElementFEMMass<Vec3Types, sofa::geometry::Edge>` will return "EdgeFEMMass".
     *
     * @return A string representing the class name.
     */
    static const std::string GetCustomClassName()
    {
        return std::string(sofa::geometry::elementTypeToString(ElementType::Element_type)) + "FEMMass";
    }

    /**
     * @brief Gets the template name based on the data types.
     * @return A string representing the template name (e.g., "Vec3d").
     */
    static const std::string GetCustomTemplateName() { return DataTypes::Name(); }

    /**
     * @brief Link to the nodal mass density component.
     *
     * This component provides the mass density at each node of the mesh.
     * It must be present in the context for the mass to be calculated correctly.
     */
    sofa::SingleLink<ElementFEMMass, NodalMassDensity,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nodalMassDensity;

    /**
     * @brief Initializes the component.
     *
     * This method performs several initialization steps:
     * 1. Initializes the topology accessor.
     * 2. Initializes the base Mass class.
     * 3. Validates that a valid nodal mass density is linked.
     * 4. Computes and assembles the global mass matrix.
     */
    void init() final;

    /**
     * @brief Indicates whether the mass matrix is diagonal.
     * @return Always returns false, as FEM mass matrices are generally not diagonal (unless lumped).
     */
    bool isDiagonal() const override { return false; }

    /**
     * @brief Adds the gravity force (f = M * g) to the force vector.
     *
     * This method computes the product of the mass matrix and the gravity vector,
     * adding the result to the force vector `f`.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param f The force vector to which the gravity force will be added.
     * @param x The current positions (unused in this implementation).
     * @param v The current velocities (unused in this implementation).
     */
    void addForce(const core::MechanicalParams* mparams,
                  sofa::DataVecDeriv_t<DataTypes>& f,
                  const sofa::DataVecCoord_t<DataTypes>& x,
                  const sofa::DataVecDeriv_t<DataTypes>& v) override;

    /**
     * @brief Accumulates the mass matrix into a global matrix accumulator.
     *
     * This is used during the assembly of the system matrix for implicit solvers.
     * It maps the stored sparse mass matrix into the global system matrix.
     *
     * @param matrices The accumulator used to build the global system matrix.
     */
    void buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices) override;

    using Inherit1::addMDx;
    /**
     * @brief Adds the product of the mass matrix and a vector to another vector (f += M * dx * factor).
     *
     * This is used for computing accelerations or in iterative solvers.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param f The result vector to which the product is added.
     * @param dx The vector to be multiplied by the mass matrix.
     * @param factor A scaling factor for the product.
     */
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv_t<DataTypes>& f, const DataVecDeriv_t<DataTypes>& dx, SReal factor) override;

    using Inherit1::accFromF;
    /**
     * @brief Supposed to compute $ a = M^{-1} f $, but triggers an error in this implementation.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param a The result vector of $M^{-1} f$.
     * @param f The vector to be multiplied by the inverse mass matrix.
     */
    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv_t<DataTypes>& a, const DataVecDeriv_t<DataTypes>& f) override;


    using Inherit1::getKineticEnergy;
    SReal getKineticEnergy(const core::MechanicalParams* mparams,
                           const DataVecDeriv_t<DataTypes>& v) const override;

    using Inherit1::getPotentialEnergy;
    SReal getPotentialEnergy(
        const core::MechanicalParams* mparams,
        const core::behavior::Mass<TDataTypes>::DataVecCoord& x) const override;

protected:

    /**
     * @brief Default constructor.
     */
    ElementFEMMass();

    /**
     * @brief Performs the internal calculation and assembly of the mass matrix.
     *
     * This method iterates over all elements, performs numerical integration to find
     * element-level mass matrices, and then assembles them into the global sparse matrix `m_globalMassMatrix`.
     */
    void elementFEMMass_init();

    /**
     * @brief Computes the mass matrix for each finite element.
     *
     * This method iterates over all elements provided by the topology and performs
     * numerical integration (quadrature) to compute the local mass matrix for each element.
     * The mass density is interpolated from the nodal values using the element's shape functions.
     *
     * @param elements The sequence of elements to process.
     * @param[out] elementMassMatrices A vector to be populated with the computed local mass matrices.
     */
    void calculateElementMassMatrix(const auto& elements, sofa::type::vector<ElementMassMatrix> &elementMassMatrices);

    /**
     * @brief Assembles the global mass matrix from individual element mass matrices.
     *
     * This method takes the precomputed local mass matrices and distributes their contributions
     * into the global sparse mass matrix `m_globalMassMatrix`. It handles the mapping from
     * local element node indices to global degree-of-freedom indices.
     *
     * @param elements The sequence of elements corresponding to the provided mass matrices.
     * @param elementMassMatrices The precomputed local mass matrices for each element.
     */
    void initializeGlobalMassMatrix(const auto& elements, const sofa::type::vector<ElementMassMatrix>& elementMassMatrices);

    /**
     * @brief Ensures that a valid NodalMassDensity component is linked.
     *
     * If no link is set, it attempts to find a suitable component in the current context.
     * If no component is found, it marks the component state as invalid.
     */
    void validateNodalMassDensity();

    /**
     * @brief Represents the global mass matrix of the system.
     *
     * This matrix is stored in a compressed sparse row format.
     * Since the mass contribution for each node is isotropic (affects all spatial dimensions
     * identically), this matrix stores only the scalar scaling factor for each identity block
     * of size `spatial_dimensions * spatial_dimensions`.
     *
     * For example, in 3D, a node's mass contribution is a 3x3 diagonal matrix `m * I`.
     * Only `m` is stored here.
     */
    GlobalMassMatrixType m_globalMassMatrix;
};

#if !defined(SOFA_COMPONENT_MASS_ELEMENTFEMMASS_CPP)
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::mass
