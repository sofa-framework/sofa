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
#include <sofa/core/BaseNodalProperty.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>

#if !defined(SOFA_COMPONENT_MASS_BASEFEMMASSMATRIX_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::mass
{

/**
 * @class BaseFEMMassMatrix
 * @brief Computes and stores the finite element matrix associated with the time derivative of a field.
 *
 * This matrix is computed by numerical integration (quadrature) over the elements, from a nodal
 * coefficient interpolated with the element shape functions:
 * \f$ C_{ij} = \int_{\Omega} c(\mathbf{x}) N_i(\mathbf{x}) N_j(\mathbf{x}) d\Omega \f$
 * where \f$ c(\mathbf{x}) \f$ is the coefficient and \f$ N_i(\mathbf{x}) \f$ are the shape functions.
 *
 * In the finite element literature, this matrix is generically called the mass matrix, whatever
 * the physics. Derived components give it a physical meaning:
 * - in solid mechanics, c is the mass density and C is the mass matrix (see FEMMass);
 * - in diffusion problems, c is a capacity (e.g. volumetric heat capacity) and C is the capacity
 *   matrix (see FEMCapacity).
 *
 * Since the contribution of each node is isotropic (identical on all the field components), only a
 * scalar is stored for each pair of nodes.
 *
 * The quantity carried by the degrees of freedom (DataTypes) can differ from the type of the
 * coordinates of the mesh (SpatialTypes). For example, a scalar field (Vec1Types) can be defined on
 * a volumetric mesh (Vec3Types). The geometry of the elements is read from the rest positions of
 * the state linked by `spatialState`.
 *
 * @tparam TDataTypes The type of the quantity carried by the degrees of freedom (e.g., Vec1Types).
 * @tparam TSpatialTypes The type of the spatial coordinates of the mesh (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template<class TDataTypes, class TSpatialTypes, class TElementType>
class BaseFEMMassMatrix :
    public core::behavior::Mass<TDataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor
{
public:
    using DataTypes = TDataTypes;
    using SpatialTypes = TSpatialTypes;
    using ElementType = TElementType;
    SOFA_ABSTRACT_CLASS2(SOFA_TEMPLATE3(BaseFEMMassMatrix, DataTypes, SpatialTypes, ElementType),
        core::behavior::Mass<TDataTypes>,
        sofa::core::behavior::TopologyAccessor);

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, SpatialTypes>;

    using Real = sofa::Real_t<DataTypes>;
    using GeometryReal = sofa::Real_t<SpatialTypes>;
    using GeometryCoord = sofa::Coord_t<SpatialTypes>;

    /// Dimension of the space in which the mesh lives
    static constexpr sofa::Size spatial_dimensions = SpatialTypes::spatial_dimensions;
    /// Number of components of the field at each node (1 for a scalar field)
    static constexpr sofa::Size field_dimensions = DataTypes::deriv_total_size;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size NumberOfDofsInElement = NumberOfNodesInElement * field_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    static_assert(TopologicalDimension <= spatial_dimensions,
                  "The element cannot have a higher dimension than the space it is embedded in");

    using ElementMatrix = sofa::type::Mat<NumberOfNodesInElement, NumberOfNodesInElement, Real>;
    using NodalCoefficient = sofa::core::BaseNodalProperty<Real>;
    using GlobalMatrixType = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<Real>;

public:

    sofa::Data<bool> d_lumping;

    /**
     * @brief Link to the state holding the spatial (rest) positions of the mesh.
     *
     * If not set, and if DataTypes and SpatialTypes are the same, the mechanical state of this
     * component is used. Otherwise, the first compatible state found in the current context is used.
     */
    sofa::SingleLink<BaseFEMMassMatrix, sofa::core::behavior::MechanicalState<SpatialTypes>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_spatialState;

    /**
     * @brief Initializes the component.
     *
     * 1. Initializes the topology accessor.
     * 2. Initializes the base Mass class.
     * 3. Validates that a valid nodal coefficient is linked.
     * 4. Validates that a valid spatial state is linked.
     * 5. Computes and assembles the global matrix.
     */
    void init() final;

    /**
     * @brief Indicates whether the matrix is diagonal.
     * @return False in general, unless the matrix is lumped.
     */
    bool isDiagonal() const override { return d_lumping.getValue(); }

    /**
     * @brief Accumulates the matrix into a global matrix accumulator.
     *
     * Each stored scalar is repeated on the diagonal of a block of size
     * `field_dimensions * field_dimensions`.
     */
    void buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices) override;

    using Inherit1::addMDx;
    /**
     * @brief Adds the product of the matrix and a vector to another vector (f += C * dx * factor).
     */
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv_t<DataTypes>& f, const DataVecDeriv_t<DataTypes>& dx, SReal factor) override;

    /**
     * @brief Compute $ a = C^{-1} f $ if the matrix is lumped. Triggers an error otherwise.
     */
    void doSolveLinearSystem(const core::MechanicalParams* mparams, DataVecDeriv_t<DataTypes>& a, const DataVecDeriv_t<DataTypes>& f) override;

    using Inherit1::getKineticEnergy;
    /**
     * @brief Computes \f$ \frac{1}{2} \mathbf{v}^T \mathbf{C} \mathbf{v} \f$.
     */
    SReal getKineticEnergy(const core::MechanicalParams* mparams,
                           const DataVecDeriv_t<DataTypes>& v) const override;

protected:

    BaseFEMMassMatrix();

    /**
     * @brief Returns the component providing the coefficient at each node (mass density,
     * capacity...), or nullptr if there is none.
     */
    virtual const NodalCoefficient* getNodalCoefficient() const = 0;

    /**
     * @brief Ensures that a valid nodal coefficient component is linked. Must mark the component
     * state as invalid otherwise.
     */
    virtual void validateNodalCoefficient() = 0;

    /**
     * @brief Ensures that a valid state providing the spatial positions is linked.
     *
     * If no link is set, it uses the mechanical state of this component if its type is compatible,
     * or attempts to find a suitable component in the current context.
     * If no component is found, it marks the component state as invalid.
     */
    void validateSpatialState();

    /**
     * @brief Computes the element matrices and assembles them into `m_globalMatrix`.
     */
    void assembleGlobalMatrix();

    /**
     * @brief Computes the matrix of each finite element, by numerical integration.
     *
     * @param elements The sequence of elements to process.
     * @param[out] elementMatrices A vector to be populated with the computed local matrices.
     */
    void computeElementMatrices(const auto& elements, sofa::type::vector<ElementMatrix>& elementMatrices);

    /**
     * @brief Assembles the global matrix from individual element matrices.
     *
     * @param elements The sequence of elements corresponding to the provided matrices.
     * @param elementMatrices The precomputed local matrices for each element.
     */
    void initializeGlobalMatrix(const auto& elements, const sofa::type::vector<ElementMatrix>& elementMatrices);

    /**
     * @brief Global matrix, in compressed sparse row format.
     *
     * Only the scalar factor of each identity block of size `field_dimensions * field_dimensions`
     * is stored. For example, for a Vec3 field, a contribution is the 3x3 diagonal matrix `m * I`,
     * and only `m` is stored.
     */
    GlobalMatrixType m_globalMatrix;
};

#if !defined(SOFA_COMPONENT_MASS_BASEFEMMASSMATRIX_CPP)
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;

extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
extern template class SOFA_COMPONENT_MASS_API BaseFEMMassMatrix<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
#endif

}  // namespace sofa::component::mass
