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
#include <sofa/component/mass/NodalMassDensity.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>

#if !defined(SOFA_COMPONENT_MASS_FEMMASS_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::mass
{

/**
 * @class FEMMass
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
 * The quantity carried by the degrees of freedom (DataTypes) can differ from the type of the
 * coordinates of the mesh (SpatialTypes). For example, a scalar field (Vec1Types) can be defined on
 * a volumetric mesh (Vec3Types). In that case, the geometry of the elements is read from the rest
 * positions of the state linked by `spatialState`.
 *
 * @tparam TDataTypes The type of the quantity carried by the degrees of freedom (e.g., Vec3Types).
 * @tparam TSpatialTypes The type of the spatial coordinates of the mesh (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template<class TDataTypes, class TSpatialTypes, class TElementType>
class FEMMass :
    public core::behavior::Mass<TDataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor
{
public:
    using DataTypes = TDataTypes;
    using SpatialTypes = TSpatialTypes;
    using ElementType = TElementType;
    SOFA_CLASS2(SOFA_TEMPLATE3(FEMMass, DataTypes, SpatialTypes, ElementType),
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

    /// The gravity (body force) is applied only if the field is a vector field of the same
    /// dimension as the space (i.e. a displacement-like field).
    static constexpr bool isGravityApplicable = (field_dimensions == spatial_dimensions);

    using ElementMassMatrix = sofa::type::Mat<NumberOfNodesInElement, NumberOfNodesInElement, Real>;

    using NodalMassDensity = ::sofa::component::mass::NodalMassDensity<Real>;
    using GlobalMassMatrixType = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<Real>;

public:

    sofa::Data<bool> d_lumping;

    /**
     * @brief Link to the nodal mass density component.
     *
     * This component provides the mass density at each node of the mesh.
     * It must be present in the context for the mass to be calculated correctly.
     */
    sofa::SingleLink<FEMMass, NodalMassDensity,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nodalMassDensity;

    /**
     * @brief Link to the state holding the spatial (rest) positions of the mesh.
     *
     * If not set, and if DataTypes and SpatialTypes are the same, the mechanical state of this
     * component is used. Otherwise, the first compatible state found in the current context is used.
     */
    sofa::SingleLink<FEMMass, sofa::core::behavior::MechanicalState<SpatialTypes>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_spatialState;

    /**
     * @brief Initializes the component.
     *
     * This method performs several initialization steps:
     * 1. Initializes the topology accessor.
     * 2. Initializes the base Mass class.
     * 3. Validates that a valid nodal mass density is linked.
     * 4. Validates that a valid spatial state is linked.
     * 5. Computes and assembles the global mass matrix.
     */
    void init() final;

    /**
     * @brief Indicates whether the mass matrix is diagonal.
     * @return Returns false in general as FEM mass matrices are generally not diagonal unless
     * lumped. In that case, it returns true.
     */
    bool isDiagonal() const override { return d_lumping.getValue(); }

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

    /**
     * @brief Compute $ a = M^{-1} f $ if the mass matrix is lumped. Triggers an error otherwise.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param a The result vector of $M^{-1} f$.
     * @param f The vector to be multiplied by the inverse mass matrix.
     */
    void doSolveLinearSystem(const core::MechanicalParams* mparams, DataVecDeriv_t<DataTypes>& a, const DataVecDeriv_t<DataTypes>& f) override;


    using Inherit1::getKineticEnergy;
    /**
     * @brief Computes the total kinetic energy of the system.
     *
     * The kinetic energy $T$ is computed as:
     * \f$ T = \frac{1}{2} \mathbf{v}^T \mathbf{M} \mathbf{v} \f$
     * where \f$ \mathbf{M} \f$ is the global mass matrix and \f$ \mathbf{v} \f$ is the velocity vector.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param v The current velocity vector.
     * @return The computed kinetic energy.
     */
    SReal getKineticEnergy(const core::MechanicalParams* mparams,
                           const DataVecDeriv_t<DataTypes>& v) const override;

    using Inherit1::getPotentialEnergy;
    /**
     * @brief Computes the total gravitational potential energy of the system.
     *
     * The potential energy $V$ is computed as:
     * \f$ V = - \mathbf{x}^T \mathbf{M} \mathbf{g} \f$
     * where \f$ \mathbf{M} \f$ is the global mass matrix, \f$ \mathbf{x} \f$ is the position vector,
     * and \f$ \mathbf{g} \f$ is the gravity vector.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param x The current position vector.
     * @return The computed gravitational potential energy.
     */
    SReal getPotentialEnergy(
        const core::MechanicalParams* mparams,
        const DataVecCoord_t<DataTypes>& x) const override;

protected:

    /**
     * @brief Default constructor.
     */
    FEMMass();

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
     * @brief Ensures that a valid state providing the spatial positions is linked.
     *
     * If no link is set, it uses the mechanical state of this component if its type is compatible,
     * or attempts to find a suitable component in the current context.
     * If no component is found, it marks the component state as invalid.
     */
    void validateSpatialState();

    /**
     * @brief Represents the global mass matrix of the system.
     *
     * This matrix is stored in a compressed sparse row format.
     * Since the mass contribution for each node is isotropic (affects all field components
     * identically), this matrix stores only the scalar scaling factor for each identity block
     * of size `field_dimensions * field_dimensions`.
     *
     * For example, in 3D, a node's mass contribution is a 3x3 diagonal matrix `m * I`.
     * Only `m` is stored here.
     */
    GlobalMassMatrixType m_globalMassMatrix;
};

#if !defined(SOFA_COMPONENT_MASS_FEMMASS_CPP)
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticEdge>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticTriangle>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTriangle>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::QuadraticQuad>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticQuad>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticTetrahedron>;

extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
extern template class SOFA_COMPONENT_MASS_API FEMMass<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::QuadraticHexahedron>;
#endif

}  // namespace sofa::component::mass
