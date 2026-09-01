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
#include <sofa/component/solidmechanics/fem/elastic/ConstantSourceTerm.h>
#include <sofa/component/solidmechanics/fem/elastic/NonConstantSourceTerm.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/fem/FiniteElement.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_FEM_SOURCE_TERM_INTEGRATOR_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class FEMSourceTermIntegrator
 * @brief Integrates source terms into consistent nodal loads.
 *
 * A source term contributes \f$ \int_{\Omega} N_a \, r \, d\Omega \f$ to the right-hand side. r is
 * the per-node density carried by a linked ConstantSourceTerm (through l_constantSources) or
 * NonConstantSourceTerm (through l_nonConstantSources). Constant terms do not depend on the
 * displacement, so they are summed and integrated once in init(); non-constant ones are
 * re-integrated at every call.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template <class TDataTypes, class TElementType>
class FEMSourceTermIntegrator :
    public sofa::core::behavior::ForceField<TDataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    SOFA_CLASS2(SOFA_TEMPLATE2(FEMSourceTermIntegrator, DataTypes, ElementType),
        sofa::core::behavior::ForceField<DataTypes>,
        sofa::core::behavior::TopologyAccessor);

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;
    using Real = sofa::Real_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using DataVecCoord = sofa::DataVecCoord_t<DataTypes>;
    using DataVecDeriv = sofa::DataVecDeriv_t<DataTypes>;

    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    using ElementMatrix = sofa::type::Mat<NumberOfNodesInElement, NumberOfNodesInElement, sofa::Real_t<DataTypes>>;
    using GlobalMatrix = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<sofa::Real_t<DataTypes>>;
    using Element = typename FiniteElement::TopologyElement;
    using ShapeFunctions = sofa::type::Vec<NumberOfNodesInElement, Real>;
    using Jacobian = sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>;
    using SourceDerivative = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

public:

    /**
     * @brief Source terms integrated by this component.
     *
     * If left empty, the ConstantSourceTerm components found in the current context are used.
     */
    sofa::MultiLink<FEMSourceTermIntegrator<DataTypes, ElementType>, ConstantSourceTerm<DataTypes>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_constantSources;

    /**
     * @brief Displacement-dependent source terms linked to this component.
     *
     * If left empty, the NonConstantSourceTerm components found in the current context are used.
     */
    sofa::MultiLink<FEMSourceTermIntegrator<DataTypes, ElementType>, NonConstantSourceTerm<DataTypes, ElementType>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nonConstantSources;

    /**
     * @brief Initializes the component.
     *
     * This method performs several initialization steps:
     * 1. Initializes the base force field.
     * 2. Initializes the topology accessor.
     * 3. Validates the linked source terms.
     * 4. Assembles the global matrix M.
     * 5. Integrates the source terms into the constant nodal force.
     */
    void init() override;

    /**
     * @brief Adds the nodal source term to the RHS vector.
     *
     * The constant terms were integrated once in init and are only accumulated here. The
     * displacement-dependent ones are integrated at the current position.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param f The force vector to which the source term will be added.
     * @param x The current positions, used only by displacement-dependent terms.
     * @param v The current velocities (unused in this implementation).
     */
    void addForce(
        const sofa::core::MechanicalParams* mparams,
        DataVecDeriv& f,
        const DataVecCoord& x,
        const DataVecDeriv& v) override;

    /**
     * @brief Applies the tangent of the displacement-dependent terms to dx.
     *
     * A no-op when l_nonConstantSources is empty.
     */
    void addDForce(const sofa::core::MechanicalParams* mparams,
        DataVecDeriv& df,
        const DataVecDeriv& dx) override;

    /**
     * @brief Assembles the stiffness contribution of the displacement-dependent terms.
     *
     * A no-op when l_nonConstantSources is empty.
     */
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    using sofa::core::behavior::ForceField<DataTypes>::getPotentialEnergy;
    /**
     * @brief Potential energy of the constant nodal load, \f$ V = -\sum_a F_a \cdot (x_a - x_{0,a}) \f$.
     */
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
        const DataVecCoord& x) const override;

    /**
     * @brief Degree of the quadrature rule integrating the element matrix M.
     */
    sofa::Data<sofa::Size> d_quadratureDegree;

    /**
     * @brief Whether to assemble/apply the stiffness of the displacement-dependent terms.
     */
    sofa::Data<bool> d_useTangentStiffness;

protected:

    /**
     * @brief Default constructor.
     */
    FEMSourceTermIntegrator();

    /**
     * @brief Ensures that valid source terms are linked, falling back to the current context.
     *
     * Applies to both l_constantSources and l_nonConstantSources.
     */
    void validateSources();

    /**
     * @brief Assembles and stores the geometry-only matrix \f$ M_{ij} = \int_{\Omega} N_i N_j \, d\Omega \f$ over each element on the rest configuration.
     */
    void assembleGlobalMatrix();

    /**
     * @brief Sums every displacement-independent source density and integrates it once into m_constantForce.
     *
     * Integration is linear, so the sum of the terms integrates to the sum of their contributions:
     * a single matrix-vector product covers all of them.
     */
    void assembleConstantForce();

    /**
     * @brief Applies the geometry-only matrix M to a nodal source term.
     */
    void applyGlobalMatrix(const VecDeriv& nodalSourceTerm,
        VecDeriv& result) const;

    /**
     * @brief Computes the geometry-only matrix of each element, caching the Jacobian of the
     * reference-to-physical mapping of every quadrature point along the way in m_referenceJacobian.
     */
    void calculateElementMatrix(const auto& elements, sofa::type::vector<ElementMatrix>& elementMatrices);

    /**
     * @brief Scatters the element matrices into the global matrix.
     */
    void initializeGlobalMatrix(const auto& elements, const sofa::type::vector<ElementMatrix>& elementMatrices);

    /**
     * @brief Geometry-only matrix \f$ M_{ij} = \int_{\Omega} N_i N_j \, d\Omega \f$ of the system.
     *
     * Stored in compressed sparse row format. Assembled once in init on the rest configuration.
     */
    GlobalMatrix m_globalMatrix;

    /**
     * @brief Nodal load of every term in l_constantSources, integrated once in init.
     *
     * @note Their contribution is integrated once, so editing the property of a linked term at run
     * time has no effect until the scene is reinitialised.
     */
    VecDeriv m_constantForce;

    /**
     * @brief Jacobian of the reference-to-physical mapping, evaluated on the rest configuration.
     *
     * Assembled once in init on the rest configuration.
     */
    sofa::type::vector<Jacobian> m_referenceJacobian;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_FEM_SOURCE_TERM_INTEGRATOR_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
