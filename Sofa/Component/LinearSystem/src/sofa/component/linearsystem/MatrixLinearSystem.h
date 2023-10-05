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

#include <sofa/component/linearsystem/config.h>
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/component/linearsystem/MappingGraph.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/component/linearsystem/matrixaccumulators/AssemblingMappedMatrixAccumulator.h>
#include <sofa/component/linearsystem/CreateMatrixDispatcher.h>
#include <optional>

namespace sofa::component::linearsystem
{

using sofa::core::behavior::BaseForceField;
using sofa::core::behavior::BaseMass;
using sofa::core::BaseMapping;
using sofa::core::matrixaccumulator::Contribution;

/**
 * Data structure storing local matrix components created during the matrix assembly and associated
 * to each component contributing to the global matrix
 */
template<Contribution c, class Real>
struct LocalMatrixMaps;

struct GroupOfComponentsAssociatedToAPairOfMechanicalStates;

/**
 * Assemble the global matrix using local matrix components
 *
 * Components add their contributions directly to the global matrix, through their local matrices.
 * Local matrices act as a proxy (they don't really store a local matrix). They have a link to the global matrix and
 * an offset parameter to add the contribution into the right entry into the global matrix.
 *
 * @tparam TMatrix The type of the data structure used to represent the global matrix. In the general cases, this type
 * derives from sofa::linearalgebra::BaseMatrix.
 * @tparam TVector The type of the data structure used to represent the vectors of the linear system: the right-hand
 * side and the solution. In the general cases, this type derives from sofa::linearalgebra::BaseVector.
 */
template<class TMatrix, class TVector>
class MatrixLinearSystem : public TypedMatrixLinearSystem<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MatrixLinearSystem, TMatrix, TVector), SOFA_TEMPLATE2(TypedMatrixLinearSystem, TMatrix, TVector));

    using Real = typename TMatrix::Real;
    using Contribution = core::matrixaccumulator::Contribution;
    using PairMechanicalStates = sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2>;

    [[nodiscard]] const MappingGraph& getMappingGraph() const;

    Data< bool > d_assembleStiffness; ///< If true, the stiffness is added to the global matrix
    Data< bool > d_assembleMass; ///< If true, the mass is added to the global matrix
    Data< bool > d_assembleDamping; ///< If true, the damping is added to the global matrix
    Data< bool > d_assembleGeometricStiffness; ///< If true, the geometric stiffness of mappings is added to the global matrix
    Data< bool > d_applyProjectiveConstraints; ///< If true, projective constraints are applied on the global matrix
    Data< bool > d_applyMappedComponents; ///< If true, mapped components contribute to the global matrix
    Data< bool > d_checkIndices; ///< If true, indices are verified before being added in to the global matrix, favoring security over speed
    Data< bool > d_parallelAssemblyIndependentMatrices; ///< If true, independent matrices (global matrix vs mapped matrices) are assembled in parallel

protected:

    MatrixLinearSystem();

    using Inherit1::m_mappingGraph;

    /**
     * Storage of all matrix accumulators
     */
    std::tuple<
        LocalMatrixMaps<Contribution::STIFFNESS          , Real>,
        LocalMatrixMaps<Contribution::MASS               , Real>,
        LocalMatrixMaps<Contribution::DAMPING            , Real>,
        LocalMatrixMaps<Contribution::GEOMETRIC_STIFFNESS, Real>
    > m_localMatrixMaps;

    std::map<BaseForceField*, core::behavior::StiffnessMatrix> m_stiffness;
    std::map<BaseForceField*, core::behavior::DampingMatrix> m_damping;
    std::map<BaseMapping*, core::GeometricStiffnessMatrix> m_geometricStiffness;
    std::map<BaseMass*, BaseAssemblingMatrixAccumulator<Contribution::MASS>*> m_mass;

    struct IndependentContributors
    {
        std::map<BaseForceField*, core::behavior::StiffnessMatrix> m_stiffness;
        std::map<BaseForceField*, core::behavior::DampingMatrix> m_damping;
        std::map<BaseMapping*, core::GeometricStiffnessMatrix> m_geometricStiffness;
        std::map<BaseMass*, BaseAssemblingMatrixAccumulator<Contribution::MASS>*> m_mass;
        int id {};
    };

    sofa::type::vector<IndependentContributors> m_independentContributors;


    /// List of shared local matrices under mappings
    sofa::type::vector< std::pair<
        PairMechanicalStates,
        std::shared_ptr<LocalMappedMatrixType<Real> >
    > > m_localMappedMatrices;



    /**
     * Asks all the matrix accumulators to accumulate the contribution of a specific type of contribution
     */
    template<Contribution c>
    void contribute(const core::MechanicalParams* mparams);

    template<Contribution c>
    void contribute(const core::MechanicalParams* mparams, IndependentContributors& contributors);

    void assembleSystem(const core::MechanicalParams* mparams) override;

    /**
     * Gather all components associated to the same mechanical state into groups
     */
    void makeLocalMatrixGroups(const core::MechanicalParams* mparams);

    void makeIndependentLocalMatrixGroups();

    /**
     * Create the matrix accumulators and associate them to all components that have a contribution
     */
    void associateLocalMatrixToComponents(const core::MechanicalParams* mparams) override;

    /**
     * Remove the matrix accumulators
     */
    void cleanLocalMatrices();

    /**
     * Return the element of the tuple corresponding to @c
     * Example: getLocalMatrixMap<Contribution::STIFFNESS>()
     */
    template<Contribution c>
    LocalMatrixMaps<c, Real>& getLocalMatrixMap();

    /**
     * Return the element of the tuple corresponding to @c
     * Example: getLocalMatrixMap<Contribution::STIFFNESS>()
     */
    template<Contribution c>
    const LocalMatrixMaps<c, Real>& getLocalMatrixMap() const;


    /// Associate a local matrix to the provided component. The type of the local matrix depends on
    /// the situtation of the component: type of the component, mapped vs non-mapped
    template<Contribution c>
    void associateLocalMatrixTo(sofa::core::matrixaccumulator::get_component_type<c>* component,
                                const core::MechanicalParams* mparams);

    template<Contribution c>
    BaseAssemblingMatrixAccumulator<c>* createLocalMatrixT(
        sofa::core::matrixaccumulator::get_component_type<c>* object,
        SReal factor);

    template<Contribution c>
    AssemblingMappedMatrixAccumulator<c, Real>* createLocalMappedMatrixT(sofa::core::matrixaccumulator::get_component_type<c>* object, SReal factor);

    using JacobianMatrixType = LocalMappedMatrixType<Real>;


    /**
     * Project the assembled matrices from mapped states to the global matrix
     */
    virtual void projectMappedMatrices(const core::MechanicalParams* mparams);


    /**
     * Build the jacobian matrices of mappings from a mapped state to its top most parents (in the
     * sense of mappings)
     */
    MappingJacobians<JacobianMatrixType> computeJacobiansFrom(BaseMechanicalState* mstate, const core::MechanicalParams* mparams, LocalMappedMatrixType<Real>* crs);

    /**
     * Assemble the matrices under mappings into the global matrix
     */
    virtual void assembleMappedMatrices(const core::MechanicalParams* mparams);

    virtual void applyProjectiveConstraints(const core::MechanicalParams* mparams);

    template <Contribution c>
    std::shared_ptr<LocalMappedMatrixType<Real> > getSharedMatrix(
        sofa::core::matrixaccumulator::get_component_type<c>* object,
        const PairMechanicalStates& pair);

    template <Contribution c>
    std::optional<type::Vec2u> getSharedMatrixSize(
        sofa::core::matrixaccumulator::get_component_type<c>* object,
        const PairMechanicalStates& pair);

    template <Contribution c>
    void setSharedMatrix(sofa::core::matrixaccumulator::get_component_type<c>* object, const PairMechanicalStates& pair, std::shared_ptr<LocalMappedMatrixType<Real> > matrix);

    /**
     * Define how zero Dirichlet boundary conditions are applied on the global matrix
     */
    struct Dirichlet final : public sofa::core::behavior::ZeroDirichletCondition
    {
        ~Dirichlet() override = default;
        void discardRowCol(sofa::Index row, sofa::Index col) override;

        sofa::type::Vec2u m_offset;

        /// The matrix to apply a zero Dirichlet boundary condition
        TMatrix* m_globalMatrix { nullptr };
    } m_discarder;

    Data<bool> m_needClearLocalMatrices { false };

    /// Get the list of components contributing to the global matrix through the contribution type @c
    template<Contribution c>
    sofa::type::vector<sofa::core::matrixaccumulator::get_component_type<c>*> getContributors() const;

    void buildGroupsOfComponentAssociatedToMechanicalStates(
        std::map< PairMechanicalStates, GroupOfComponentsAssociatedToAPairOfMechanicalStates>& groups);

    /// Given a Mechanical State and its matrix, identifies the nodes affected by the matrix
    std::vector<unsigned int> identifyAffectedDoFs(BaseMechanicalState* mstate, LocalMappedMatrixType<Real>* crs);

    /// An object with factory methods to create local matrices
    std::tuple<
        std::unique_ptr<CreateMatrixDispatcher<Contribution::STIFFNESS          >>,
        std::unique_ptr<CreateMatrixDispatcher<Contribution::MASS               >>,
        std::unique_ptr<CreateMatrixDispatcher<Contribution::DAMPING            >>,
        std::unique_ptr<CreateMatrixDispatcher<Contribution::GEOMETRIC_STIFFNESS>>
    > m_createDispatcher;

    /// Define the type of dispatcher, itself defining the type of local matrices
    /// To override if matrix accumulation methods differs from this class.
    virtual void makeCreateDispatcher();

private:
    template<Contribution c>
    static std::unique_ptr<CreateMatrixDispatcher<c>> makeCreateDispatcher();
};

template<Contribution c, class Real>
struct LocalMatrixMaps
{
    using ListMatrixType = sofa::core::matrixaccumulator::get_list_abstract_strong_type<c>;
    using ComponentType = sofa::core::matrixaccumulator::get_component_type<c>;
    using PairMechanicalStates = sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2>;

    /// The local matrix (value) that has been created and associated to a mapped component (key)
    std::map< ComponentType*, std::map<PairMechanicalStates, AssemblingMappedMatrixAccumulator<c, Real>*> > mappedLocalMatrix;
    /// A verification strategy allowing to verify that the matrix indices provided are valid
    std::map< ComponentType*, std::shared_ptr<core::matrixaccumulator::RangeVerification> > indexVerificationStrategy;


    std::map< ComponentType*, std::map<PairMechanicalStates, BaseAssemblingMatrixAccumulator<c>* > > componentLocalMatrix;

    void clear()
    {
        for (const auto& [component, matrixMap] : componentLocalMatrix)
        {
            for (const auto& [pair, matrix] : matrixMap)
            {
                component->removeSlave(matrix);
            }
        }

        mappedLocalMatrix.clear();
        indexVerificationStrategy.clear();
        componentLocalMatrix.clear();
    }
};

struct GroupOfComponentsAssociatedToAPairOfMechanicalStates
{
    std::set<BaseForceField*> forcefieds;
    std::set<BaseMass*> masses;
    std::set<BaseMapping*> mappings;

    friend std::ostream& operator<<(std::ostream& os,
        const GroupOfComponentsAssociatedToAPairOfMechanicalStates& group);
};

inline std::ostream& operator<<(std::ostream& os,
    const GroupOfComponentsAssociatedToAPairOfMechanicalStates& group)
{
    constexpr auto join = [](const auto& components)
    {
        return sofa::helper::join(components.begin(), components.end(),
            [](auto* component) { return component ? component->getPathName() : "null"; }, ",");
    };

    if (!group.masses.empty())
    {
        os << "masses [" << join(group.masses) << "]";
        if (!group.forcefieds.empty() || !group.mappings.empty()) os << ", ";
    }
    if (!group.forcefieds.empty())
    {
        os << "force fields [" << join(group.forcefieds) << "]";
        if (!group.mappings.empty()) os << ", ";
    }
    if (!group.mappings.empty())
    {
        os << "mappings [" << join(group.mappings) << "]";
    }
    return os;
}

} //namespace sofa::component::linearsystem
