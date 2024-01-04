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
#include <optional>
#include <unordered_set>
#include <mutex>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.inl>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/linearsystem/MatrixMapping.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalIdentityBlocksInJacobianVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalIdentityBlocksInJacobianVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalResetConstraintVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateJacobian.h>
using sofa::simulation::mechanicalvisitor::MechanicalAccumulateJacobian;

namespace sofa::component::linearsystem
{

template <class TMatrix, class TVector>
MatrixLinearSystem<TMatrix, TVector>::MatrixLinearSystem()
    : Inherit1()
    , d_assembleStiffness         (initData(&d_assembleStiffness,          true,  "assembleStiffness",          "If true, the stiffness is added to the global matrix"))
    , d_assembleMass              (initData(&d_assembleMass,               true,  "assembleMass",               "If true, the mass is added to the global matrix"))
    , d_assembleDamping           (initData(&d_assembleDamping,            true,  "assembleDamping",            "If true, the damping is added to the global matrix"))
    , d_assembleGeometricStiffness(initData(&d_assembleGeometricStiffness, true,  "assembleGeometricStiffness", "If true, the geometric stiffness of mappings is added to the global matrix"))
    , d_applyProjectiveConstraints(initData(&d_applyProjectiveConstraints, true,  "applyProjectiveConstraints", "If true, projective constraints are applied on the global matrix"))
    , d_applyMappedComponents     (initData(&d_applyMappedComponents,      true,  "applyMappedComponents",      "If true, mapped components contribute to the global matrix"))
    , d_checkIndices              (initData(&d_checkIndices,               false, "checkIndices",               "If true, indices are verified before being added in to the global matrix, favoring security over speed"))
    , d_parallelAssemblyIndependentMatrices
        (initData(&d_parallelAssemblyIndependentMatrices, false, "parallelAssemblyIndependentMatrices", "If true, independent matrices (global matrix vs mapped matrices) are assembled in parallel"))
{
    this->addUpdateCallback("updateCheckIndices", {&d_checkIndices}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        m_needClearLocalMatrices.setValue(true);
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {&m_needClearLocalMatrices});
}

template <class TMatrix, class TVector>
template <Contribution c>
void MatrixLinearSystem<TMatrix, TVector>::contribute(const core::MechanicalParams* mparams)
{
    sofa::helper::ScopedAdvancedTimer buildTimer("build" + std::string(core::matrixaccumulator::GetContributionName<c>()));

    for (auto* contributor : getContributors<c>())
    {
        if (Inherit1::template getContributionFactor<c>(mparams, contributor) != 0._sreal)
        {
            if constexpr (c == Contribution::STIFFNESS)
            {
                contributor->buildStiffnessMatrix(&m_stiffness[contributor]);
            }
            else if constexpr (c == Contribution::MASS)
            {
                contributor->buildMassMatrix(m_mass[contributor]);
            }
            else if constexpr (c == Contribution::DAMPING)
            {
                contributor->buildDampingMatrix(&m_damping[contributor]);
            }
            else if constexpr (c == Contribution::GEOMETRIC_STIFFNESS)
            {
                contributor->buildGeometricStiffnessMatrix(&m_geometricStiffness[contributor]);
            }
        }
    }
}

template <class TMatrix, class TVector>
template <Contribution c>
void MatrixLinearSystem<TMatrix, TVector>::contribute(
    const core::MechanicalParams* mparams,
    IndependentContributors& contributors)
{
    if constexpr (c == Contribution::STIFFNESS)
    {
        for (auto& [component, stiffnessMatrix] : contributors.m_stiffness)
        {
            if (Inherit1::template getContributionFactor<c>(mparams, component) != 0._sreal)
            {
                component->buildStiffnessMatrix(&stiffnessMatrix);
            }
        }
    }
    else if constexpr (c == Contribution::MASS)
    {
        for (auto& [component, massMatrix] : contributors.m_mass)
        {
            if (Inherit1::template getContributionFactor<c>(mparams, component) != 0._sreal)
            {
                component->buildMassMatrix(massMatrix);
            }
        }
    }
    else if constexpr (c == Contribution::DAMPING)
    {
        for (auto& [component, dampingMatrix] : contributors.m_damping)
        {
            if (Inherit1::template getContributionFactor<c>(mparams, component) != 0._sreal)
            {
                component->buildDampingMatrix(&dampingMatrix);
            }
        }
    }
    else if constexpr (c == Contribution::GEOMETRIC_STIFFNESS)
    {
        for (auto& [component, geometricStiffnessMatrix] : contributors.m_geometricStiffness)
        {
            if (Inherit1::template getContributionFactor<c>(mparams, component) != 0._sreal)
            {
                component->buildGeometricStiffnessMatrix(&geometricStiffnessMatrix);

            }
        }
    }
}

template<class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::assembleSystem(const core::MechanicalParams* mparams)
{
    if (this->getSystemMatrix()->rowSize() == 0 || this->getSystemMatrix()->colSize() == 0)
    {
        msg_error() << "Global system matrix is not resized appropriatly (" << this->getPathName() << ")";
        return;
    }

    SCOPED_TIMER_VARNAME(assembleSystemTimer, "AssembleSystem");

    {
        SCOPED_TIMER_VARNAME(buildMatricesTimer, "buildMatrices");

        simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
        assert(taskScheduler);

        if (d_parallelAssemblyIndependentMatrices.getValue() && taskScheduler && taskScheduler->getThreadCount() < 1)
        {
            taskScheduler->init(0);
        }

        const simulation::ForEachExecutionPolicy execution = d_parallelAssemblyIndependentMatrices.getValue() ?
            simulation::ForEachExecutionPolicy::PARALLEL :
            simulation::ForEachExecutionPolicy::SEQUENTIAL;

        const bool assembleStiffness = d_assembleStiffness.getValue();
        const bool assembleMass = d_assembleMass.getValue();
        const bool assembleDamping = d_assembleDamping.getValue();
        const bool assembleGeometricStiffness = d_assembleGeometricStiffness.getValue();

        int counter{};
        for (auto& c : m_independentContributors)
        {
            c.id = counter++;
        }

        simulation::forEach(execution, *taskScheduler,
            m_independentContributors.begin(), m_independentContributors.end(),
            [this, mparams, assembleStiffness, assembleMass, assembleDamping, assembleGeometricStiffness](IndependentContributors& contributors)
            {
                helper::ScopedAdvancedTimer timerContributors("buildContributors" + std::to_string(contributors.id));

                if (assembleStiffness)
                {
                    helper::ScopedAdvancedTimer timerStiffness("buildStiffness" + std::to_string(contributors.id));
                    contribute<Contribution::STIFFNESS>(mparams, contributors);
                }

                if (assembleMass)
                {
                    helper::ScopedAdvancedTimer timerMass("buildMass" + std::to_string(contributors.id));
                    contribute<Contribution::MASS>(mparams, contributors);
                }

                if (assembleDamping)
                {
                    helper::ScopedAdvancedTimer timerDamping("buildDamping" + std::to_string(contributors.id));
                    contribute<Contribution::DAMPING>(mparams, contributors);
                }

                if (assembleGeometricStiffness)
                {
                    helper::ScopedAdvancedTimer timerGeometricStiffness("buildGeometricStiffness" + std::to_string(contributors.id));
                    contribute<Contribution::GEOMETRIC_STIFFNESS>(mparams, contributors);
                }
            });
    }

    if (d_applyMappedComponents.getValue() && m_mappingGraph.hasAnyMapping())
    {
        assembleMappedMatrices(mparams);
    }

    if (d_applyProjectiveConstraints.getValue())
    {
        applyProjectiveConstraints(mparams);
    }
}


inline sofa::type::vector<core::behavior::BaseMechanicalState*> retrieveAssociatedMechanicalState(
    const sofa::core::behavior::StateAccessor* component)
{
    const auto& mstatesLinks = component->getMechanicalStates();

    sofa::type::vector<core::behavior::BaseMechanicalState*> mstates;
    mstates.reserve(mstatesLinks.size());
    for (auto m : mstatesLinks)
    {
        mstates.push_back(m);
    }

    //remove duplicates: it may happen for InteractionForceFields
    std::sort( mstates.begin(), mstates.end() );
    mstates.erase( std::unique( mstates.begin(), mstates.end() ), mstates.end() );

    return mstates;
}

inline sofa::type::vector<core::behavior::BaseMechanicalState*> retrieveAssociatedMechanicalState(BaseMapping* component)
{
    type::vector<BaseMechanicalState*> mstates = component->getMechFrom();

    //remove duplicates: it may happen for MultiMappings
    std::sort( mstates.begin(), mstates.end() );
    mstates.erase( std::unique( mstates.begin(), mstates.end() ), mstates.end() );

    return mstates;
}

/// Generate all possible pairs of Mechanical States from a list of Mechanical States
inline auto generatePairs(const sofa::type::vector<core::behavior::BaseMechanicalState*>& mstates)
-> sofa::type::vector<sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2> >
{
    sofa::type::vector<sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2> > pairs;
    pairs.reserve(mstates.size() * mstates.size());
    for (auto* a : mstates)
    {
        for (auto* b : mstates)
        {
            pairs.emplace_back(a, b);
        }
    }
    return pairs;
}

template <class TMatrix, class TVector>
template<Contribution c>
auto MatrixLinearSystem<TMatrix, TVector>::getSharedMatrix(
    sofa::core::matrixaccumulator::get_component_type<c>* object, const PairMechanicalStates& pair)
-> std::shared_ptr<LocalMappedMatrixType<Real> >
{
    const auto& localMaps = getLocalMatrixMap<c>().mappedLocalMatrix;
    const auto localMapIt = localMaps.find(object);
    if (localMapIt != localMaps.end())
    {
        const auto mappedAccumulatorsIt = localMapIt->second.find(pair);
        if (mappedAccumulatorsIt != localMapIt->second.end())
        {
            if (auto* accumulator = mappedAccumulatorsIt->second)
            {
                if (const auto& accumulatorMat = accumulator->getMatrix())
                {
                    if (accumulatorMat)
                    {
                        return accumulatorMat;
                    }
                }
            }
        }
    }
    return nullptr;
}

template <class TMatrix, class TVector>
template<Contribution c>
auto MatrixLinearSystem<TMatrix, TVector>::getSharedMatrixSize(
    sofa::core::matrixaccumulator::get_component_type<c>* object, const PairMechanicalStates& pair)
-> std::optional<type::Vec2u>
{
    const auto& localMaps = getLocalMatrixMap<c>().mappedLocalMatrix;
    const auto localMapIt = localMaps.find(object);
    if (localMapIt != localMaps.end())
    {
        const auto mappedAccumulatorsIt = localMapIt->second.find(pair);
        if (mappedAccumulatorsIt != localMapIt->second.end())
        {
            if (auto* accumulator = mappedAccumulatorsIt->second)
            {
                return accumulator->getMatrixSize();
            }
        }
    }
    return {};
}

template <class TMatrix, class TVector>
template<Contribution c>
void MatrixLinearSystem<TMatrix, TVector>::setSharedMatrix(
    sofa::core::matrixaccumulator::get_component_type<c>* object, const PairMechanicalStates& pair, std::shared_ptr<LocalMappedMatrixType<Real> > matrix)
{
    const auto& localMaps = getLocalMatrixMap<c>().mappedLocalMatrix;
    const auto localMapIt = localMaps.find(object);
    if (localMapIt != localMaps.end())
    {
        const auto mappedAccumulatorsIt = localMapIt->second.find(pair);
        if (mappedAccumulatorsIt != localMapIt->second.end())
        {
            if (auto* accumulator = mappedAccumulatorsIt->second)
            {
                accumulator->shareMatrix(matrix);
            }
        }
    }
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::buildGroupsOfComponentAssociatedToMechanicalStates(
    std::map<PairMechanicalStates, GroupOfComponentsAssociatedToAPairOfMechanicalStates>& groups)
{
    for (auto* ff : this->m_forceFields)
    {
        const auto mstates = retrieveAssociatedMechanicalState(ff);
        for (auto& pair : generatePairs(mstates))
        {
            groups[pair].forcefieds.insert(ff);
        }
    }

    for (auto* mass : this->m_masses)
    {
        const auto mstates = retrieveAssociatedMechanicalState(mass);
        for (auto& pair : generatePairs(mstates))
        {
            groups[pair].masses.insert(mass);
        }
    }

    for (auto* mapping : this->m_mechanicalMappings)
    {
        const auto mstates = retrieveAssociatedMechanicalState(mapping);
        for (auto& pair : generatePairs(mstates))
        {
            groups[pair].mappings.insert(mapping);
        }
    }
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::makeLocalMatrixGroups(const core::MechanicalParams* mparams)
{
    SOFA_UNUSED(mparams);

    const bool assembleStiffness = d_assembleStiffness.getValue();
    const bool assembleDamping = d_assembleDamping.getValue();
    const bool assembleMass = d_assembleMass.getValue();
    const bool assembleGeometricStiffness = d_assembleGeometricStiffness.getValue();

    m_localMappedMatrices.clear();

    std::map<PairMechanicalStates, GroupOfComponentsAssociatedToAPairOfMechanicalStates> groups;
    buildGroupsOfComponentAssociatedToMechanicalStates(groups);

    for (const auto& [pair, group] : groups)
    {
        const bool isMapped1 = m_mappingGraph.hasAnyMappingInput(pair[0]);
        const bool isMapped2 = m_mappingGraph.hasAnyMappingInput(pair[1]);

        if (isMapped1 || isMapped2)
        {
            std::shared_ptr<LocalMappedMatrixType<Real> > mat;

            if (assembleStiffness || assembleDamping)
            for (auto* component : group.forcefieds)
            {
                if (assembleStiffness)
                {
                    mat = getSharedMatrix<Contribution::STIFFNESS>(component, pair);
                }
                if (!mat && assembleDamping)
                {
                    mat = getSharedMatrix<Contribution::DAMPING>(component, pair);
                }
                if (mat)
                {
                    break;
                }
            }
            if (!mat)
            {
                if (assembleMass)
                for (auto* component : group.masses)
                {
                    mat = getSharedMatrix<Contribution::MASS>(component, pair);
                    if (mat)
                    {
                        break;
                    }
                }
            }
            if (!mat)
            {
                if (assembleGeometricStiffness)
                for (auto* component : group.mappings)
                {
                    mat = getSharedMatrix<Contribution::GEOMETRIC_STIFFNESS>(component, pair);
                    if (mat)
                    {
                        break;
                    }
                }
            }

            if (!mat && (
                (assembleStiffness && !group.forcefieds.empty()) ||
                (assembleDamping && !group.forcefieds.empty()) ||
                (assembleMass && !group.masses.empty()) ||
                (assembleGeometricStiffness && !group.mappings.empty())
            ))
            {
                std::string mstateNames = pair[0]->getPathName();
                if (pair[0] != pair[1])
                {
                    mstateNames += " and " + pair[1]->getPathName();
                }

                constexpr auto join = [](const auto& components)
                {
                    return sofa::helper::join(components.begin(), components.end(),
                        [](auto* component) { return component ? component->getPathName() : "null"; }, ",");
                };

                std::stringstream ss;
                if (!group.masses.empty() && assembleMass)
                {
                    ss << "masses [" << join(group.masses) << "]";
                    if (!group.forcefieds.empty() || !group.mappings.empty()) ss << ", ";
                }
                if (!group.forcefieds.empty() && (assembleDamping || assembleStiffness))
                {
                    ss << "force fields [" << join(group.forcefieds) << "]";
                    if (!group.mappings.empty()) ss << ", ";
                }
                if (!group.mappings.empty() && assembleGeometricStiffness)
                {
                    ss << "mappings [" << join(group.mappings) << "]";
                }

                msg_info() << "Create a matrix to be mapped, shared among the following components: "
                    << ss.str() << ", for a contribution on mechanical state " << mstateNames;
                mat = std::make_shared<LocalMappedMatrixType<Real> >();
            }

            if (mat)
            {
                std::optional<type::Vec2u> matrixSize;
                if (assembleStiffness || assembleDamping)
                    for (auto* component : group.forcefieds)
                    {
                        if (assembleStiffness)
                        {
                            setSharedMatrix<Contribution::STIFFNESS>(component, pair, mat);
                            if (!matrixSize.has_value())
                            {
                                matrixSize = getSharedMatrixSize<Contribution::STIFFNESS>(component, pair);
                            }
                        }
                        if (assembleDamping)
                        {
                            setSharedMatrix<Contribution::DAMPING>(component, pair, mat);

                            if (!matrixSize.has_value())
                            {
                                matrixSize = getSharedMatrixSize<Contribution::DAMPING>(component, pair);
                            }
                        }
                    }

                if (assembleMass)
                {
                    for (auto* component : group.masses)
                    {
                        setSharedMatrix<Contribution::MASS>(component, pair, mat);
                        if (!matrixSize.has_value())
                        {
                            matrixSize = getSharedMatrixSize<Contribution::MASS>(component, pair);
                        }
                    }
                }

                if (assembleGeometricStiffness)
                {
                    for (auto* component : group.mappings)
                    {
                        setSharedMatrix<Contribution::GEOMETRIC_STIFFNESS>(component, pair, mat);
                        if (!matrixSize.has_value())
                        {
                            matrixSize = getSharedMatrixSize<Contribution::GEOMETRIC_STIFFNESS>(component, pair);
                        }
                    }
                }

                if (matrixSize)
                {
                    mat->resize((*matrixSize)[0], (*matrixSize)[1]);
                }

                m_localMappedMatrices.emplace_back(pair, mat);
            }
        }
    }
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::makeIndependentLocalMatrixGroups()
{
    m_independentContributors.clear();

    IndependentContributors nonMappedContributors;
    IndependentContributors mappedContributors;

    for (auto& [component, localMatrix] : m_stiffness)
    {
        const auto& mappedMatrices = getLocalMatrixMap<Contribution::STIFFNESS>().mappedLocalMatrix;
        if (mappedMatrices.find(component) == mappedMatrices.end()) //this component is not mapped
        {
            //confirmation that this component is not mapped:
            if (m_mappingGraph.hasAnyMappingInput(component))
            {
                dmsg_error() << "A mapped component has no mapped local matrix. This should not happen.";
                continue;
            }

            nonMappedContributors.m_stiffness.insert({component, localMatrix});
        }
        else
        {
            mappedContributors.m_stiffness.insert({component, localMatrix});
        }
    }

    for (auto& [component, localMatrix] : m_damping)
    {
        const auto& mappedMatrices = getLocalMatrixMap<Contribution::DAMPING>().mappedLocalMatrix;
        if (mappedMatrices.find(component) == mappedMatrices.end()) //this component is not mapped
        {
            //confirmation that this component is not mapped:
            if (m_mappingGraph.hasAnyMappingInput(component))
            {
                dmsg_error() << "A mapped component has no mapped local matrix. This should not happen.";
                continue;
            }

            nonMappedContributors.m_damping.insert({component, localMatrix});
        }
        else
        {
            mappedContributors.m_damping.insert({component, localMatrix});
        }
    }

    for (auto& [component, localMatrix] : m_geometricStiffness)
    {
        const auto& mappedMatrices = getLocalMatrixMap<Contribution::GEOMETRIC_STIFFNESS>().mappedLocalMatrix;
        if (mappedMatrices.find(component) == mappedMatrices.end()) //this component is not mapped
        {
            nonMappedContributors.m_geometricStiffness.insert({component, localMatrix});
        }
        else
        {
            mappedContributors.m_geometricStiffness.insert({component, localMatrix});
        }
    }

    for (auto& [component, localMatrix] : m_mass)
    {
        const auto& mappedMatrices = getLocalMatrixMap<Contribution::MASS>().mappedLocalMatrix;
        if (mappedMatrices.find(component) == mappedMatrices.end()) //this component is not mapped
            {
            //confirmation that this component is not mapped:
            if (m_mappingGraph.hasAnyMappingInput(component))
            {
                dmsg_error() << "A mapped component has no mapped local matrix. This should not happen.";
                continue;
            }

            nonMappedContributors.m_mass.insert({component, localMatrix});
            }
        else
        {
            mappedContributors.m_mass.insert({component, localMatrix});
        }
    }

    m_independentContributors.push_back(nonMappedContributors);
    m_independentContributors.push_back(mappedContributors);
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::cleanLocalMatrices()
{
    getLocalMatrixMap<Contribution::STIFFNESS>().clear();
    getLocalMatrixMap<Contribution::MASS>().clear();
    getLocalMatrixMap<Contribution::DAMPING>().clear();
    getLocalMatrixMap<Contribution::GEOMETRIC_STIFFNESS>().clear();

    m_stiffness.clear();
    m_damping.clear();
    m_geometricStiffness.clear();
    m_mass.clear();
    m_independentContributors.clear();
}

template <class TMatrix, class TVector>
template <Contribution c>
LocalMatrixMaps<c, typename MatrixLinearSystem<TMatrix, TVector>::Real>&
MatrixLinearSystem<TMatrix, TVector>::getLocalMatrixMap()
{
    return std::get<LocalMatrixMaps<c, Real> >(m_localMatrixMaps);
}

template <class TMatrix, class TVector>
template <Contribution c>
const LocalMatrixMaps<c, typename MatrixLinearSystem<TMatrix, TVector>::Real>&
MatrixLinearSystem<TMatrix, TVector>::getLocalMatrixMap() const
{
    return std::get<LocalMatrixMaps<c, Real> >(m_localMatrixMaps);
}

template<class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::associateLocalMatrixToComponents(const core::MechanicalParams* mparams)
{
    SCOPED_TIMER("InitializeSystem");

    m_needClearLocalMatrices.updateIfDirty();
    if (m_needClearLocalMatrices.getValue())
    {
        cleanLocalMatrices();

        m_needClearLocalMatrices.setValue(false);
    }

    const sofa::Size totalSize = m_mappingGraph.getTotalNbMainDofs();
    this->d_matrixSize.setValue({totalSize, totalSize});
    m_discarder.m_globalMatrix = this->getSystemMatrix();

    {
        SCOPED_TIMER_VARNAME(resizeTimer, "resizeSystem");
        const auto rowSize = this->getSystemMatrix() ? this->getSystemMatrix()->rowSize() : 0;
        const auto colSize = this->getSystemMatrix() ? this->getSystemMatrix()->colSize() : 0;
        this->resizeSystem(totalSize);
        const auto newRowSize = this->getSystemMatrix() ? this->getSystemMatrix()->rowSize() : 0;
        const auto newColSize = this->getSystemMatrix() ? this->getSystemMatrix()->colSize() : 0;
        msg_info_when(newRowSize != rowSize || newColSize != colSize) <<
            "System matrix is resized from " << rowSize << " x " << colSize << " to " << newRowSize << " x " << newColSize;
    }
    {
        SCOPED_TIMER_VARNAME(clearSystemTimer, "clearSystem");
        this->clearSystem();
    }

    {
        SCOPED_TIMER_VARNAME(localMatricesTimer, "initializeLocalMatrices");

        if (d_assembleMass.getValue())
        {
            for (auto* m : this->m_masses)
            {
                associateLocalMatrixTo<Contribution::MASS>(m, mparams);
            }
        }

        if (d_assembleStiffness.getValue())
        {
            for (auto* ff : this->m_forceFields)
            {
                associateLocalMatrixTo<Contribution::STIFFNESS>(ff, mparams);
            }
        }

        if (d_assembleDamping.getValue())
        {
            for (auto* ff : this->m_forceFields)
            {
                associateLocalMatrixTo<Contribution::DAMPING>(ff, mparams);
            }
        }

        if (d_assembleGeometricStiffness.getValue())
        {
            for (auto* m : this->m_mechanicalMappings)
            {
                associateLocalMatrixTo<Contribution::GEOMETRIC_STIFFNESS>(m, mparams);
            }
        }

        makeLocalMatrixGroups(mparams);
        makeIndependentLocalMatrixGroups();
    }
}

template <class TMatrix, class TVector>
template <Contribution c>
sofa::type::vector<sofa::core::matrixaccumulator::get_component_type<c>*> MatrixLinearSystem<
TMatrix, TVector>::getContributors() const
{
    if constexpr (c == Contribution::STIFFNESS || c == Contribution::DAMPING)
    {
        return this->m_forceFields;
    }
    else if constexpr (c == Contribution::MASS)
    {
        return this->m_masses;
    }
    else if constexpr (c == Contribution::GEOMETRIC_STIFFNESS)
    {
        return this->m_mechanicalMappings;
    }
}

template <class TMatrix, class TVector>
const MappingGraph& MatrixLinearSystem<TMatrix, TVector>::getMappingGraph() const
{
    return m_mappingGraph;
}

template <class TMatrix, class TVector>
template <core::matrixaccumulator::Contribution c>
void MatrixLinearSystem<TMatrix, TVector>::associateLocalMatrixTo(
    sofa::core::matrixaccumulator::get_component_type<c>* component,
    const core::MechanicalParams* mparams)
{
    const sofa::type::vector<core::behavior::BaseMechanicalState*> mstates =
        retrieveAssociatedMechanicalState(component);
    if (mstates.empty())
    {
        msg_error() << "The component " << component->getPathName() << " is not associated to any mechanical state";
        return;
    }

    // generate all possible pairs of mechanical states
    const auto mstatesPairs = generatePairs(mstates);

    LocalMatrixMaps<c, Real>& matrixMaps = getLocalMatrixMap<c>();

    // The factor that will be applied to all contributions from this component
    const auto factor = Inherit1::template getContributionFactor<c>(mparams, component);

    // index checking strategy
    auto& strategy = matrixMaps.indexVerificationStrategy[component];
    if (d_checkIndices.getValue() && !strategy)
    {
        strategy = std::make_shared<core::matrixaccumulator::RangeVerification>();
        strategy->m_messageComponent = component;
    }



    auto& componentLocalMatrix = matrixMaps.componentLocalMatrix[component];
    for (const auto& pairs : mstatesPairs)
    {
        auto* mstate0 = pairs[0];
        auto* mstate1 = pairs[1];

        const bool isMapped0 = this->getMappingGraph().hasAnyMappingInput(mstate0);
        const bool isMapped1 = this->getMappingGraph().hasAnyMappingInput(mstate1);
        const bool isAnyMapped = isMapped0 || isMapped1;

        auto it = componentLocalMatrix.find(pairs);
        if (it == componentLocalMatrix.end())
        {
            BaseAssemblingMatrixAccumulator<c>* mat { nullptr };
            if (isAnyMapped) //is component mapped?
            {
                AssemblingMappedMatrixAccumulator<c, Real>* mappedMatrix = createLocalMappedMatrixT<c>(component, factor);

                matrixMaps.mappedLocalMatrix[component][pairs] = mappedMatrix;
                mat = mappedMatrix;
            }
            else
            {
                mat = createLocalMatrixT<c>(component, factor);
            }

            msg_info() << "No local matrix found: a new local matrix of type "
                << mat->getClassName() << " (template " << mat->getTemplateName()
                << ") is created and associated to " << component->getPathName()
                << " for a contribution on states " << mstate0->getPathName()
                << " and " << mstate1->getPathName();

            auto insertResult = componentLocalMatrix.insert({pairs, mat});
            it = insertResult.first;

            if constexpr (c == Contribution::STIFFNESS)
            {
                m_stiffness[component].setMatrixAccumulator(mat, mstate0, mstate1);
                m_stiffness[component].setMechanicalParams(mparams);
            }
            else if constexpr (c == Contribution::DAMPING)
            {
                m_damping[component].setMatrixAccumulator(mat, mstate0, mstate1);
                m_damping[component].setMechanicalParams(mparams);
            }
            else if constexpr (c == Contribution::GEOMETRIC_STIFFNESS)
            {
                m_geometricStiffness[component].setMatrixAccumulator(mat, mstate0, mstate1);
                m_geometricStiffness[component].setMechanicalParams(mparams);
            }
            else if constexpr (c == Contribution::MASS)
            {
                m_mass[component] = mat;
            }
        }

        BaseAssemblingMatrixAccumulator<c>* localMatrix = it->second;
        if (!localMatrix)
        {
            dmsg_fatal() << "Local matrix is invalid";
        }

        const auto matrixSize1 = mstate0->getMatrixSize();
        const auto matrixSize2 = mstate1->getMatrixSize();
        if (!isAnyMapped) // mapped components don't add their contributions directly into the global matrix
        {
            localMatrix->setGlobalMatrix(this->getSystemMatrix());

            const auto position = this->m_mappingGraph.getPositionInGlobalMatrix(mstate0, mstate1);
            localMatrix->setPositionInGlobalMatrix(position);
        }
        localMatrix->setMatrixSize({matrixSize1, matrixSize2});
        if (strategy)
        {
            strategy->maxRowIndex = matrixSize1;
            strategy->maxColIndex = matrixSize2 - 1;
        }
    }

}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::makeCreateDispatcher()
{
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::STIFFNESS          >>>(m_createDispatcher) = makeCreateDispatcher<Contribution::STIFFNESS          >();
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::MASS               >>>(m_createDispatcher) = makeCreateDispatcher<Contribution::MASS               >();
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::DAMPING            >>>(m_createDispatcher) = makeCreateDispatcher<Contribution::DAMPING            >();
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::GEOMETRIC_STIFFNESS>>>(m_createDispatcher) = makeCreateDispatcher<Contribution::GEOMETRIC_STIFFNESS>();
}

template <class TMatrix, class TVector>
template <Contribution c>
std::unique_ptr<CreateMatrixDispatcher<c>> MatrixLinearSystem<TMatrix, TVector>
::makeCreateDispatcher()
{
    struct MyCreateMatrixDispatcher : CreateMatrixDispatcher<c>
    {
        typename BaseAssemblingMatrixAccumulator<c>::SPtr
        createLocalMappedMatrix() override
        {
            return sofa::core::objectmodel::New<AssemblingMappedMatrixAccumulator<c, Real>>();
        }

    protected:

        typename BaseAssemblingMatrixAccumulator<c>::SPtr
        createLocalMatrix() const override
        {
            return sofa::core::objectmodel::New<AssemblingMatrixAccumulator<c>>();
        }

        typename BaseAssemblingMatrixAccumulator<c>::SPtr
        createLocalMatrixWithIndexChecking() const override
        {
            return sofa::core::objectmodel::New<AssemblingMatrixAccumulator<c, core::matrixaccumulator::RangeVerification>>();
        }
    };

    return std::make_unique<MyCreateMatrixDispatcher>();
}

/**
 * Generic function to configure a local matrix and associate it to a component
 */
template <core::matrixaccumulator::Contribution c>
void configureCreatedMatrixComponent(typename BaseAssemblingMatrixAccumulator<c>::SPtr mat,
    typename BaseAssemblingMatrixAccumulator<c>::ComponentType* object, const SReal factor, bool printLog)
{
    constexpr std::string_view contribution = core::matrixaccumulator::GetContributionName<c>();
    mat->setName(std::string(contribution) + "_matrix");
    mat->f_printLog.setValue(printLog);
    mat->setFactor(factor);
    mat->associateObject(object);
    mat->addTag(core::objectmodel::Tag(core::behavior::tagSetupByMatrixLinearSystem));
    object->addSlave(mat);
}

template <class TMatrix, class TVector>
template <core::matrixaccumulator::Contribution c>
BaseAssemblingMatrixAccumulator<c>* MatrixLinearSystem<TMatrix, TVector>::createLocalMatrixT(
    sofa::core::matrixaccumulator::get_component_type<c>* object, SReal factor)
{
    this->makeCreateDispatcher();
    auto& dispatcher = std::get<std::unique_ptr<CreateMatrixDispatcher<c>>>(m_createDispatcher);
    typename BaseAssemblingMatrixAccumulator<c>::SPtr localMatrix = dispatcher->createLocalMatrix(d_checkIndices.getValue());
    configureCreatedMatrixComponent<c>(localMatrix, object, factor, !this->notMuted());

    if (d_checkIndices.getValue())
    {
        if (auto concreteLocalMatrix
            = dynamic_cast<AssemblingMatrixAccumulator<c, core::matrixaccumulator::RangeVerification>*>(localMatrix.get()))
        {
            const auto it = getLocalMatrixMap<c>().indexVerificationStrategy.find(object);
            if (it != getLocalMatrixMap<c>().indexVerificationStrategy.end())
            {
                concreteLocalMatrix->indexVerificationStrategy = it->second;
            }
        }
    }

    return localMatrix.get();
}

template <class TMatrix, class TVector>
template <core::matrixaccumulator::Contribution c>
AssemblingMappedMatrixAccumulator<c, typename MatrixLinearSystem<TMatrix, TVector>::Real>*
MatrixLinearSystem<TMatrix, TVector>::createLocalMappedMatrixT(
    sofa::core::matrixaccumulator::get_component_type<c>* object, SReal factor)
{
    this->makeCreateDispatcher();
    auto& dispatcher = std::get<std::unique_ptr<CreateMatrixDispatcher<c>>>(m_createDispatcher);
    typename BaseAssemblingMatrixAccumulator<c>::SPtr m = dispatcher->createLocalMappedMatrix();
    configureCreatedMatrixComponent<c>(m, object, factor, !this->notMuted());
    return dynamic_cast<AssemblingMappedMatrixAccumulator<c, Real>*>(m.get());
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::projectMappedMatrices(const core::MechanicalParams* mparams)
{
    auto cparams = core::ConstraintParams(*mparams);

    for (const auto& [pair, mappedMatrix] : m_localMappedMatrices)
    {
        if (!mappedMatrix)
        {
            msg_error() << "Mapped matrix not created properly";
            continue;
        }

        LocalMappedMatrixType<Real>* crs = mappedMatrix.get();

        crs->compress();
        if (crs->colsValue.empty())
        {
            continue;
        }

        const MappingJacobians<JacobianMatrixType> J0 = computeJacobiansFrom(pair[0], mparams, crs);
        const MappingJacobians<JacobianMatrixType> J1 =
                            (pair[0] == pair[1]) ? J0 : computeJacobiansFrom(pair[1], mparams, crs);

        const sofa::type::fixed_array<MappingJacobians<JacobianMatrixType>, 2> mappingMatricesMap { J0, J1 };

        sofa::component::linearsystem::addMappedMatrixToGlobalMatrixEigen(
            pair, crs, mappingMatricesMap, m_mappingGraph, this->getSystemMatrix());
    }
}

template <class TMatrix, class TVector>
std::vector<unsigned int> MatrixLinearSystem<TMatrix, TVector>::identifyAffectedDoFs(BaseMechanicalState* mstate, LocalMappedMatrixType<Real>* crs)
{
    const auto blockSize = mstate->getMatrixBlockSize();
    std::set<unsigned int> setAffectedDoFs;

    for (std::size_t it_rows_k = 0; it_rows_k < crs->rowIndex.size(); it_rows_k++)
    {
        const auto row = crs->rowIndex[it_rows_k];
        {
            const sofa::SignedIndex dofId = row / blockSize;
            setAffectedDoFs.insert(dofId);
        }
        typename LocalMappedMatrixType<Real>::Range rowRange(crs->rowBegin[it_rows_k], crs->rowBegin[it_rows_k + 1]);
        for (auto xj = rowRange.begin(); xj < rowRange.end(); ++xj) // for each non-null block
        {
            const sofa::SignedIndex col = crs->colsIndex[xj];
            const sofa::SignedIndex dofId = col / blockSize;
            setAffectedDoFs.insert(dofId);
        }
    }

    return std::vector( setAffectedDoFs.begin(), setAffectedDoFs.end() );
}

template <class TMatrix, class TVector>
auto MatrixLinearSystem<TMatrix, TVector>::computeJacobiansFrom(BaseMechanicalState* mstate, const core::MechanicalParams* mparams, LocalMappedMatrixType<Real>* crs)
-> MappingJacobians<JacobianMatrixType>
{
    auto cparams = core::ConstraintParams(*mparams);

    MappingJacobians<JacobianMatrixType> jacobians(*mstate);

    if (!m_mappingGraph.hasAnyMappingInput(mstate))
    {
        return jacobians;
    }

    MechanicalResetConstraintVisitor(&cparams).execute(this->getSolveContext());

    auto mappingJacobianId = sofa::core::MatrixDerivId::mappingJacobian();

    // optimisation to build only the relevent entries of the jacobian matrices
    // The relevent entries are the ones that have a influence on the result
    // of the product J^T * K * J.
    // J does not need to be fully computed if K is sparse.
    {
        const std::vector<unsigned> listAffectedDoFs = identifyAffectedDoFs(mstate, crs);

        if (listAffectedDoFs.empty())
        {
            return jacobians;
        }
        mstate->buildIdentityBlocksInJacobian(listAffectedDoFs, mappingJacobianId);
    }

    // apply the mappings from the bottom to the top, so it builds the jacobian
    // matrices, transforming the space from the input DoFs to the space of the
    // top most DoFs
    const auto parentMappings = getMappingGraph().getBottomUpMappingsFrom(mstate);
    for (auto* mapping : parentMappings)
    {
        mapping->applyJT(&cparams, mappingJacobianId, mappingJacobianId);
    }

    // copy the jacobian matrix stored in the mechanical states into a local
    // matrix data structure
    const auto inputs = m_mappingGraph.getTopMostMechanicalStates(mstate);
    for (auto* input : inputs)
    {
        auto J = std::make_shared<LocalMappedMatrixType<Real> >();
        jacobians.addJacobianToTopMostParent(J, input);
        J->resize(mstate->getMatrixSize(), input->getMatrixSize());
        unsigned int offset {};
        input->copyToBaseMatrix(J.get(), mappingJacobianId, offset);
        J->fullRows();
    }

    return jacobians;
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::assembleMappedMatrices(const core::MechanicalParams* mparams)
{
    if (this->getSystemMatrix()->rowSize() == 0 || this->getSystemMatrix()->colSize() == 0)
    {
        msg_error() << "Global system matrix is not resized appropriatly";
        return;
    }

    SCOPED_TIMER_VARNAME(buildMappedMatricesTimer, "projectMappedMatrices");
    projectMappedMatrices(mparams);
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::applyProjectiveConstraints(const core::MechanicalParams* mparams)
{
    SOFA_UNUSED(mparams);
    SCOPED_TIMER_VARNAME(applyProjectiveConstraintTimer, "applyProjectiveConstraint");
    for (auto* constraint : this->m_projectiveConstraints)
    {
        if (constraint)
        {
            const auto& mstates = constraint->getMechanicalStates();
            if (!mstates.empty())
            {
                m_discarder.m_offset = this->getMappingGraph().getPositionInGlobalMatrix(mstates.front());
                constraint->applyConstraint(&m_discarder);
            }
        }
    }
}

template <class TMatrix, class TVector>
void MatrixLinearSystem<TMatrix, TVector>::Dirichlet::discardRowCol(sofa::Index row, sofa::Index col)
{
    if (row == col && m_offset[0] == m_offset[1])
    {
        m_globalMatrix->clearRowCol(row + m_offset[0]);
    }
    else
    {
        m_globalMatrix->clearRow(row + m_offset[0]);
        m_globalMatrix->clearCol(col + m_offset[1]);
    }
    m_globalMatrix->set(row + m_offset[0], col  + m_offset[1], 1.);
}

}
