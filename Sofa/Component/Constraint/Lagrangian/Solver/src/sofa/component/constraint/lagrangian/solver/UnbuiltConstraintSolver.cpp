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

#include <sofa/component/constraint/lagrangian/solver/UnbuiltConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/UnbuiltConstraintProblem.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>

#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::constraint::lagrangian::solver
{

UnbuiltConstraintSolver::UnbuiltConstraintSolver()
: GenericConstraintSolver()
, d_initialGuess(initData(&d_initialGuess, true, "initialGuess", "Activate constraint force history to improve convergence (hot start)"))
{
    
}

void UnbuiltConstraintSolver::doBuildSystem( const core::ConstraintParams *cParams, GenericConstraintProblem * problem, unsigned int numConstraints)
{
    SOFA_UNUSED(cParams);
    UnbuiltConstraintProblem* c_current_cp = dynamic_cast<UnbuiltConstraintProblem*>(problem);
    if (c_current_cp == nullptr)
    {
        msg_error()<<"Constraint problem must derive from UnbuiltConstraintProblem";
        return;
    }

    // Initialize constraint sequence ONCE before iterating over constraint corrections
    c_current_cp->constraints_sequence.resize(numConstraints);
    std::iota(c_current_cp->constraints_sequence.begin(), c_current_cp->constraints_sequence.end(), 0);

    for (const auto& cc : l_constraintCorrections)
    {
        if (!cc->isActive()) continue;

        // some constraint corrections (e.g LinearSolverConstraintCorrection)
        // can change the order of the constraints, to optimize later computations
        cc->resetForUnbuiltResolution(c_current_cp->getF(), c_current_cp->constraints_sequence);
    }

    sofa::linearalgebra::SparseMatrix<SReal>* localWdiag = &c_current_cp->Wdiag;
    localWdiag->resize(numConstraints, numConstraints);

    // for each contact, the constraint corrections that are involved with the contact are memorized
    c_current_cp->cclist_elems.clear();
    c_current_cp->cclist_elems.resize(numConstraints);
    const int nbCC = l_constraintCorrections.size();
    for (unsigned int i = 0; i < numConstraints; i++)
        c_current_cp->cclist_elems[i].resize(nbCC, nullptr);

    unsigned int nbObjects = 0;
    for (unsigned int c_id = 0; c_id < numConstraints;)
    {
        bool foundCC = false;
        nbObjects++;
        const unsigned int l = c_current_cp->constraintsResolutions[c_id]->getNbLines();

        for (unsigned int j = 0; j < l_constraintCorrections.size(); j++)
        {
            core::behavior::BaseConstraintCorrection* cc = l_constraintCorrections[j];
            if (!cc->isActive()) continue;
            if (cc->hasConstraintNumber(c_id))
            {
                c_current_cp->cclist_elems[c_id][j] = cc;
                cc->getBlockDiagonalCompliance(localWdiag, c_id, c_id + l - 1);
                foundCC = true;
            }
        }

        msg_error_when(!foundCC) << "No constraintCorrection found for constraint" << c_id ;

        SReal** w =  c_current_cp->getW();
        for(unsigned int m = c_id; m < c_id + l; m++)
            for(unsigned int n = c_id; n < c_id + l; n++)
                w[m][n] = localWdiag->element(m, n);

        c_id += l;
    }

    addRegularization(c_current_cp->W, d_regularizationTerm.getValue());
    addRegularization(c_current_cp->Wdiag, d_regularizationTerm.getValue());

}

void UnbuiltConstraintSolver::doPreApplyCorrection()
{
    // Save forces for hot-start in next timestep
    keepContactForcesValue();
}
void UnbuiltConstraintSolver::doPreClearCorrection(const core::ConstraintParams* cparams)
{
    getConstraintInfo(cparams);
}

void UnbuiltConstraintSolver::doPostClearCorrection()
{
    computeInitialGuess();
}

void UnbuiltConstraintSolver::initializeConstraintProblems()
{
    for (unsigned i=0; i< CP_BUFFER_SIZE; ++i)
    {
        m_cpBuffer[i] = std::make_unique<UnbuiltConstraintProblem>(this);
    }
    current_cp = m_cpBuffer[0].get();
}

void UnbuiltConstraintSolver::getConstraintInfo(const core::ConstraintParams* cparams)
{
    if (d_initialGuess.getValue() && (m_numConstraints != 0))
    {
        SCOPED_TIMER("GetConstraintInfo");
        m_constraintBlockInfo.clear();
        m_constraintIds.clear();
        simulation::mechanicalvisitor::MechanicalGetConstraintInfoVisitor(cparams, m_constraintBlockInfo, m_constraintIds).execute(getContext());
    }
}

void UnbuiltConstraintSolver::computeInitialGuess()
{
    if (!d_initialGuess.getValue() || m_numConstraints == 0)
        return;

    SCOPED_TIMER("InitialGuess");

    SReal* force = current_cp->getF();
    const int numConstraints = current_cp->getDimension();

    // First, zero all forces
    for (int c = 0; c < numConstraints; c++)
    {
        force[c] = 0.0;
    }

    // Then restore forces from previous timestep for matching persistent IDs
    for (const ConstraintBlockInfo& info : m_constraintBlockInfo)
    {
        if (!info.parent) continue;
        if (!info.hasId) continue;

        auto previt = m_previousConstraints.find(info.parent);
        if (previt == m_previousConstraints.end()) continue;

        const ConstraintBlockBuf& buf = previt->second;
        const int c0 = info.const0;
        const int nbl = (info.nbLines < buf.nbLines) ? info.nbLines : buf.nbLines;

        for (int c = 0; c < info.nbGroups; ++c)
        {
            auto it = buf.persistentToConstraintIdMap.find(m_constraintIds[info.offsetId + c]);
            if (it == buf.persistentToConstraintIdMap.end()) continue;

            const int prevIndex = it->second;
            if (prevIndex >= 0 && prevIndex + nbl <= static_cast<int>(m_previousForces.size()))
            {
                for (int l = 0; l < nbl; ++l)
                {
                    force[c0 + c * nbl + l] = m_previousForces[prevIndex + l];
                }
            }
        }
    }
}

void UnbuiltConstraintSolver::keepContactForcesValue()
{
    if (!d_initialGuess.getValue())
        return;

    SCOPED_TIMER("KeepForces");

    const SReal* force = current_cp->getF();
    const unsigned int numConstraints = current_cp->getDimension();

    // Store current forces
    m_previousForces.resize(numConstraints);
    for (unsigned int c = 0; c < numConstraints; ++c)
    {
        m_previousForces[c] = force[c];
    }

    // Clear previous history (mark all as invalid)
    for (auto& previousConstraint : m_previousConstraints)
    {
        ConstraintBlockBuf& buf = previousConstraint.second;
        for (auto& it2 : buf.persistentToConstraintIdMap)
        {
            it2.second = -1;
        }
    }

    // Fill info from current constraint IDs
    for (const ConstraintBlockInfo& info : m_constraintBlockInfo)
    {
        if (!info.parent) continue;
        if (!info.hasId) continue;

        ConstraintBlockBuf& buf = m_previousConstraints[info.parent];
        buf.nbLines = info.nbLines;

        for (int c = 0; c < info.nbGroups; ++c)
        {
            buf.persistentToConstraintIdMap[m_constraintIds[info.offsetId + c]] = info.const0 + c * info.nbLines;
        }
    }

    // Update constraint count for next iteration
    m_numConstraints = numConstraints;
}




}
