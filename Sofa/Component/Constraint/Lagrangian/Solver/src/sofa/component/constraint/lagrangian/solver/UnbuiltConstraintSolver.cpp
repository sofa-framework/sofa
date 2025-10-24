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

namespace sofa::component::constraint::lagrangian::solver
{

void UnbuiltConstraintSolver::doBuildSystem( const core::ConstraintParams *cParams, GenericConstraintProblem * problem, unsigned int numConstraints)
{
    SOFA_UNUSED(cParams);
    UnbuiltConstraintProblem* c_current_cp = dynamic_cast<UnbuiltConstraintProblem*>(problem);
    if (c_current_cp == nullptr)
    {
        msg_error()<<"Constraint problem must derive from UnbuiltConstraintProblem";
        return;
    }

    for (const auto& cc : l_constraintCorrections)
    {
        if (!cc->isActive()) continue;

        c_current_cp->constraints_sequence.resize(numConstraints);
        std::iota(c_current_cp->constraints_sequence.begin(), c_current_cp->constraints_sequence.end(), 0);

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

void UnbuiltConstraintSolver::initializeConstraintProblems()
{
    for (unsigned i=0; i< CP_BUFFER_SIZE; ++i)
    {
        m_cpBuffer[i] = new UnbuiltConstraintProblem(this);
    }
    current_cp = m_cpBuffer[0];
}


}