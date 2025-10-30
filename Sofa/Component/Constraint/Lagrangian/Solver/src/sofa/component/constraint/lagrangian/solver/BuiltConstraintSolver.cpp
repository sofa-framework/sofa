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

#include <sofa/component/constraint/lagrangian/solver/BuiltConstraintSolver.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>

namespace sofa::component::constraint::lagrangian::solver
{
    
BuiltConstraintSolver::BuiltConstraintSolver()
: d_multithreading(initData(&d_multithreading, false, "multithreading", "Build compliances concurrently"))
{}

void BuiltConstraintSolver::init()
{
    Inherit1::init();
    if(d_multithreading.getValue())
    {
        simulation::MainTaskSchedulerFactory::createInRegistry()->init();
    }
}

void BuiltConstraintSolver::doBuildSystem( const core::ConstraintParams *cParams, GenericConstraintProblem * problem ,unsigned int numConstraints)
{
    SOFA_UNUSED(numConstraints);
    SCOPED_TIMER_VARNAME(getComplianceTimer, "Get Compliance");
    dmsg_info() <<" computeCompliance in "  << l_constraintCorrections.size()<< " constraintCorrections" ;

    const bool multithreading = d_multithreading.getValue();

    const simulation::ForEachExecutionPolicy execution = multithreading ?
        simulation::ForEachExecutionPolicy::PARALLEL :
        simulation::ForEachExecutionPolicy::SEQUENTIAL;

    simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler);

    //Used to prevent simultaneous accesses to the main compliance matrix
    std::mutex mutex;

    //Visits all constraint corrections to compute the compliance matrix projected
    //in the constraint space.
    simulation::forEachRange(execution, *taskScheduler,  l_constraintCorrections.begin(),  l_constraintCorrections.end(),
        [&cParams, this, &multithreading, &mutex, problem](const auto& range)
        {
            ComplianceWrapper compliance(problem->W, multithreading);

            for (auto it = range.start; it != range.end; ++it)
            {
                core::behavior::BaseConstraintCorrection* cc = *it;
                if (cc->isActive())
                {
                    cc->addComplianceInConstraintSpace(cParams, &compliance.matrix());
                }
            }

            std::lock_guard guard(mutex);
            compliance.assembleMatrix();
        });

    addRegularization(problem->W,  d_regularizationTerm.getValue());
    dmsg_info() << " computeCompliance_done "  ;
}


BuiltConstraintSolver::ComplianceWrapper::ComplianceMatrixType& BuiltConstraintSolver::ComplianceWrapper::matrix()
{
    if (m_isMultiThreaded)
    {
        if (!m_threadMatrix)
        {
            m_threadMatrix = std::make_unique<ComplianceMatrixType>();
            m_threadMatrix->resize(m_complianceMatrix.rowSize(), m_complianceMatrix.colSize());
        }
        return *m_threadMatrix;
    }
    return m_complianceMatrix;
}

void BuiltConstraintSolver::ComplianceWrapper::assembleMatrix() const
{
    if (m_threadMatrix)
    {
        for (linearalgebra::BaseMatrix::Index j = 0; j < m_threadMatrix->rowSize(); ++j)
        {
            for (linearalgebra::BaseMatrix::Index l = 0; l < m_threadMatrix->colSize(); ++l)
            {
                m_complianceMatrix.add(j, l, m_threadMatrix->element(j,l));
            }
        }
    }
}

}