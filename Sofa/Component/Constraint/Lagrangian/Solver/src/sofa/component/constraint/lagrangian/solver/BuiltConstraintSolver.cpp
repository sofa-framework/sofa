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
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/ParallelForEach.h>
#include <Eigen/Eigenvalues>

namespace sofa::component::constraint::lagrangian::solver
{
    
BuiltConstraintSolver::BuiltConstraintSolver()
: d_multithreading(initData(&d_multithreading, false, "multithreading", "Build compliances concurrently"))
, d_useSVDForRegularization(initData(&d_useSVDForRegularization, false, "useSVDForRegularization", "Use SVD decomposiiton of the compliance matrix to project singular values smaller than regularization to the regularization term. Only works with built"))
, d_svdSingularValueNullSpaceCriteriaFactor(initData(&d_svdSingularValueNullSpaceCriteriaFactor, 0.01, "svdSingularValueNullSpaceCriteriaFactor", "Fraction of the highest singular value bellow which a singular value will be supposed to belong to the nullspace"))
, d_svdSingularVectorNullSpaceCriteriaFactor(initData(&d_svdSingularVectorNullSpaceCriteriaFactor, 0.001, "svdSingularVectorNullSpaceCriteriaFactor", "Absolute value bellow which a component of a normalized base vector will be considered null"))
{}

void BuiltConstraintSolver::init()
{
    Inherit1::init();
    if(d_multithreading.getValue())
    {
        simulation::MainTaskSchedulerFactory::createInRegistry()->init();
    }
}

void BuiltConstraintSolver::addRegularization(linearalgebra::BaseMatrix& W, const SReal regularization)
{

    if (d_useSVDForRegularization.getValue())
    {
        if (regularization>std::numeric_limits<SReal>::epsilon())
        {
            auto * FullW =  dynamic_cast<sofa::linearalgebra::LPtrFullMatrix<SReal> * >(&W);
            if (FullW == nullptr)
            {
                msg_error()<<"BuiltConstraintSolver expect a LPtrFullMatrix for W but didn't receive one. The constraint problem is wrong. Please fix the code or just deactivate SVD regularization.";
                return;
            }
            const size_t problemSize = FullW->rowSize();

            Eigen::Map<Eigen::MatrixX<SReal>> EigenW(FullW->ptr(),problemSize, problemSize) ;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd( EigenW, Eigen::ComputeFullV | Eigen::ComputeFullU );



            //Given the SVD, loop over all singular values, and those that are smaller than 1% of the highest one are considered to be the null space
            std::vector<bool> nullSpaceIndicator(problemSize, false);
            int nullSpaceBegin = -1;
            for(size_t i=0; i<problemSize; i++)
            {
                if (fabs(svd.singularValues()(i)) < fabs(svd.singularValues()(0)) * d_svdSingularValueNullSpaceCriteriaFactor.getValue())
                {
                    nullSpaceBegin = i;
                    break;
                }
            }

            //Now for all vector of the null space basis, we look at the indices where the coefficient
            //is greater than 1% of the norm of the vector, this is the constraints that
            //belong to the null space and thus have other one that are antagonists
            for(int i=nullSpaceBegin; (i != -1) && (i<problemSize); ++i)
            {
                for(size_t j=0; j<problemSize; j++)
                    nullSpaceIndicator[j] = nullSpaceIndicator[j] ||  fabs(svd.matrixV().col(i)(j)) > d_svdSingularVectorNullSpaceCriteriaFactor.getValue();
            }

            if (f_printLog.getValue())
            {
                std::stringstream msg ;
                msg <<"Unregularized diagonal : ";
                for(size_t i=0; i<problemSize; i++)
                    msg<<EigenW(i,i) << "  ";
                msg_info()<< msg.str();
            }

            //Because the eigen matrix uses the buffer of W to store the matrix, this is sufficient to set the value.
            //Now using the indicator vector, set the regularization for the constraints that belongs
            //to the null space to the regularization term
            for(size_t i=0; i<problemSize; i++)
                EigenW(i,i) += nullSpaceIndicator[i]*d_regularizationTerm.getValue();

            if (f_printLog.getValue())
            {
                std::stringstream msg ;
                msg <<"Null space : ";
                for(size_t i=0; i<problemSize; i++)
                    msg<<nullSpaceIndicator[i] << "  ";
                msg_info()<< msg.str();

                msg.flush();
                msg <<"Regularized diagonal : ";
                for(size_t i=0; i<problemSize; i++)
                    msg<<EigenW(i,i) << "  ";
                msg_info()<< msg.str();
            }

        }
    }
    else
    {
        Inherit1::addRegularization(W, regularization);
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
