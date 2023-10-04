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
#include <sofa/component/linearsolver/direct/SVDLinearSolver.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace sofa::component::linearsolver::direct
{

using core::VecId;
using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;

template<class TMatrix, class TVector>
SVDLinearSolver<TMatrix,TVector>::SVDLinearSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_minSingularValue( initData(&f_minSingularValue,(Real)1.0e-6,"minSingularValue","Thershold under which a singular value is set to 0, for the stabilization of ill-conditioned system.") )
    , f_conditionNumber( initData(&f_conditionNumber,(Real)0.0,"conditionNumber","Condition number of the matrix: ratio between the largest and smallest singular values. Computed in method solve.") )
{
}

/// Solve Mx=b
template<class TMatrix, class TVector>
void SVDLinearSolver<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("SVD");
#endif

    SCOPED_TIMER_VARNAME(svdSolveTimer, "Solve-SVD");

    const bool verbose  = f_verbose.getValue();

    /// Convert the matrix and the right-hand vector to Eigen objects
    using EigenVectorX = Eigen::Matrix<SReal, Eigen::Dynamic, 1>;
    using EigenMatrixX = Eigen::Matrix<SReal, Eigen::Dynamic, -1>;

    EigenMatrixX m(M.rowSize(),M.colSize());
    EigenVectorX rhs(M.rowSize());
    {
        SCOPED_TIMER_VARNAME(convertTimer, "convertToEigen");
        for(unsigned i=0; i<(unsigned)M.rowSize(); i++ )
        {
            for( unsigned j=0; j<(unsigned)M.colSize(); j++ )
                m(i,j) = M.element(i, j);
            rhs(i) = b[i];
        }
    }

    msg_info_when(verbose) << "solve, Here is the matrix m:  "
                           << m ;

    /// Compute the SVD decomposition and the condition number
    Eigen::JacobiSVD<EigenMatrixX> svd;
    {
        SCOPED_TIMER_VARNAME(svdDecompositionTimer, "SVDDecomposition");
        svd.compute(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
        f_conditionNumber.setValue( (Real)(svd.singularValues()(0) / svd.singularValues()(M.rowSize()-1)) );
    }

    if(verbose)
    {
        msg_info() << "solve, the singular values are:" << msgendl << svd.singularValues()  << msgendl
                   << "Its left singular vectors are the columns of the thin U matrix: " << msgendl
                   << svd.matrixU() << msgendl
                   << "Its right singular vectors are the columns of the thin V matrix:" msgendl
                   << svd.matrixV() ;
    }
    else
    {
        msg_info() << "solve, the singular values are:" << msgendl << svd.singularValues()  << msgendl;
    }

    /// Solve the equation system and copy the solution to the SOFA vector
    {
        SCOPED_TIMER_VARNAME(solveSvdTimer, "solveFromSVD");
        EigenVectorX Ut_b = svd.matrixU().transpose() *  rhs;
        EigenVectorX S_Ut_b(M.colSize());
        for( unsigned i=0; i<(unsigned)M.colSize(); i++ )   /// product with the diagonal matrix, using the threshold for near-null values
        {
            if( svd.singularValues()[i] > f_minSingularValue.getValue() )
                S_Ut_b[i] = Ut_b[i]/svd.singularValues()[i];
            else
                S_Ut_b[i] = (Real)0.0 ;
        }
        EigenVectorX solution = svd.matrixV() * S_Ut_b;
        for(unsigned i=0; i<(unsigned)M.rowSize(); i++ )
        {
            x[i] = (Real) solution(i);
        }

        dmsg_info() << "solve, rhs vector = " << msgendl << rhs.transpose() << msgendl
                << " solution =   \n" << msgendl << x << msgendl
                << " verification, mx - b = " << msgendl << (m * solution - rhs ).transpose() << msgendl;
    }
}

} // namespace sofa::component::linearsolver::direct
