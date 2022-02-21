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
#include <SofaBaseLinearSolver/config.h>

#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/helper/map.h>

namespace sofa::component::linearsolver
{

/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class CGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CGLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<unsigned> d_maxIter; ///< maximum number of iterations of the Conjugate Gradient solution
    Data<SReal> d_tolerance; ///< desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)
    Data<SReal> d_smallDenominatorThreshold; ///< minimum value of the denominator in the conjugate Gradient solution
    Data<bool> d_warmStart; ///< Use previous solution as initial solution
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graph; ///< Graph of residuals at each iteration

protected:

    CGLinearSolver();

    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: p = p*beta + r
    inline void cgstep_beta(const core::ExecParams* params, Vector& p, Vector& r, SReal beta);
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: x += p*alpha, r -= q*alpha
    inline void cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, SReal alpha);

    int timeStepCount{0};
    bool equilibriumReached{false};

public:
    void init() override;
    void reinit() override {};

    void resetSystem() override;

    void setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams) override;

    /// Solve iteratively the linear system Ax=b following a conjugate gradient descent
    void solve (Matrix& A, Vector& x, Vector& b) override;

    //Temporary function to warn the user when old attribute names are used until v21.12
    void parse( sofa::core::objectmodel::BaseObjectDescription* arg ) override
    {
        Inherit::parse(arg);

        if (arg->getAttribute("verbose"))
        {
            msg_warning() << "input data 'verbose' changed for 'printLog', please update your scene (see PR#2098)";
        }
    }

    SOFA_ATTRIBUTE_DISABLED__CGLINEARSOLVER_DATANAME("To fix your code, use d_maxIter")
    DeprecatedAndRemoved f_maxIter;
    SOFA_ATTRIBUTE_DISABLED__CGLINEARSOLVER_DATANAME("To fix your code, use d_tolerance")
    DeprecatedAndRemoved f_tolerance;
    SOFA_ATTRIBUTE_DISABLED__CGLINEARSOLVER_DATANAME("To fix your code, use d_smallDenominatorThreshold")
    DeprecatedAndRemoved f_smallDenominatorThreshold;
    SOFA_ATTRIBUTE_DISABLED__CGLINEARSOLVER_DATANAME("To fix your code, use d_warmStart")
    DeprecatedAndRemoved f_warmStart;
    SOFA_ATTRIBUTE_DISABLED__CGLINEARSOLVER_DATANAME("To fix your code, use d_graph")
    DeprecatedAndRemoved f_graph;
};

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(const core::ExecParams* /*params*/, Vector& p, Vector& r, SReal beta);

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, SReal alpha);

#if  !defined(SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_CPP)
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::FullMatrix<double>, linearalgebra::FullVector<double> >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::SparseMatrix<double>, linearalgebra::FullVector<double> >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<double>, linearalgebra::FullVector<double> >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<2,2,double> >, linearalgebra::FullVector<double> >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,double> >, linearalgebra::FullVector<double> >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<4,4,double> >, linearalgebra::FullVector<double> >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<6,6,double> >, linearalgebra::FullVector<double> >;
extern template class SOFA_SOFABASELINEARSOLVER_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<8,8,double> >, linearalgebra::FullVector<double> >;


#endif

} // namespace sofa::component::linearsolver
