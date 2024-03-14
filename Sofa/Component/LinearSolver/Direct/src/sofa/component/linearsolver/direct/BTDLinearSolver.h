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
#include <sofa/component/linearsolver/direct/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/BTDMatrix.h>
#include <sofa/linearalgebra/BlockVector.h>
#include <cmath>
#include <sofa/type/Mat.h>
#include <sofa/component/linearsolver/direct/MatrixLinearSystem[BTDMatrix].h>
#include <sofa/component/linearsolver/direct/TypedMatrixLinearSystem[BTDMatrix].h>

namespace sofa::component::linearsolver::direct
{
/// Linear system solver using Thomas Algorithm for Block Tridiagonal matrices
///
/// References:
/// Conte, S.D., and deBoor, C. (1972). Elementary Numerical Analysis. McGraw-Hill, New York
/// http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
/// http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
/// http://www4.ncsu.edu/eos/users/w/white/www/white/ma580/chap2.5.PDF
template<class Matrix, class Vector>
class BTDLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<Matrix,Vector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BTDLinearSolver, Matrix, Vector), SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, Matrix, Vector));

    Data<bool> d_verbose; ///< Dump system state at each iteration
    Data<bool> d_problem; ///< display debug informations about subpartSolve computation
    Data<bool> d_subpartSolve; ///< Allows for the computation of a subpart of the system
    Data<bool> d_verification; ///< verification of the subpartSolve

    SOFA_ATTRIBUTE_DISABLED__BTDLINEARSOLVER_DATABLOCKSIZE("d_blockSize has been deleted, as it was never actually used.")
    DeprecatedAndRemoved d_blockSize;

    typedef typename Vector::SubVectorType SubVector;
    typedef typename Matrix::SubMatrixType SubMatrix;
    typedef typename Vector::Real Real;
    typedef typename Matrix::BlockType BlocType;
    typedef typename sofa::SignedIndex Index;
    typedef std::list<Index> ListIndex;
    typedef std::pair<Index,Index> IndexPair;
    typedef std::map<IndexPair, SubMatrix> MysparseM;
    typedef typename std::map<IndexPair, SubMatrix>::iterator MysparseMit;

    //type::vector<SubMatrix> alpha;
    type::vector<SubMatrix> alpha_inv;
    type::vector<SubMatrix> lambda;
    type::vector<SubMatrix> B;
    typename Matrix::InvMatrixType Minv;  //inverse matrix

    //////////////////////////// for subpartSolve
    MysparseM H; // force transfer
    MysparseMit H_it;
    Vector bwdContributionOnLH;  //
    Vector fwdContributionOnRH;

    Vector _rh_buf;		 //				// buf the right hand term
    //Vector _df_buf;		 //
    SubVector _acc_rh_bloc;		// accumulation of rh through the browsing of the structure
    SubVector _acc_lh_bloc;		// accumulation of lh through the browsing of the strucutre
    Index	current_bloc, first_block;
    std::vector<SubVector> Vec_dRH;			// buf the dRH on block that are not current_bloc...
    ////////////////////////////

    type::vector<Index> nBlockComputedMinv;
    Vector Y;
protected:
    BTDLinearSolver()
        : d_verbose( initData(&d_verbose,false,"verbose","Dump system state at each iteration") )
        , d_problem(initData(&d_problem, false,"showProblem", "display debug informations about subpartSolve computation") )
        , d_subpartSolve(initData(&d_subpartSolve, false,"subpartSolve", "Allows for the computation of a subpart of the system") )
        , d_verification(initData(&d_verification, false,"verification", "verification of the subpartSolve"))
    {

    }
public:
    void my_identity(SubMatrix& Id, const Index size_id);

    void invert(SubMatrix& Inv, const BlocType& m);

    void invert(Matrix& M) override;

    void computeMinvBlock(Index i, Index j);

    double getMinvElement(Index i, Index j);

    /// Solve Mx=b
    void solve (Matrix& /*M*/, Vector& x, Vector& b) override;

    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    bool addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) override;

    /// Init the partial solve
    void init_partial_solve() override;

    using MatrixLinearSolver<Matrix,Vector>::partial_solve;
    /// partial solve :
    /// b is accumulated
    /// db is a sparse vector that is added to b
    /// partial_x is a sparse vector (with sparse map given) that provide the result of M x = b+db
    /// Solve Mx=b
    //void partial_solve_old(ListIndex&  Iout, ListIndex&  Iin , bool NewIn);
    void partial_solve(ListIndex&  Iout, ListIndex&  Iin , bool NewIn) override;

    void init_partial_inverse(const Index &nb, const Index &bsize);

    template<class RMatrix, class JMatrix>
    bool addJMInvJt(RMatrix& result, JMatrix& J, double fact);


    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override
    {
        Inherit1::parse(arg);

        if (arg->getAttribute("blockSize"))
        {
            msg_deprecated() << "The attribute 'blockSize' is deprecated since SOFA 23.06." << msgendl
                << "This data was not take into account to get the block size, as it can be deduced automatically from the Matrix template parameter.";
        }
    }

private:


    Index _indMaxNonNullForce; // point with non null force which index is the greatest and for which globalAccumulate was not proceed

    Index _indMaxFwdLHComputed;  // indice of node from which bwdLH is accurate

    /// private functions for partial solve
    /// step1=> accumulate RH locally for the InBloc (only if a new force is detected on RH)
    void bwdAccumulateRHinBloc(Index indMaxBloc);   // indMaxBloc should be equal to _indMaxNonNullForce


    /// step2=> accumulate LH globally to step down the value of current_bloc to 0
    void bwdAccumulateLHGlobal( );


    /// step3=> accumulate RH globally to step up the value of current_bloc to the smallest value needed in OutBloc
    void fwdAccumulateRHGlobal(Index indMinBloc);


    /// step4=> compute solution for the indices in the bloc
    /// (and accumulate the potential local dRH (set in Vec_dRH) [set in step1] that have not been yet taken into account by the global bwd and fwd
    void fwdComputeLHinBloc(Index indMaxBloc);

};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API BTDLinearSolver< linearalgebra::BTDMatrix<6, SReal>, linearalgebra::BlockVector<6, SReal> >;
#endif
} //namespace sofa::component::linearsolver::direct
