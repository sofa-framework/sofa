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

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/component/linearsolver/direct/SparseCommon.h>
#include <sofa/component/linearsolver/direct/SparseLDLBlockOperations.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/linearalgebra/DiagonalSystemSolver.h>
#include <sofa/linearalgebra/TriangularSystemSolver.h>
#include <sofa/component/linearsolver/ordering/OrderingMethodAccessor.h>


namespace sofa::component::linearsolver::direct
{

namespace details
{
/// Type of the blocks of a matrix type. A scalar if the matrix type does not define blocks.
template<class TMatrix>
struct SparseLDLMatrixBlock { using type = typename TMatrix::Real; };

template<class TMatrix> requires requires { typename TMatrix::Block; }
struct SparseLDLMatrixBlock<TMatrix> { using type = typename TMatrix::Block; };
}

/**
 * Default structure for a LDL factorization.
 *
 * The factorized matrix is made of square blocks of type VecBlock::value_type: either a scalar,
 * or a dense NxN matrix (e.g. type::Mat<3,3,SReal>). All the indices (sizes, permutation,
 * patterns) are block indices. In the scalar case, a block index is a scalar index.
 */
template<class VecInt,class VecBlock>
class SparseLDLImplInvertData : public MatrixInvertData {
public :
    ~SparseLDLImplInvertData() override = default;

    /// number of block rows of the factorized matrix
    int n { 0 };

    /// number of scalar rows of the linear system. It can be smaller than n * blockSize if the
    /// last block is incomplete (in that case, the unused rows are padded with identity)
    int systemSize { 0 };

    int P_nnz { 0 }, L_nnz { 0 };

    //CSR pattern of the matrix to invert
    VecInt P_rowind, P_colptr;

    //CSC matrix L, the lower unitriangular matrix of the LDL decomposition.
    //Equivalently, L^T in CSR format: after the factorization, L_values contains the blocks of
    //L^T (i.e. the transposed blocks of L), so that L^T can be used directly in a CSR solver.
    VecInt L_rowind, L_colptr;
    VecBlock L_values;

    //CSC matrix L^T, the transpose of the lower unitriangular matrix of the LDL decomposition.
    //Equivalently, L in CSR format: LT_values contains the blocks of L.
    VecInt LT_rowind, LT_colptr;
    VecBlock LT_values;

    //permutation
    VecInt perm, invperm;

    //diagonal blocks of D^-1
    VecBlock invD;

    type::vector<int> Parent;
    bool new_factorization_needed { true };
};

inline void CSPARSE_symbolic (int n,int * M_colptr,int * M_rowind,int * colptr,int * perm,int * invperm,int * Parent, int * Flag, int * Lnz)
{
    for (int k = 0 ; k < n ; k++)
    {
        Parent [k] = -1 ;	    // parent of k is not yet known 
        Flag [k] = k ;		    // mark node k as visited 
        Lnz [k] = 0 ;		    // count of nonzeros in column k of L 
        const int kk = perm[k];  // kth original, or permuted, column 
        for (int p = M_colptr[kk] ; p < M_colptr[kk+1] ; p++)
        {
            // A (i,k) is nonzero (original or permuted A) 
            int i = invperm[M_rowind[p]];
            if (i < k)
            {
                // follow path from i to root of etree, stop at flagged node 
                for ( ; Flag [i] != k ; i = Parent [i])
                {
                    // find parent of i if not yet determined 
                    if (Parent [i] == -1) Parent [i] = k ;
                    Lnz [i]++ ;				// L (k,i) is nonzero 
                    Flag [i] = k ;			// mark i as visited 
                }
            }
        }
    }

    colptr[0] = 0 ;
    for (int k = 0 ; k < n ; k++) colptr[k+1] = colptr[k] + Lnz[k] ;
}

/**
 * Numeric factorization (up-looking LDL^T), for a matrix made of square blocks (or scalars).
 * The matrix M is given in CSR format. Since it is assumed symmetric, row k of M is
 * interpreted as column k.
 * On output, D contains the diagonal factors (see SparseLDLBlockOperations): they must be
 * finalized with SparseLDLBlockOperations::finalizeDiagonal to obtain D^{-1}.
 * Return false if a pivot is singular.
 */
template<class Block>
inline bool CSPARSE_numeric(int n, const int * M_colptr, const int * M_rowind, const Block * M_values,
                            const int * colptr, int * rowind, Block * values, Block * D,
                            const int * perm, const int * invperm, const int * Parent,
                            int * Flag, int * Lnz, int * Pattern, Block * Y)
{
    using Ops = SparseLDLBlockOperations<Block>;

    for (int k = 0 ; k < n ; k++)
    {
        Ops::clear(Y[k]);       // Y(0:k) is now all zero
        int top = n ;           // stack for pattern is empty
        Flag [k] = k ;          // mark node k as visited
        Lnz [k] = 0 ;           // count of nonzeros in column k of L
        const int kk = perm[k]; // kth original, or permuted, column
        for (int p = M_colptr[kk] ; p < M_colptr[kk+1] ; p++)
        {
            const auto r = M_rowind[p] ;
            assert(r < n && r >= 0);
            int i = invperm[r];     // get A(i,k)
            if (i < k)
            {
                // the stored block is A(k,i), and A(i,k) = A(k,i)^T
                Ops::addTransposed(Y[i], M_values[p]); // scatter A(i,k) into Y (sum duplicates)
                int len = 0;
                for ( ; Flag[i] != k ; i = Parent[i])
                {
                    Pattern [len++] = i ;   // L(k,i) is nonzero
                    Flag [i] = k ;          // mark i as visited
                }
                while (len > 0) Pattern[--top] = Pattern [--len] ;
            }
            else if (i == k)
            {
                Ops::add(Y[k], M_values[p]);
            }
        }

        // compute numerical values kth row of L (a sparse triangular solve)
        Block Dk = Y[k];        // get D(k,k) and clear Y(k)
        Ops::clear(Y[k]);
        for ( ; top < n ; top++)
        {
            const int i = Pattern [top] ;   // Pattern [top:n-1] is pattern of L(:,k)
            const Block yi = Y [i] ;        // get and clear Y(i)
            Ops::clear(Y[i]);
            int p = colptr[i];
            // The loop bound is stored in a local variable on purpose: otherwise, the compiler
            // must reload colptr[i] and Lnz[i] at each iteration, since it cannot prove that the
            // writes into Y do not modify them (MSVC does not use type-based alias analysis).
            // Measured on a scalar matrix: ~25% faster numeric factorization.
            const int pend = colptr[i] + Lnz[i];
            for ( ; p < pend ; p++)
            {
                Ops::subtractProduct(Y[rowind[p]], values[p], yi);
            }
            Ops::computeLowerEntry(values[p], yi, D[i]);    // the nonzero entry L(k,i)
            Ops::subtractProduct(Dk, values[p], yi);        // D(k) -= L(k,i) * Y(i)
            rowind[p] = k ;     // store L(k,i) in column form of L
            Lnz[i]++ ;          // increment count of nonzeros in col i
        }

        if (!Ops::setDiagonalFactor(D[k], Dk))
        {
            msg_error("SparseLDLSolver") << "Failed to factorize, D(" << k << "," << k << ") is singular" ;
            return false;
        }
    }
    return true;
}

template<class TMatrix, class TVector, class TThreadManager>
class SparseLDLSolverImpl : public ordering::OrderingMethodAccessor<sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager> >
{
public :
    SOFA_CLASS(
        SOFA_TEMPLATE3(SparseLDLSolverImpl,TMatrix,TVector,TThreadManager),
        SOFA_TEMPLATE(ordering::OrderingMethodAccessor, SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,TThreadManager))
    );
    typedef ordering::OrderingMethodAccessor<sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector, TThreadManager> > Inherit;

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef TThreadManager ThreadManager;
    typedef typename TMatrix::Real Real;

    /// Type of the blocks of the matrix: a scalar, or a square type::Mat
    using Block = typename details::SparseLDLMatrixBlock<TMatrix>::type;
    using BlockOperations = SparseLDLBlockOperations<Block>;
    static constexpr int BlockSize = BlockOperations::N;

protected :

    Data<bool> d_precomputeSymbolicDecomposition; ///< If true the solver will reuse the precomputed symbolic decomposition. Otherwise it will recompute it at each step.
    Data<int> d_L_nnz; ///< Number of non-zero values in the lower triangular matrix of the factorization. The lower, the faster the system is solved.

    SparseLDLSolverImpl()
    : d_precomputeSymbolicDecomposition(initData(&d_precomputeSymbolicDecomposition, true ,"precomputeSymbolicDecomposition", "If true, the solver will reuse the precomputed symbolic decomposition, meaning that it will store the shape of [factor matrix] on the first step, or when its shape changes, and then it will only update its coefficients. When the shape of the matrix changes, a new factorization is computed."
                                                                                                                              "If false, the solver will compute the entire decomposition at each step"))
    , d_L_nnz(initData(&d_L_nnz, 0, "L_nnz", "Number of non-zero values in the lower triangular matrix of the factorization. The lower, the faster the system is solved.", true, true))
    {}

    template<class VecInt,class VecBlock>
    void solve_cpu(Real * x,const Real * b,SparseLDLImplInvertData<VecInt,VecBlock> * data)
    {
        using Ops = SparseLDLBlockOperations<typename VecBlock::value_type>;
        constexpr int N = Ops::N;

        const int n = data->n;
        if (n == 0)
        {
            return;
        }

        const int * perm = data->perm.data();
        const int systemSize = data->systemSize;

        Tmp.clear();
        Tmp.fastResize(n * N);

        // A x = b
        //   <=> (L * D * L^T) * x = b
        //   <=> L y = b  with y = D * L^T * x  # Step 1: compute y from the system L y = b
        //   <=> D z = y  with z = L^T * x      # Step 2: compute z from the system D z = y
        //   <=> L^T * x = z                    # Step 3: compute x from the system L^T x = z

        // b, x, y and z can be read/written in the same vector:
        Real* const t = Tmp.data();

        // apply the (block) permutation to the right-hand side
        for (int i = 0; i < n; ++i)
        {
            const int src = perm[i] * N;
            for (int c = 0; c < N; ++c)
            {
                t[i * N + c] = (src + c < systemSize) ? b[src + c] : Real(0);
            }
        }

        // Step 1: compute y from the system L y = b
        // Note that L^T, stored in CSC, corresponds to L in CSR
        sofa::linearalgebra::solveLowerUnitriangularSystemCSR(n, t, t,
            data->LT_colptr.data(), data->LT_rowind.data(), data->LT_values.data());

        // Step 2: compute z from the system D z = y
        sofa::linearalgebra::solveDiagonalSystemUsingInvertedValues(n, t, t, data->invD.data());

        // Step 3: compute x from the system L^T x = z
        // Note that L, stored in CSC, corresponds to L^T in CSR
        sofa::linearalgebra::solveUpperUnitriangularSystemCSR(n, t, t,
            data->L_colptr.data(), data->L_rowind.data(), data->L_values.data());

        // apply the permutation to the solution
        for (int i = 0; i < n; ++i)
        {
            const int dst = perm[i] * N;
            for (int c = 0; c < N; ++c)
            {
                if (dst + c < systemSize)
                {
                    x[dst + c] = t[i * N + c];
                }
            }
        }
    }

    void LDL_ordering(int n, int nnz, int* M_colptr, int* M_rowind, Real* M_values, int* perm, int* invperm)
    {
        SOFA_UNUSED(M_values);
        core::behavior::BaseOrderingMethod::SparseMatrixPattern pattern;
        pattern.matrixSize = n;
        pattern.numberOfNonZeros = nnz;
        pattern.rowBegin = M_colptr;
        pattern.colsIndex = M_rowind;

        this->l_orderingMethod->computePermutation(pattern, perm, invperm);
    }

    void LDL_symbolic (int n,int * M_colptr,int * M_rowind,int * colptr,int * perm,int * invperm,int * Parent) {
        Lnz.clear();
        Flag.clear();
        Pattern.clear();

        Lnz.resize(n);
        Flag.resize(n);
        Pattern.resize(n);

        CSPARSE_symbolic(n,M_colptr,M_rowind,colptr,perm,invperm,Parent,Flag.data(),Lnz.data());
    }

    bool LDL_numeric(int n,
                     const int* M_colptr, const int* M_rowind, const Block* M_values,
                     const int* colptr, int* rowind, Block* values,
                     Block* D, const int* perm, const int* invperm, const int* Parent)
    {
        Y.resize(n);

        return CSPARSE_numeric<Block>(n,M_colptr,M_rowind,M_values,colptr,rowind,values,D,perm,invperm,Parent,Flag.data(),Lnz.data(),Pattern.data(),Y.data());
    }

    /**
     * Factorization of the matrix M, given in CSR format, with blocks of type Block.
     * n is the number of block rows. systemSize is the number of scalar rows of the linear system
     * (n * BlockSize by default).
     * Return false if the factorization failed.
     */
    template<class VecInt,class VecBlock>
    bool factorize(int n, int * M_colptr, int * M_rowind, const Block * M_values, SparseLDLImplInvertData<VecInt,VecBlock> * data, int systemSize = -1)
    {
        static_assert(std::is_same_v<typename VecBlock::value_type, Block>);

        data->new_factorization_needed =
            data->P_colptr.size() == 0 ||
            data->P_rowind.size() == 0 ||
            compareMatrixShape(n, M_colptr, M_rowind, data->n, (int*)data->P_colptr.data(), (int*)data->P_rowind.data());

        data->n = n;
        data->systemSize = systemSize < 0 ? n * BlockSize : systemSize;
        data->P_nnz = M_colptr[data->n];

        // we test if the matrix has the same struct as previous factorized matrix
        if (data->new_factorization_needed  || !d_precomputeSymbolicDecomposition.getValue() )
        {
            SCOPED_TIMER_VARNAME(factorizationTimer, "symbolic_factorization");
            msg_info() << "Recomputing new factorization" ;

            data->perm.clear();data->perm.fastResize(data->n);
            data->invperm.clear();data->invperm.fastResize(data->n);
            data->invD.clear();data->invD.fastResize(data->n);
            data->P_colptr.clear();data->P_colptr.fastResize(data->n+1);
            data->L_colptr.clear();data->L_colptr.fastResize(data->n+1);
            data->LT_colptr.clear();data->LT_colptr.fastResize(data->n+1);
            data->P_rowind.clear();data->P_rowind.fastResize(data->P_nnz);

            memcpy(data->P_colptr.data(),M_colptr,(data->n+1) * sizeof(int));
            memcpy(data->P_rowind.data(),M_rowind,data->P_nnz * sizeof(int));

            //ordering function (computed on the block pattern)
            LDL_ordering( data->n , data->P_nnz, M_colptr , M_rowind , nullptr, data->perm.data(), data->invperm.data() );

            data->Parent.clear();
            data->Parent.resize(data->n);

            //symbolic factorization
            LDL_symbolic(data->n,M_colptr,M_rowind,data->L_colptr.data(),
                         data->perm.data(),data->invperm.data(),data->Parent.data());

            data->L_nnz = data->L_colptr[data->n];
            d_L_nnz.setValue(data->L_nnz * BlockSize * BlockSize);

            data->L_rowind.clear();data->L_rowind.fastResize(data->L_nnz);
            data->L_values.clear();data->L_values.fastResize(data->L_nnz);
            data->LT_rowind.clear();data->LT_rowind.fastResize(data->L_nnz);
            data->LT_values.clear();data->LT_values.fastResize(data->L_nnz);
        }

        Block * D = data->invD.data();
        int * rowind = data->L_rowind.data();
        int * colptr = data->L_colptr.data();
        Block * values = data->L_values.data();
        int * tran_rowind = data->LT_rowind.data();
        int * tran_colptr = data->LT_colptr.data();
        Block * tran_values = data->LT_values.data();

        //Numeric Factorization
        {
            SCOPED_TIMER_VARNAME(factorizationTimer, "numeric_factorization");
            if (!LDL_numeric(data->n, M_colptr, M_rowind, M_values, colptr, rowind, values, D,
                        data->perm.data(), data->invperm.data(), data->Parent.data()))
            {
                return false;
            }

            //inverse the diagonal
            for (int i = 0; i < data->n; i++)
            {
                BlockOperations::finalizeDiagonal(D[i]);
            }
        }

        if (data->new_factorization_needed  || !d_precomputeSymbolicDecomposition.getValue() )
        {
            //Compute transpose in tran_colptr, tran_rowind, tran_values, tran_D
            tran_countvec.clear();
            tran_countvec.resize(data->n);

            //First we count the number of value on each row.
            for (int j=0;j<data->L_nnz;j++) tran_countvec[rowind[j]]++;

            //Now we make a scan to build tran_colptr
            tran_colptr[0] = 0;
            for (int j=0;j<data->n;j++) tran_colptr[j+1] = tran_colptr[j] + tran_countvec[j];
        }

        //we clear tran_countvec because we use it now to store how many values are written on each line
        tran_countvec.clear();
        tran_countvec.resize(data->n);

        for (int j = 0; j < data->n; j++)
        {
            for (int i = colptr[j]; i < colptr[j + 1]; i++)
            {
                const int line = rowind[i];
                tran_rowind[tran_colptr[line] + tran_countvec[line]] = j;
                tran_values[tran_colptr[line] + tran_countvec[line]] = values[i];
                tran_countvec[line]++;
            }
        }

        // L in CSC is L^T in CSR, provided that its blocks are transposed
        if constexpr (!BlockOperations::IsScalar)
        {
            for (int i = 0; i < data->L_nnz; ++i)
            {
                BlockOperations::transposeInPlace(values[i]);
            }
        }

        return true;
    }

    type::vector<Real> Tmp;
protected : //the following variables are used during the factorization they cannot be used in the main thread !

    type::vector<Block> Y;
    type::vector<int> Lnz,Flag,Pattern;
    type::vector<int> tran_countvec;
};

} // namespace sofa::component::linearsolver::direct
