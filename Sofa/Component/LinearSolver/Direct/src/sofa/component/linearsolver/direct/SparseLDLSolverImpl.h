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
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/component/linearsolver/direct/SparseCommon.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/linearalgebra/DiagonalSystemSolver.h>
#include <sofa/linearalgebra/TriangularSystemSolver.h>
#include <Eigen/SparseCore>
#include <Eigen/MetisSupport>
#include <Eigen/OrderingMethods>


namespace sofa::component::linearsolver::direct
{

//defaut structure for a LDL factorization
template<class VecInt,class VecReal>
class SparseLDLImplInvertData : public MatrixInvertData {
public :
    ~SparseLDLImplInvertData() override = default;

    int n, P_nnz, L_nnz;

    //CSR matrix P, a copy of the matrix to invert
    VecInt P_rowind, P_colptr;
    VecReal P_values;

    //CSC matrix L, the lower unitriangular matrix of the LDL decomposition
    VecInt L_rowind, L_colptr;
    VecReal L_values;

    //CSC matrix L^T, the transpose of the lower unitriangular matrix of the LDL decomposition
    VecInt LT_rowind, LT_colptr;
    VecReal LT_values;

    //permutation
    VecInt perm, invperm;

    //diagonal of D^-1
    VecReal invD;

    type::vector<int> Parent;
    bool new_factorization_needed;
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

template<class Real>
inline void CSPARSE_numeric(int n,int * M_colptr,int * M_rowind,Real * M_values,int * colptr,int * rowind,Real * values,Real * D,int * perm,int * invperm,int * Parent, int * Flag, int * Lnz, int * Pattern, Real * Y)
{
    Real yi, l_ki ;
    int i, p, kk, len, top ;

    for (int k = 0 ; k < n ; k++)
    {
        Y [k] = 0.0 ;		    // Y(0:k) is now all zero 
        top = n ;		    // stack for pattern is empty 
        Flag [k] = k ;		    // mark node k as visited 
        Lnz [k] = 0 ;		    // count of nonzeros in column k of L 
        kk = perm[k];  // kth original, or permuted, column 
        for (p = M_colptr[kk] ; p < M_colptr[kk+1] ; p++)
        {
            i = invperm[M_rowind[p]];	// get A(i,k) 
            if (i <= k)
            {
                Y[i] += M_values[p] ;  // scatter A(i,k) into Y (sum duplicates) 
                for (len = 0 ; Flag[i] != k ; i = Parent[i])
                {
                    Pattern [len++] = i ;   // L(k,i) is nonzero 
                    Flag [i] = k ;	    // mark i as visited 
                }
                while (len > 0) Pattern[--top] = Pattern [--len] ;
            }
        }
        // compute numerical values kth row of L (a sparse triangular solve) 
        D[k] = Y [k] ;		    // get D(k,k) and clear Y(k) 
        Y[k] = 0.0 ;
        for ( ; top < n ; top++)
        {
            i = Pattern [top] ;	    // Pattern [top:n-1] is pattern of L(:,k) 
            yi = Y [i] ;	    // get and clear Y(i) 
            Y [i] = 0.0 ;
            for (p = colptr[i] ; p < colptr[i] + Lnz [i] ; p++)
            {
                Y[rowind[p]] -= values[p] * yi ;
            }
            l_ki = yi / D[i] ;	    // the nonzero entry L(k,i) 
            D[k] -= l_ki * yi ;
            rowind[p] = k ;	    // store L(k,i) in column form of L 
            values[p] = l_ki ;
            Lnz[i]++ ;		    // increment count of nonzeros in col i 
        }

        if (D[k] == 0.0)
        {
            msg_error("SparseLDLSolver") << "Failed to factorize, D(k,k) is zero" ;
            return;
        }
    }
}

template<class TMatrix, class TVector, class TThreadManager>
class SparseLDLSolverImpl : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager>
{
public :
    SOFA_CLASS(SOFA_TEMPLATE3(SparseLDLSolverImpl,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,TThreadManager));
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector, TThreadManager> Inherit;

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef TThreadManager ThreadManager;
    typedef typename TMatrix::Real Real;

protected :

    Data<bool> d_precomputeSymbolicDecomposition; ///< If true the solver will reuse the precomputed symbolic decomposition. Otherwise it will recompute it at each step.
    core::objectmodel::lifecycle::DeprecatedData d_applyPermutation{this, "v24.06", "v24.12", "applyPermutation", "Use the Data 'ordering'"};
    Data<sofa::helper::OptionsGroup> d_orderingMethod; ///< Ordering method
    Data<int> d_L_nnz; ///< Number of non-zero values in the lower triangular matrix of the factorization. The lower, the faster the system is solved.

    static constexpr const char* s_defaultOrderingMethod { "Metis" };

    SparseLDLSolverImpl() : Inherit()
    , d_precomputeSymbolicDecomposition(initData(&d_precomputeSymbolicDecomposition, true ,"precomputeSymbolicDecomposition", "If true the solver will reuse the precomputed symbolic decomposition. Otherwise it will recompute it at each step."))
    , d_orderingMethod(initData(&d_orderingMethod, sofa::helper::OptionsGroup{"Natural", "AMD", "COLAMD", "Metis"}, "ordering", "Ordering method"))
    , d_L_nnz(initData(&d_L_nnz, 0, "L_nnz", "Number of non-zero values in the lower triangular matrix of the factorization. The lower, the faster the system is solved.", true, true))
    {
        sofa::helper::getWriteAccessor(d_orderingMethod)->setSelectedItem(s_defaultOrderingMethod);
    }

    template<class VecInt,class VecReal>
    void solve_cpu(Real * x,const Real * b,SparseLDLImplInvertData<VecInt,VecReal> * data)
    {
        int n = data->n;
        if (n == 0)
        {
            return;
        }

        const int * perm = data->perm.data();

        Tmp.clear();
        Tmp.fastResize(n);

        // A x = b
        //   <=> (L * D * L^T) * x = b
        //   <=> L y = b  with y = D * L^T * x  # Step 1: compute y from the system L y = b
        //   <=> D z = y  with z = L^T * x      # Step 2: compute z from the system D z = y
        //   <=> L^T * x = z                    # Step 3: compute x from the system L^T x = z

        // b, x, y and z can be read/written in the same vector:
        Real* const bPermuted = Tmp.data();
        Real* const xPermuted = Tmp.data();
        Real* const y = Tmp.data();
        Real* const z = Tmp.data();

        // apply the permutation to the right-hand side
        for (int i = 0; i < n; ++i)
        {
            bPermuted[i] = b[perm[i]];
        }

        // Step 1: compute y from the system L y = b
        // Note that L^T, stored in CSC, corresponds to L in CSR
        sofa::linearalgebra::solveLowerUnitriangularSystemCSR(n, bPermuted, y,
            data->LT_colptr.data(), data->LT_rowind.data(), data->LT_values.data());

        // Step 2: compute z from the system D z = y
        sofa::linearalgebra::solveDiagonalSystemUsingInvertedValues(n, y, z, data->invD.data());

        // Step 3: compute x from the system L^T x = z
        // Note that L, stored in CSC, corresponds to L^T in CSR
        sofa::linearalgebra::solveUpperUnitriangularSystemCSR(n, z, xPermuted,
            data->L_colptr.data(), data->L_rowind.data(), data->L_values.data());

        // apply the permutation to the solution
        for (int i = 0; i < n; ++i)
        {
            x[perm[i]] = xPermuted[i];
        }
    }

    void naturalOrder(int n, int* perm, int* invperm)
    {
        for (int j = 0; j < n; ++j)
        {
            perm[j] = j;
            invperm[j] = j;
        }
    }

    void LDL_ordering(int n, int nnz, int * M_colptr,int * M_rowind, Real* M_values, int * perm,int * invperm)
    {
        const auto orderingMethod = sofa::helper::getReadAccessor(d_orderingMethod)->getSelectedItem();
        if (orderingMethod != "Natural")
        {
            const auto computeOrdering = [&](auto ordering)
            {
                using PermutationType = typename decltype(ordering)::PermutationType;
                PermutationType permutation;
                using EigenSparseMatrix = Eigen::SparseMatrix<Real, Eigen::ColMajor>;
                using EigenSparseMatrixMap = Eigen::Map<const EigenSparseMatrix>;
                const auto map = EigenSparseMatrixMap( n, n, nnz, M_colptr, M_rowind, M_values);
                ordering.template operator()<const EigenSparseMatrix>(map, permutation);

                const PermutationType inv = permutation.inverse();

                for (int j = 0; j < n; ++j)
                {
                    perm[j] = permutation.indices()(j);
                    invperm[j] = inv.indices()(j) ;
                }
            };

            if (orderingMethod == "Metis")
            {
                // This could be the way to call Metis using the Eigen API, but
                // it triggers failing unit tests compared to the other method
                // computeOrdering(Eigen::MetisOrdering<int>());

                type::vector<int> xadj,adj,t_xadj,t_adj;
                csrToAdj( n, M_colptr, M_rowind, adj, xadj, t_adj, t_xadj, tran_countvec );

                /*
                int numflag = 0, options = 0;
                The new API of metis requires pointers on numflag and "options" which are "structure" to parametrize the factorization
                 We give NULL and NULL to use the default option (see doc of metis for details) !
                 If you have the error "SparseLDLSolver failure to factorize, D(k,k) is zero" that probably means that you use the previsou version of metis.
                 In this case you have to download and install the last version from : www.cs.umn.edu/~metis‎
                */
                METIS_NodeND(&n , xadj.data(), adj.data(), nullptr, nullptr, perm,invperm);
            }
            else if (orderingMethod == "COLAMD")
            {
                computeOrdering(Eigen::COLAMDOrdering<int>());
            }
            else if (orderingMethod == "AMD")
            {
                computeOrdering(Eigen::AMDOrdering<int>());
            }
            else
            {
                msg_error() << "Ordering method '" << orderingMethod << "' not supported: fallback to natural order";
                naturalOrder(n, perm, invperm);
            }
        }
        else 
        {
            naturalOrder(n, perm, invperm);
        }
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

    void LDL_numeric(int n,
                     int* M_colptr, int* M_rowind, Real* M_values,
                     int* colptr, int* rowind, Real* values,
                     Real* D, int* perm, int* invperm, int* Parent)
    {
        Y.resize(n);

        CSPARSE_numeric<Real>(n,M_colptr,M_rowind,M_values,colptr,rowind,values,D,perm,invperm,Parent,Flag.data(),Lnz.data(),Pattern.data(),Y.data());
    }

    template<class VecInt,class VecReal>
    void factorize(int n,int * M_colptr, int * M_rowind, Real * M_values, SparseLDLImplInvertData<VecInt,VecReal> * data)
    {
        data->new_factorization_needed =
            data->P_colptr.size() == 0 ||
            data->P_rowind.size() == 0 ||
            compareMatrixShape(n, M_colptr, M_rowind, data->n, (int*)data->P_colptr.data(), (int*)data->P_rowind.data());

        data->n = n;
        data->P_nnz = M_colptr[data->n];
        data->P_values.clear();
        data->P_values.fastResize(data->P_nnz);
        memcpy(data->P_values.data(), M_values, data->P_nnz * sizeof(Real));

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

            //ordering function
            LDL_ordering( data->n , data->P_nnz, M_colptr , M_rowind , M_values, data->perm.data(), data->invperm.data() );

            data->Parent.clear();
            data->Parent.resize(data->n);

            //symbolic factorization
            LDL_symbolic(data->n,M_colptr,M_rowind,data->L_colptr.data(),
                         data->perm.data(),data->invperm.data(),data->Parent.data());

            data->L_nnz = data->L_colptr[data->n];
            d_L_nnz.setValue(data->L_nnz);

            data->L_rowind.clear();data->L_rowind.fastResize(data->L_nnz);
            data->L_values.clear();data->L_values.fastResize(data->L_nnz);
            data->LT_rowind.clear();data->LT_rowind.fastResize(data->L_nnz);
            data->LT_values.clear();data->LT_values.fastResize(data->L_nnz);
        }

        Real * D = data->invD.data();
        int * rowind = data->L_rowind.data();
        int * colptr = data->L_colptr.data();
        Real * values = data->L_values.data();
        int * tran_rowind = data->LT_rowind.data();
        int * tran_colptr = data->LT_colptr.data();
        Real * tran_values = data->LT_values.data();

        //Numeric Factorization
        {
            SCOPED_TIMER_VARNAME(factorizationTimer, "numeric_factorization");
            LDL_numeric(data->n, M_colptr, M_rowind, M_values, colptr, rowind, values, D,
                        data->perm.data(), data->invperm.data(), data->Parent.data());

            //inverse the diagonal
            for (int i = 0; i < data->n; i++)
            {
                D[i] = 1 / D[i];
            }
        }

        // split the bloc diag in data->Bdiag

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
    }

    type::vector<Real> Tmp;
protected : //the following variables are used during the factorization they cannot be used in the main thread !

    type::vector<Real> Y;
    type::vector<int> Lnz,Flag,Pattern;
    type::vector<int> tran_countvec;
};

} // namespace sofa::component::linearsolver::direct
