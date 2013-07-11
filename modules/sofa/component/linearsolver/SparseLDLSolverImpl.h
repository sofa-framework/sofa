/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVERIMPL_H
#define SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVERIMPL_H

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>

#ifdef SOFA_HAVE_METIS
extern "C" {
#include <metis.h>
}
#endif

namespace sofa
{

namespace component
{

namespace linearsolver
{

//defaut structure for a LDL factorization
template<class VecInt,class VecReal>
class SpaseLDLImplInvertData : public MatrixInvertData {
public :
    int n, P_nnz, L_nnz;
    VecInt P_rowind,P_colptr,L_rowind,L_colptr,LT_rowind,LT_colptr;
    VecInt perm, invperm;
    VecReal P_values,L_values,LT_values,invD;
    helper::vector<int> Parent;
};

template<class TMatrix, class TVector, class TThreadManager>
class SparseLDLSolverImpl : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager>
{
public :
    SOFA_CLASS(SOFA_TEMPLATE3(SparseLDLSolverImpl,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,TThreadManager));
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector, TThreadManager> Inherit;

public:
    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef TThreadManager ThreadManager;
    typedef typename TMatrix::Real Real;

protected :

    SparseLDLSolverImpl() : Inherit() {}

    template<class VecInt,class VecReal>
    void solve_cpu(Real * x,const Real * b,SpaseLDLImplInvertData<VecInt,VecReal> * data) {
        int n = data->n;
        const Real * invD = &data->invD[0];
        const int * perm = &data->perm[0];
        const int * L_colptr = &data->L_colptr[0];
        const int * L_rowind = &data->L_rowind[0];
        const Real * L_values = &data->L_values[0];
        const int * LT_colptr = &data->LT_colptr[0];
        const int * LT_rowind = &data->LT_rowind[0];
        const Real * LT_values = &data->LT_values[0];

        Tmp.resize(n);

        for (int j = 0 ; j < n ; j++) {
            Real acc = b[perm[j]];
            for (int p = LT_colptr [j] ; p < LT_colptr[j+1] ; p++) {
                acc -= LT_values[p] * Tmp[LT_rowind[p]];
            }
            Tmp[j] = acc;
        }

        for (int j = n-1 ; j >= 0 ; j--) {
            Tmp[j] *= invD[j];

            for (int p = L_colptr[j] ; p < L_colptr[j+1] ; p++) {
                Tmp[j] -= L_values[p] * Tmp[L_rowind[p]];
            }

            x[perm[j]] = Tmp[j];
        }
    }

    template<class VecInt,class VecReal>
    void factorize(TMatrix & M,SpaseLDLImplInvertData<VecInt,VecReal> * data) {
        Mfiltered.copyNonZeros(M);
        Mfiltered.compress();

        int * M_colptr = (int *) &Mfiltered.getRowBegin()[0];
        int * M_rowind = (int *) &Mfiltered.getColsIndex()[0];
        Real * M_values = (Real *) &Mfiltered.getColsValue()[0];

        bool new_factorization_needed = need_symbolic_factorization(data->P_colptr,data->P_rowind);

        // we test if the matrix has the same struct as previous factorized matrix
        if (new_factorization_needed) {
            sout << "RECOMPUTE NEW FACTORIZATION" << sendl;
            data->n = M.colSize();
            data->P_nnz = M_colptr[data->n];

            data->perm.clear();data->perm.resize(data->n);
            data->invperm.clear();data->invperm.resize(data->n);
            data->invD.clear();data->invD.resize(data->n);
            data->P_colptr.clear();data->P_colptr.resize(data->n+1);
            data->L_colptr.clear();data->L_colptr.resize(data->n+1);
            data->LT_colptr.clear();data->LT_colptr.resize(data->n+1);
            data->P_rowind.clear();data->P_rowind.resize(data->P_nnz);
            data->P_values.clear();data->P_values.resize(data->P_nnz);

            memcpy(&data->P_colptr[0],M_colptr,(data->n+1) * sizeof(int));
            memcpy(&data->P_rowind[0],M_rowind,data->P_nnz * sizeof(int));
            memcpy(&data->P_values[0],M_values,data->P_nnz * sizeof(Real));

            //ordering function
            LDL_ordering(data->n,M_colptr,M_rowind,&data->perm[0],&data->invperm[0]);

            data->Parent.clear();
            data->Parent.resize(data->n);

            //symbolic factorization
            LDL_symbolic(data->n,M_colptr,M_rowind,&data->L_colptr[0],&data->perm[0],&data->invperm[0],&data->Parent[0]);

            data->L_nnz = data->L_colptr[data->n];

            data->L_rowind.clear();data->L_rowind.resize(data->L_nnz);
            data->L_values.clear();data->L_values.resize(data->L_nnz);
            data->LT_rowind.clear();data->LT_rowind.resize(data->L_nnz);
            data->LT_values.clear();data->LT_values.resize(data->L_nnz);
        }

        Real * D = &data->invD[0];
        int * rowind = &data->L_rowind[0];
        int * colptr = &data->L_colptr[0];
        Real * values = &data->L_values[0];
        int * tran_rowind = &data->LT_rowind[0];
        int * tran_colptr = &data->LT_colptr[0];
        Real * tran_values = &data->LT_values[0];

        //Numeric Factorization
        LDL_numeric(data->n,M_colptr,M_rowind,M_values,colptr,rowind,values,D,&data->perm[0],&data->invperm[0],&data->Parent[0]);

        //inverse the diagonal
        for (int i=0;i<data->n;i++) D[i] = 1.0/D[i];

        if (new_factorization_needed) {
            //Compute transpose in tran_colptr, tran_rowind, tran_values, tran_D
            tran_countvec.clear();
            tran_countvec.resize(data->n);

            //First we count the number of value on each row.
            for (int j=0;j<data->L_nnz;j++) tran_countvec[rowind[j]]++;

            //Now we make a scan to build tran_colptr
            tran_colptr[0] = 0;
            for (int j=0;j<data->n;j++) tran_colptr[j+1] = tran_colptr[j] + tran_countvec[j];
        }

        //we clear tran_countvec becaus we use it now to stro hown many value are written on each line
        tran_countvec.clear();
        tran_countvec.resize(data->n);

        for (int j=0;j<data->n;j++) {
          for (int i=colptr[j];i<colptr[j+1];i++) {
            int line = rowind[i];
            tran_rowind[tran_colptr[line] + tran_countvec[line]] = j;
            tran_values[tran_colptr[line] + tran_countvec[line]] = values[i];
            tran_countvec[line]++;
          }
        }
    }

    void LDL_ordering(int n,int * M_colptr,int * M_rowind,int * perm,int * invperm)
    {
#ifdef SOFA_HAVE_METIS
        int  num_flag     = 0;
        int  options_flag = 0;

        xadj.resize(n+1);
        adj.resize(M_colptr[n]-n);

        int it = 0;
        for (int j=0; j<n; j++)
        {
            xadj[j] = M_colptr[j] - j;

            for (int ip = M_colptr[j]; ip < M_colptr[j+1]; ip++)
            {
                int i = M_rowind[ip];
                if (i != j) adj[it++] = i;
            }
        }
        xadj[n] = M_colptr[n] - n;

        METIS_NodeND(&n, &xadj[0],&adj[0], &num_flag, &options_flag, perm,invperm);
#else
        for (int i=0; i<n; i++)
        {
            perm[i] = i;
            invperm[i] = i;
        }
#endif
    }

    void LDL_symbolic (int n,int * M_colptr,int * M_rowind,int * colptr,int * perm,int * invperm,int * Parent)
    {
        Lnz.clear();
        Flag.clear();
        Pattern.clear();

        Lnz.resize(n);
        Flag.resize(n);
        Pattern.resize(n);

        for (int k = 0 ; k < n ; k++)
        {
            Parent [k] = -1 ;	    /* parent of k is not yet known */
            Flag [k] = k ;		    /* mark node k as visited */
            Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
            int kk = perm[k];  /* kth original, or permuted, column */
            for (int p = M_colptr[kk] ; p < M_colptr[kk+1] ; p++)
            {
                /* A (i,k) is nonzero (original or permuted A) */
                int i = invperm[M_rowind[p]];
                if (i < k)
                {
                    /* follow path from i to root of etree, stop at flagged node */
                    for ( ; Flag [i] != k ; i = Parent [i])
                    {
                        /* find parent of i if not yet determined */
                        if (Parent [i] == -1) Parent [i] = k ;
                        Lnz [i]++ ;				/* L (k,i) is nonzero */
                        Flag [i] = k ;			/* mark i as visited */
                    }
                }
            }
        }

        colptr[0] = 0 ;
        for (int k = 0 ; k < n ; k++) colptr[k+1] = colptr[k] + Lnz[k] ;
    }

    void LDL_numeric(int n,int * M_colptr,int * M_rowind,Real * M_values,int * colptr,int * rowind,Real * values,Real * D,int * perm,int * invperm,int * Parent)
    {
        Real yi, l_ki ;
        int i, p, kk, len, top ;

        Y.resize(n);

        for (int k = 0 ; k < n ; k++)
        {
            Y [k] = 0.0 ;		    /* Y(0:k) is now all zero */
            top = n ;		    /* stack for pattern is empty */
            Flag [k] = k ;		    /* mark node k as visited */
            Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
            kk = perm[k];  /* kth original, or permuted, column */
            for (p = M_colptr[kk] ; p < M_colptr[kk+1] ; p++)
            {
                i = invperm[M_rowind[p]];	/* get A(i,k) */
                if (i <= k)
                {
                    Y[i] += M_values[p] ;  /* scatter A(i,k) into Y (sum duplicates) */
                    for (len = 0 ; Flag[i] != k ; i = Parent[i])
                    {
                        Pattern [len++] = i ;   /* L(k,i) is nonzero */
                        Flag [i] = k ;	    /* mark i as visited */
                    }
                    while (len > 0) Pattern[--top] = Pattern [--len] ;
                }
            }
            /* compute numerical values kth row of L (a sparse triangular solve) */
            D[k] = Y [k] ;		    /* get D(k,k) and clear Y(k) */
            Y[k] = 0.0 ;
            for ( ; top < n ; top++)
            {
                i = Pattern [top] ;	    /* Pattern [top:n-1] is pattern of L(:,k) */
                yi = Y [i] ;	    /* get and clear Y(i) */
                Y [i] = 0.0 ;
                for (p = colptr[i] ; p < colptr[i] + Lnz [i] ; p++)
                {
                    Y[rowind[p]] -= values[p] * yi ;
                }
                l_ki = yi / D[i] ;	    /* the nonzero entry L(k,i) */
                D[k] -= l_ki * yi ;
                rowind[p] = k ;	    /* store L(k,i) in column form of L */
                values[p] = l_ki ;
                Lnz[i]++ ;		    /* increment count of nonzeros in col i */
            }
            if (D[k] == 0.0)
            {
                serr << "SparseLDLSolver failure to factorize, D(k,k) is zero" << sendl;
                return;
            }
        }
    }

    template<class VecInt>
    bool need_symbolic_factorization(const VecInt & P_colptr,const VecInt & P_rowind) {
        if (P_colptr.size() != Mfiltered.getRowBegin().size()) return true;
        if (P_rowind.size() != Mfiltered.getColsIndex().size()) return true;

        const int * M_colptr = (int *) &Mfiltered.getRowBegin()[0];
        const int * M_rowind = (int *) &Mfiltered.getColsIndex()[0];
        const int * colptr = &P_colptr[0];
        const int * rowind = &P_rowind[0];

        for (unsigned i=0;i<P_colptr.size();i++) {
            if (M_colptr[i]!=colptr[i]) return true;
        }

        for (unsigned i=0;i<P_rowind.size();i++) {
            if (M_rowind[i]!=rowind[i]) return true;
        }

        return false;
    }

    helper::vector<Real> Tmp;
private : //the folowing variables are used during the factorization they canno be used in the main thread !
    helper::vector<int> xadj,adj;
    helper::vector<Real> Y;
    helper::vector<int> Lnz,Flag,Pattern;
    sofa::component::linearsolver::CompressedRowSparseMatrix<Real> Mfiltered;
    helper::vector<int> tran_countvec;
//    helper::vector<int> perm, invperm; //premutation inverse

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
