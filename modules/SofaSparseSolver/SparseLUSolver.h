/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <sofa/helper/map.h>
#include <math.h>
#include <csparse.h>

extern "C" {
#include <metis.h>
}

namespace sofa
{

namespace component
{

namespace linearsolver
{

//defaut structure for a LU factorization
template<class Real>
class SpaseLUInvertData : public MatrixInvertData {
public :

    css *S;//symbolique
    csn *N;// numeric
    cs A;
    helper::vector<int> A_i, A_p;//row_ind and col_ptr
    helper::vector<Real> A_x;//val
    helper::vector<int> perm, invperm;//permutation, inverse permutation
    Real * tmp;
    SpaseLUInvertData()
    {
        S=NULL; N=NULL; tmp=NULL;
    }

    ~SpaseLUInvertData()
    {
        if (S) cs_sfree (S);
        if (N) cs_nfree (N);
        if (tmp) cs_free (tmp);
    }
};

/// Direct linear solver based on Sparse LU factorization, implemented with the CSPARSE library
template<class TMatrix, class TVector, class TThreadManager= NoThreadManager>
class SparseLUSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(SparseLUSolver,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,TThreadManager));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;

    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager> Inherit;

    Data<bool> f_verbose;
    Data<double> f_tol;
    Data< double > f_filterValue;

    SparseLUSolver();
    void solve (Matrix& M, Vector& x, Vector& b);
    void invert(Matrix& M);

    //void init ();

protected :

    MatrixInvertData * createInvertData() {
        return new SpaseLUInvertData<Real>();
    }

    sofa::component::linearsolver::CompressedRowSparseMatrix<Real> Mfiltered;

    void LU_ordering(int n,int * M_colptr,int * M_rowind,int * perm,int * invperm) {
        //Compute transpose in tran_colptr, tran_rowind, tran_values, tran_D
        tran_countvec.clear();
        tran_countvec.resize(n);

        //First we count the number of value on each row.
        for (int j=0;j<n;j++) {
          for (int i=M_colptr[j];i<M_colptr[j+1];i++) {
              int col = M_rowind[i];
              if (col>j) tran_countvec[col]++;
          }
        }

        //Now we make a scan to build tran_colptr
        t_xadj.resize(n+1);
        t_xadj[0] = 0;
        for (int j=0;j<n;j++) t_xadj[j+1] = t_xadj[j] + tran_countvec[j];

        //we clear tran_countvec becaus we use it now to stro hown many value are written on each line
        tran_countvec.clear();
        tran_countvec.resize(n);

        t_adj.resize(t_xadj[n]);
        for (int j=0;j<n;j++) {
          for (int i=M_colptr[j];i<M_colptr[j+1];i++) {
            int line = M_rowind[i];
            if (line>j) {
                t_adj[t_xadj[line] + tran_countvec[line]] = j;
                tran_countvec[line]++;
            }
          }
        }

        adj.clear();
        xadj.resize(n+1);
        xadj[0] = 0;
        for (int j=0; j<n; j++)
        {
            //copy the lower part
            for (int ip = t_xadj[j]; ip < t_xadj[j+1]; ip++) {
                adj.push_back(t_adj[ip]);
            }

            //copy only the upper part
            for (int ip = M_colptr[j]; ip < M_colptr[j+1]; ip++) {
                int col = M_rowind[ip];
                if (col > j) adj.push_back(col);
            }

            xadj[j+1] = adj.size();
        }

        //int numflag = 0, options = 0;
        // The new API of metis requires pointers on numflag and "options" which are "structure" to parametrize the factorization
        // We give NULL and NULL to use the default option (see doc of metis for details) !
        // If you have the error "SparseLDLSolver failure to factorize, D(k,k) is zero" that probably means that you use the previsou version of metis.
        // In this case you have to download and install the last version from : www.cs.umn.edu/~metis‎
        METIS_NodeND(&n, &xadj[0],&adj[0], NULL, NULL, perm,invperm);
    }

    /* compute vnz, Pinv, leftmost, m2 from A and parent */
    static int *CSPARSE_vcount (const cs *A, const int *parent, int *m2, int *vnz)
    {
        int i, k, p, pa, n = A->n, m = A->m, *Ap = A->p, *Ai = A->i ;
        int *Pinv = (int *) cs_malloc (2*m+n, sizeof (int)), *leftmost = Pinv + m + n ;
        int *w = (int *) cs_malloc (m+3*n, sizeof (int)) ;
        int *next = w, *head = w + m, *tail = w + m + n, *nque = w + m + 2*n ;
        if (!Pinv || !w) return (cs_idone (Pinv, NULL, w, 0)) ;
        for (k = 0 ; k < n ; k++) head [k] = -1 ;	/* queue k is empty */
        for (k = 0 ; k < n ; k++) tail [k] = -1 ;
        for (k = 0 ; k < n ; k++) nque [k] = 0 ;
        for (i = 0 ; i < m ; i++) leftmost [i] = -1 ;
        for (k = n-1 ; k >= 0 ; k--)
        {
        for (p = Ap [k] ; p < Ap [k+1] ; p++)
        {
            leftmost [Ai [p]] = k ;	    /* leftmost[i] = min(find(A(i,:)))*/
        }
        }
        for (i = m-1 ; i >= 0 ; i--)	    /* scan rows in reverse order */
        {
        Pinv [i] = -1 ;			    /* row i is not yet ordered */
        k = leftmost [i] ;
        if (k == -1) continue ;		    /* row i is empty */
        if (nque [k]++ == 0) tail [k] = i ; /* first row in queue k */
        next [i] = head [k] ;		    /* put i at head of queue k */
        head [k] = i ;
        }
        (*vnz) = 0 ;
        (*m2) = m ;
        for (k = 0 ; k < n ; k++)		    /* find row permutation and nnz(V)*/
        {
        i = head [k] ;			    /* remove row i from queue k */
        (*vnz)++ ;			    /* count V(k,k) as nonzero */
        if (i < 0) i = (*m2)++ ;	    /* add a fictitious row */
        Pinv [i] = k ;			    /* associate row i with V(:,k) */
        if (--nque [k] <= 0) continue ;	    /* skip if V(k+1:m,k) is empty */
        (*vnz) += nque [k] ;		    /* nque [k] = nnz (V(k+1:m,k)) */
        if ((pa = parent [k]) != -1)	    /* move all rows to parent of k */
        {
            if (nque [pa] == 0) tail [pa] = tail [k] ;
            next [tail [k]] = head [pa] ;
            head [pa] = next [i] ;
            nque [pa] += nque [k] ;
        }
        }
        for (i = 0 ; i < m ; i++) if (Pinv [i] < 0) Pinv [i] = k++ ;
        return (cs_idone (Pinv, NULL, w, 1)) ;
    }

    /* symbolic analysis for LU */
    css *CSPARSE_sqr (const cs *A, int * /*Q*/)
    {
    //    int n, ok = 1;
    //    css *S ;
    //    if (!A) return (NULL) ;		    /* check inputs */
    //    n = A->n ;
    //    S = (css*) cs_calloc (1, sizeof (css)) ;	    /* allocate symbolic analysis */
    //    if (!S) return (NULL) ;		    /* out of memory */
    //    S->Q = Q;		    /* fill-reducing ordering */
    //    S->unz = 4*(A->p [n]) + n ;	    /* for LU factorization only, */
    //    S->lnz = S->unz ;		    /* guess nnz(L) and nnz(U) */
    //    return (ok ? S : cs_sfree (S)) ;

        int order=-1; int qr=0;

        int n, k, ok = 1, *post ;
        css *S ;
        if (!A) return (NULL) ;		    /* check inputs */
        n = A->n ;
        S = (css*) cs_calloc (1, sizeof (css)) ;	    /* allocate symbolic analysis */
        if (!S) return (NULL) ;		    /* out of memory */
        S->Q = cs_amd (A, order) ;		    /* fill-reducing ordering */
        if (order >= 0 && !S->Q) return (cs_sfree (S)) ;
        if (qr)				    /* QR symbolic analysis */
        {
            cs *C = (order >= 0) ? cs_permute (A, NULL, S->Q, 0) : ((cs *) A) ;
            S->parent = cs_etree (C, 1) ;	    /* etree of C'*C, where C=A(:,Q) */
            post = cs_post (n, S->parent) ;
            S->cp = cs_counts (C, S->parent, post, 1) ;  /* col counts chol(C'*C) */
            cs_free (post) ;
            ok = C && S->parent && S->cp ;
            if (ok) S->Pinv = CSPARSE_vcount (C, S->parent, &(S->m2), &(S->lnz)) ;
            ok = ok && S->Pinv ;
            if (ok) for (S->unz = 0, k = 0 ; k < n ; k++) S->unz += S->cp [k] ;
            if (order >= 0) cs_spfree (C) ;
        }
        else
        {
            S->unz = 4*(A->p [n]) + n ;	    /* for LU factorization only, */
            S->lnz = S->unz ;		    /* guess nnz(L) and nnz(U) */
        }
        return (ok ? S : cs_sfree (S)) ;

    }

    /* [L,U,Pinv]=lu(A, [Q lnz unz]). lnz and unz can be guess */
    csn *CSPARSE_lu (const cs *A, const css *S, double tol)
    {
        cs *L, *U ;
        csn *N ;
        double pivot, *Lx, *Ux, *x,  a, t ;
        int *Lp, *Li, *Up, *Ui, *xi, *Q, *Pinv, n, ipiv, k, top, p, i, col, lnz,unz;
        if (!A || !S) return (NULL) ;		    /* check inputs */
        n = A->n ;
        Q = S->Q ; lnz = S->lnz ; unz = S->unz ;
        x = (double*) cs_malloc (n, sizeof (double)) ;
        xi = (int*) cs_malloc (2*n, sizeof (int)) ;
        N = (csn*)cs_calloc (1, sizeof (csn)) ;
        if (!x || !xi || !N) return (cs_ndone (N, NULL, xi, x, 0)) ;
        N->L = L = cs_spalloc (n, n, lnz, 1, 0) ;	    /* initial L and U */
        N->U = U = cs_spalloc (n, n, unz, 1, 0) ;

        N->Pinv = Pinv = (int*) cs_malloc (n, sizeof (int)) ;
        if (!L || !U || !Pinv) return (cs_ndone (N, NULL, xi, x, 0)) ;
        Lp = L->p ; Up = U->p ;
        for (i = 0 ; i < n ; i++) x [i] = 0 ;	    /* clear workspace */
        for (i = 0 ; i < n ; i++) Pinv [i] = -1 ;	    /* no rows pivotal yet */
        for (k = 0 ; k <= n ; k++) Lp [k] = 0 ;	    /* no cols of L yet */
        lnz = unz = 0 ;
        for (k = 0 ; k < n ; k++)	    /* compute L(:,k) and U(:,k) */
        {
            /* --- Triangular solve --------------------------------------------- */
            Lp [k] = lnz ;		    /* L(:,k) starts here */
            Up [k] = unz ;		    /* U(:,k) starts here */
            if ((lnz + n > L->nzmax && !cs_sprealloc (L, 2*L->nzmax + n)) ||
                    (unz + n > U->nzmax && !cs_sprealloc (U, 2*U->nzmax + n)))
            {
                return (cs_ndone (N, NULL, xi, x, 0)) ;
            }
            Li = L->i ; Lx = L->x ; Ui = U->i ; Ux = U->x ;
            col = Q ? (Q [k]) : k ;
            top = cs_splsolve (L, A, col, xi, x, Pinv) ; /* x = L\A(:,col) */
            /* --- Find pivot --------------------------------------------------- */
            ipiv = -1 ;
            a = -1 ;
            for (p = top ; p < n ; p++)
            {
                i = xi [p] ;	    /* x(i) is nonzero */
                if (Pinv [i] < 0)	    /* row i is not pivotal */
                {
                    if ((t = fabs (x [i])) > a)
                    {
                        a = t ;	    /* largest pivot candidate so far */
                        ipiv = i ;
                    }
                }
                else		    /* x(i) is the entry U(Pinv[i],k) */
                {
                    Ui [unz] = Pinv [i] ;
                    Ux [unz++] = x [i] ;
                }
            }
            if (ipiv == -1 || a <= 0) return (cs_ndone (N, NULL, xi, x, 0)) ;
            if (Pinv [col] < 0 && fabs (x [col]) >= a*tol) ipiv = col ;
            /* --- Divide by pivot ---------------------------------------------- */
            pivot = x [ipiv] ;	    /* the chosen pivot */
            Ui [unz] = k ;		    /* last entry in U(:,k) is U(k,k) */
            Ux [unz++] = pivot ;
            Pinv [ipiv] = k ;	    /* ipiv is the kth pivot row */
            Li [lnz] = ipiv ;	    /* first entry in L(:,k) is L(k,k) = 1 */
            Lx [lnz++] = 1 ;
            for (p = top ; p < n ; p++) /* L(k+1:n,k) = x / pivot */
            {
                i = xi [p] ;
                if (Pinv [i] < 0)	    /* x(i) is an entry in L(:,k) */
                {
                    Li [lnz] = i ;	    /* save unpermuted row in L */
                    Lx [lnz++] = x [i] / pivot ;	/* scale pivot column */
                }
                x [i] = 0 ;		    /* x [0..n-1] = 0 for next k */
            }
        }


        /* --- Finalize L and U ------------------------------------------------- */
        Lp [n] = lnz ;
        Up [n] = unz ;
        Li = L->i ;			    /* fix row indices of L for final Pinv */
        for (p = 0 ; p < lnz ; p++) Li [p] = Pinv [Li [p]] ;
        cs_sprealloc (L, 0) ;	    /* remove extra space from L and U */
        cs_sprealloc (U, 0) ;
        return (cs_ndone (N, NULL, xi, x, 1)) ;	/* success */
    }

protected : //the folowing variables are used during the factorization they canno be used in the main thread !
    helper::vector<int> xadj,adj,t_xadj,t_adj;
    helper::vector<Real> Y;
    helper::vector<int> Lnz,Flag,Pattern;
    helper::vector<int> tran_countvec;
//    helper::vector<int> perm, invperm; //premutation inverse
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
