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
#include <sofa/component/linearsolver/ParallelMatrixLinearSolver.inl>

extern "C" {
#include <metis.h>
}

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TMatrix, class TVector>
class SparseLDLSolverImpl : public sofa::component::linearsolver::ParallelMatrixLinearSolver<TMatrix,TVector>
{
public :
    SOFA_CLASS(SOFA_TEMPLATE2(SparseLDLSolverImpl,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::ParallelMatrixLinearSolver,TMatrix,TVector));
    typedef sofa::component::linearsolver::ParallelMatrixLinearSolver<TMatrix,TVector> Inherit;
    typedef typename Inherit::JMatrixType JMatrixType;
    typedef typename Inherit::ResMatrixType ResMatrixType;

public:
    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename TMatrix::Real Real;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const SparseLDLSolverImpl<TMatrix,TVector>* = NULL)
    {
        return TMatrix::Name();
    }

protected :

    SparseLDLSolverImpl() : Inherit() {}

    void LDL_ordering(int n,int * M_colptr,int * M_rowind,int * perm,int * invperm)
    {
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
    }

    void LDL_symbolic (int n,int * M_colptr,int * M_rowind,int * colptr,int * perm,int * invperm)
    {
        Parent.clear();
        Lnz.clear();
        Flag.clear();
        Pattern.clear();
        Parent.resize(n);
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

    void LDL_numeric(int n,int * M_colptr,int * M_rowind,Real * M_values,int * colptr,int * rowind,Real * values,Real * D,int * perm,int * invperm)
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
                std::cerr << "CudaSparseLDLSolver failure to factorize, D(k,k) is zero" << std::endl;
                return;
            }
        }
    }


    helper::vector<int> xadj,adj;
    helper::vector<Real> Y;
    helper::vector<int> Parent,Lnz,Flag,Pattern;
//    helper::vector<int> perm, invperm; //premutation inverse

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
