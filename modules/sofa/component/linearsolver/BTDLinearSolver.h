/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_H

#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/simulation/tree/MatrixLinearSolver.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Linear system solver using the default (LU factorization) algorithm
template<class Matrix, class Vector>
class BTDLinearSolver : public sofa::simulation::tree::MatrixLinearSolver<Matrix,Vector>, public virtual sofa::core::objectmodel::BaseObject
{
public:
    Data<bool> f_verbose;

    typedef typename Matrix::SubMatrixType SubMatrix;

    //helper::vector<SubMatrix> alpha;
    helper::vector<SubMatrix> alpha_inv;
    helper::vector<SubMatrix> lambda;
    helper::vector<SubMatrix> B;
    Vector Y;

    Data<int> f_blockSize;

    BTDLinearSolver()
        : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
        , f_blockSize( initData(&f_blockSize,6,"blockSize","dimension of the blocks in the matrix") )
    {
    }

    /// Factorize M
    ///
    ///     [ A0 C0 0  0  ]         [ a0 0  0  0  ] [ I  l0 0  0  ]
    /// M = [ B1 A1 C1 0  ] = L U = [ B1 a1 0  0  ] [ 0  I  l1 0  ]
    ///     [ 0  B2 A2 C2 ]         [ 0  B2 a2 0  ] [ 0  0  I  l2 ]
    ///     [ 0  0  B3 A3 ]         [ 0  0  B3 a3 ] [ 0  0  0  I  ]
    ///     [ a0 a0l0    0       0       ]
    /// M = [ B1 B1l0+a1 a1l1    0       ]
    ///     [ 0  B2      B2l1+a2 a2l2    ]
    ///     [ 0  0       B3      B3l2+a3 ]
    /// L X = [ a0X0 B1X0+a1X1 B2X1+a2X2 B3X2+a3X3 ]
    ///        [                       inva0                   0             0     0 ]
    /// Linv = [               -inva1B1inva0               inva1             0     0 ]
    ///        [         inva2B2inva1B1inva0       -inva2B2inva1         inva2     0 ]
    ///        [ -inva3B3inva2B2inva1B1inva0 inva3B3inva2B2inva1 -inva3B3inva2 inva3 ]
    /// U X = [ X0+l0X1 X1+l1X2 X2+l2X3 X3 ]
    /// Uinv = [ I -l0 l0l1 -l0l1l2 ]
    ///        [ 0   I  -l1    l1l2 ]
    ///        [ 0   0    I     -l2 ]
    ///        [ 0   0    0       I ]
    ///
    ///                    [ (I+l0(I+l1(I+l2inva3B3)inva2B2)inva1B1)inva0 -l0(I+l1(I+l2inva3B3)inva2B2)inva1 l0l1(inva2+l2inva3B3inva2) -l0l1l2inva3 ]
    /// Minv = Uinv Linv = [    -((I+l1(I+l2inva3B3)inva2B2)inva1B1)inva0    (I+l1(I+l2inva3B3)inva2B2)inva1  -l1(inva2+l2inva3B3inva2)    l1l2inva3 ]
    ///                    [         (((I+l2inva3B3)inva2B2)inva1B1)inva0       -((I+l2inva3B3)inva2B2)inva1      inva2+l2inva3B3inva2     -l2inva3 ]
    ///                    [                  -inva3B3inva2B2inva1B1inva0                inva3B3inva2B2inva1             -inva3B3inva2        inva3 ]
    ///
    ///                    [ inva0-l0(Minv10)              (-l0)(Minv11)              (-l0)(Minv12)           (-l0)(Minv13) ]
    /// Minv = Uinv Linv = [         (Minv11)(-B1inva0) inva1-l1(Minv21)              (-l1)(Minv22)           (-l1)(Minv23) ]
    ///                    [         (Minv21)(-B1inva0)         (Minv22)(-B2inva1) inva2-l2(Minv32)           (-l2)(Minv33) ]
    ///                    [         (Minv31)(-B1inva0)         (Minv32)(-B2inva1)         (Minv33)(-B3inva2)       inva3   ]
    ///
    /// if M is symmetric (Ai = Ait and Bi+1 = C1t) :
    /// li = invai*Ci = (invai)t*(Bi+1)t = (B(i+1)invai)t
    ///
    ///                    [ inva0-l0(Minv10)     Minv10t          Minv20t      Minv30t ]
    /// Minv = Uinv Linv = [  (Minv11)(-l0t)  inva1-l1(Minv21)     Minv21t      Minv31t ]
    ///                    [  (Minv21)(-l0t)   (Minv22)(-l1t)  inva2-l2(Minv32) Minv32t ]
    ///                    [  (Minv31)(-l0t)   (Minv32)(-l1t)   (Minv33)(-l2t)   inva3  ]
    ///
    void invert(Matrix& M)
    {
        const bool verbose  = f_verbose.getValue();
        const int bsize = f_blockSize.getValue();
        const int nb = M.rowSize() / bsize;
        if (nb == 0) return;
        //alpha.resize(nb);
        alpha_inv.resize(nb);
        lambda.resize(nb-1);
        B.resize(nb);

        SubMatrix A, C;

        M.getSubMatrix(0*bsize,0*bsize,bsize,bsize,A);
        //if (verbose) std::cout << "A[0] = " << A << std::endl;
        M.getSubMatrix(0*bsize,1*bsize,bsize,bsize,C);
        //if (verbose) std::cout << "C[0] = " << C << std::endl;
        //alpha[0] = A;
        alpha_inv[0] = A.i();
        if (verbose) std::cout << "alpha_inv[0] = " << alpha_inv[0] << std::endl;
        lambda[0] = alpha_inv[0]*C;
        if (verbose) std::cout << "lambda[0] = " << lambda[0] << std::endl;
        //if (verbose) std::cout << "C[0] = alpha[0]*lambda[0] = " << alpha[0]*lambda[0] << std::endl;
        for (int i=1; i<nb; ++i)
        {
            M.getSubMatrix((i  )*bsize,(i  )*bsize,bsize,bsize,A);
            //if (verbose) std::cout << "A["<<i<<"] = " << A << std::endl;
            M.getSubMatrix((i  )*bsize,(i-1)*bsize,bsize,bsize,B[i]);
            //if (verbose) std::cout << "B["<<i<<"] = " << B[i] << std::endl;
            //alpha[i] = (A - B[i]*lambda[i-1]);
            alpha_inv[i] = (A - B[i]*lambda[i-1]).i();
            if (verbose) std::cout << "alpha_inv["<<i<<"] = " << alpha_inv[i] << std::endl;
            //if (verbose) std::cout << "A["<<i<<"] = B["<<i<<"]*lambda["<<i-1<<"]+alpha["<<i<<"] = " << B[i]*lambda[i-1]+alpha[i] << std::endl;
            if (i<nb-1)
            {
                M.getSubMatrix((i  )*bsize,(i+1)*bsize,bsize,bsize,C);
                //if (verbose) std::cout << "C["<<i<<"] = " << C << std::endl;
                lambda[i] = alpha_inv[i]*C;
                if (verbose) std::cout << "lambda["<<i<<"] = " << lambda[i] << std::endl;
                //if (verbose) std::cout << "C["<<i<<"] = alpha["<<i<<"]*lambda["<<i<<"] = " << alpha[i]*lambda[i] << std::endl;
            }
        }
    }

    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b)
    {
        using std::cerr;
        using std::endl;

        const bool verbose  = f_verbose.getValue() || f_printLog.getValue();

        if( verbose )
        {
            cerr<<"BTDLinearSolver, b = "<< b <<endl;
            cerr<<"BTDLinearSolver, M = "<< M <<endl;
        }

        invert(M);

        const int bsize = f_blockSize.getValue();
        const int nb = M.rowSize() / bsize;
        if (nb == 0) return;

        //if (verbose) std::cout << "D["<<0<<"] = " << b.sub(0,bsize) << std::endl;
        x.sub(0,bsize) = alpha_inv[0] * b.sub(0,bsize);
        //if (verbose) std::cout << "Y["<<0<<"] = " << x.sub(0,bsize) << std::endl;
        for (int i=1; i<nb; ++i)
        {
            //if (verbose) std::cout << "D["<<i<<"] = " << b.sub(i*bsize,bsize) << std::endl;
            x.sub(i*bsize,bsize) = alpha_inv[i]*(b.sub(i*bsize,bsize) - B[i]*x.sub((i-1)*bsize,bsize));
            //if (verbose) std::cout << "Y["<<i<<"] = " << x.sub(i*bsize,bsize) << std::endl;
        }
        //x.sub((nb-1)*bsize,bsize) = Y.sub((nb-1)*bsize,bsize);
        //if (verbose) std::cout << "x["<<nb-1<<"] = " << x.sub((nb-1)*bsize,bsize) << std::endl;
        for (int i=nb-2; i>=0; --i)
        {
            x.sub(i*bsize,bsize) /* = Y.sub(i*bsize,bsize)- */ -= lambda[i]*x.sub((i+1)*bsize,bsize);
            //if (verbose) std::cout << "x["<<i<<"] = " << x.sub(i*bsize,bsize) << std::endl;
        }

        // x is the solution of the system
        if( verbose )
        {
            cerr<<"BTDLinearSolver::solve, solution = "<<x<<endl;
        }
    }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
