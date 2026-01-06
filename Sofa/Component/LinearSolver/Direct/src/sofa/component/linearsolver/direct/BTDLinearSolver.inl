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
#include <sofa/component/linearsolver/direct/BTDLinearSolver.h>
#include <sofa/linearalgebra/FullMatrix.h>

namespace sofa::component::linearsolver::direct
{

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
///                    [ inva0-l0(Minv11)(-l0t)     Minv10t          Minv20t      Minv30t ]
/// Minv = Uinv Linv = [  (Minv11)(-l0t)  inva1-l1(Minv22)(-l1t)     Minv21t      Minv31t ]
///                    [  (Minv21)(-l0t)   (Minv22)(-l1t)  inva2-l2(Minv33)(-l2t) Minv32t ]
///                    [  (Minv31)(-l0t)   (Minv32)(-l1t)   (Minv33)(-l2t)   inva3  ]
///
template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::my_identity(SubMatrix& Id, const Index size_id)
{
    Id.resize(size_id,size_id);
    for (Index i=0; i<size_id; i++)
        Id.set(i,i,1.0);
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::invert(SubMatrix& Inv, const BlocType& m)
{
    SubMatrix M;
    M = m;
    // Check for diagonal matrices
    Index i0 = 0;
    const Index n = M.Nrows();
    Inv.resize(n,n);
    while (i0 < n)
    {
        Index j0 = i0+1;
        double eps = M.element(i0,i0)*1.0e-10;
        while (j0 < n)
            if (fabs(M.element(i0,j0)) > eps) break;
            else ++j0;
        if (j0 == n)
        {
            // i0 row is the identity
            Inv.set(i0,i0,(float)1.0/M.element(i0,i0));
            ++i0;
        }
        else break;
    }
    if (i0 < n)
//if (i0 == 0)
        Inv = M.i();
    //else if (i0 < n)
    //        Inv.sub(i0,i0,n-i0,n-i0) = M.sub(i0,i0,n-i0,n-i0).i();
    //else return true;
    //return false;
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::invert(Matrix& M)
{
    const bool verbose = d_verbose.getValue();

    msg_info_when(verbose) << "BTDLinearSolver, invert Matrix = "<< M ;

    constexpr Index bsize = Matrix::getSubMatrixDim();
    const Index nb = M.rowSize() / bsize;
    if (nb == 0) return;
    //alpha.resize(nb);
    alpha_inv.resize(nb);
    lambda.resize(nb-1);
    B.resize(nb);

    /////////////////////////// subpartSolve init ////////////

    if(d_subpartSolve.getValue() )
    {
        this->init_partial_inverse(nb,bsize);
    }

    SubMatrix A, C;
    //Index ndiag = 0;
    M.getAlignedSubMatrix(0,0,bsize,bsize,A);
    M.getAlignedSubMatrix(0,1,bsize,bsize,C);
    invert(alpha_inv[0],A);
    msg_info_when(verbose) << "alpha_inv[0] = " << alpha_inv[0] ;
    lambda[0] = alpha_inv[0]*C;
    msg_info_when(verbose) << "lambda[0] = " << lambda[0] ;

    for (Index i=1; i<nb; ++i)
    {
        M.getAlignedSubMatrix((i  ),(i  ),bsize,bsize,A);
        M.getAlignedSubMatrix((i  ),(i-1),bsize,bsize,B[i]);

        BlocType Temp1= B[i]*lambda[i-1];
        BlocType Temp2= A - Temp1;
        invert(alpha_inv[i], Temp2);

        msg_info_when(verbose) << "alpha_inv["<<i<<"] = " << alpha_inv[i] ;
        if (i<nb-1)
        {
            M.getAlignedSubMatrix((i  ),(i+1),bsize,bsize,C);
            lambda[i] = alpha_inv[i]*C;

            msg_info_when(verbose) << "lambda["<<i<<"] = " << lambda[i] ;
        }
    }
    nBlockComputedMinv.resize(nb);
    for (Index i=0; i<nb; ++i)
        nBlockComputedMinv[i] = 0;

    // WARNING : cost of resize here : ???
    Minv.resize(nb*bsize,nb*bsize);
    Minv.setAlignedSubMatrix((nb-1),(nb-1),bsize,bsize,alpha_inv[nb-1]);

    nBlockComputedMinv[nb-1] = 1;

    if(d_subpartSolve.getValue() )
    {
        SubMatrix iHi; // bizarre: pb compilation avec SubMatrix nHn_1 = B[i] *alpha_inv[i];
        my_identity(iHi, bsize);
        H.insert( make_pair(  IndexPair(nb-1, nb-1), iHi  ) );

        // on calcule les blocks diagonaux jusqu'au bout!!
        // TODO : ajouter un compteur "first_block" qui évite de descendre les déplacements jusqu'au block 0 dans partial_solve si ce block n'a pas été appelé
        computeMinvBlock(0, 0);
    }
}



///
///                    [ inva0-l0(Minv10)     Minv10t          Minv20t      Minv30t ]
/// Minv = Uinv Linv = [  (Minv11)(-l0t)  inva1-l1(Minv21)     Minv21t      Minv31t ]
///                    [  (Minv21)(-l0t)   (Minv22)(-l1t)  inva2-l2(Minv32) Minv32t ]
///                    [  (Minv31)(-l0t)   (Minv32)(-l1t)   (Minv33)(-l2t)   inva3  ]
///

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::computeMinvBlock(Index i, Index j)
{
    if (i < j)
    {
        // i < j correspond to the upper diagonal
        // for the computation, we use the lower diagonal matrix
        const Index t = i; i = j; j = t;
    }
    if (nBlockComputedMinv[i] > i-j) return; // the block was already computed



    ///// the block was not computed yet :

    // the block is computed now :
    // 1. all the diagonal block between N and i need to be computed
    constexpr Index bsize = Matrix::getSubMatrixDim();
    sofa::SignedIndex i0 = i;
    while (nBlockComputedMinv[i0]==0)
        ++i0;
    // i0 is the "closest" block of the diagonal that is computed
    // we need to compute all the Minv[i0][i0] (with i0>=i) till i0=i
    while (i0 > i)
    {
        if (nBlockComputedMinv[i0] == 1) // only the block on the diagonal is computed : need of the block [i0][i0-1]
        {
            // compute block (i0,i0-1)
            //Minv[i0][i0-1] = Minv[i0][i0]*-L[i0-1].t()
            Minv.asub((i0  ),(i0-1),bsize,bsize) = Minv.asub((i0  ),(i0  ),bsize,bsize)*(-(lambda[i0-1].t()));
            ++nBlockComputedMinv[i0];

            if(d_subpartSolve.getValue() )
            {
                // store -L[i0-1].t() H structure
                SubMatrix iHi_1;
                iHi_1 = - lambda[i0-1].t();
                H.insert( make_pair(  IndexPair(i0, i0-1), iHi_1  ) );
                // compute block (i0,i0-1) :  the upper diagonal blocks Minv[i0-1][i0]
                Minv.asub((i0-1),(i0),bsize,bsize) = -lambda[i0-1] * Minv.asub((i0  ),(i0  ),bsize,bsize);
            }

        }


        // compute block (i0-1,i0-1)  : //Minv[i0-1][i0-1] = inv(M[i0-1][i0-1]) + L[i0-1] * Minv[i0][i0-1]
        Minv.asub((i0-1),(i0-1),bsize,bsize) = alpha_inv[i0-1] - lambda[i0-1]*Minv.asub((i0  ),(i0-1),bsize,bsize);

        if(d_subpartSolve.getValue() )
        {
            // store Id in H structure
            SubMatrix iHi;
            my_identity(iHi, bsize);
            H.insert( make_pair(  IndexPair(i0-1, i0-1), iHi  ) );
        }

        ++nBlockComputedMinv[i0-1]; // now Minv[i0-1][i0-1] is computed so   nBlockComputedMinv[i0-1] = 1
        --i0;                       // we can go down to the following block (till we reach i)
    }


    //2. all the block on the lines of block i between the diagonal and the block j are computed
    // i0=i

    SignedIndex j0 = i-nBlockComputedMinv[i];


    /////////////// ADD : Calcul pour faire du partial_solve //////////
    // first iHj is initialized to iHj0+1 (that is supposed to be already computed)
    SubMatrix iHj ;
    if(d_subpartSolve.getValue() )
    {


        H_it = H.find( IndexPair(i0,j0+1) );

        if (H_it == H.end())
        {
            my_identity(iHj, bsize);
            msg_error_when(i0 != j0 + 1) << "element(" << i0 << "," << j0 + 1 << ") not found : nBlockComputedMinv[i] = " << nBlockComputedMinv[i];
        }
        else
        {
            iHj = H_it->second;
        }

    }
    /////////////////////////////////////////////////////////////////////

    while (j0 >= j)
    {
        // compute block (i0,j0)
        // Minv[i][j0] = Minv[i][j0+1] * (-L[j0].t)
        Minv.asub((i0  ),(j0  ),bsize,bsize) = Minv.asub((i0  ),(j0+1),bsize,bsize)*(-lambda[j0].t());
        if(d_subpartSolve.getValue() )
        {
            // iHj0 = iHj0+1 * (-L[j0].t)
            iHj = iHj * -lambda[j0].t();
            H.insert(make_pair(IndexPair(i0,j0),iHj));

            // compute block (j0,i0)  the upper diagonal blocks Minv[j0][i0]
            Minv.asub((j0  ),(i0  ),bsize,bsize) = -lambda[j0]*Minv.asub((j0+1),(i0),bsize,bsize);
        }
        ++nBlockComputedMinv[i0];
        --j0;
    }
}

template<class Matrix, class Vector>
double BTDLinearSolver<Matrix,Vector>::getMinvElement(Index i, Index j)
{
    constexpr Index bsize = Matrix::getSubMatrixDim();
    if (i < j)
    {
        // lower diagonal
        return getMinvElement(j,i);
    }
    computeMinvBlock(i/bsize, j/bsize);
    return Minv.element(i,j);
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::solve (Matrix& /*M*/, Vector& x, Vector& b)
{
    const bool verbose = d_verbose.getValue();

    msg_info_when(verbose) << "solve, b = "<< b;

    constexpr Index bsize = Matrix::getSubMatrixDim();
    const Index nb = b.size() / bsize;
    if (nb == 0) return;

    x.asub(0,bsize) = alpha_inv[0] * b.asub(0,bsize);
    for (Index i=1; i<nb; ++i)
    {
        x.asub(i,bsize) = alpha_inv[i]*(b.asub(i,bsize) - B[i]*x.asub((i-1),bsize));
    }
    for (sofa::SignedIndex i=nb-2; i>=0; --i)
    {
        x.asub(i,bsize) /* = Y.asub(i,bsize)- */ -= lambda[i]*x.asub((i+1),bsize);
    }

    // x is the solution of the system
    msg_info_when(verbose) << "solve, solution = "<<x;

}

template<class Matrix, class Vector>
bool BTDLinearSolver<Matrix,Vector>::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    using namespace sofa::linearalgebra;

    if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    else if (FullMatrix<float>* r = dynamic_cast<FullMatrix<float>*>(result))
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    else if (linearalgebra::BaseMatrix* r = result)
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    return false;
}



///////////////////////////////////////
///////  partial solve  //////////
///////////////////////////////////////


template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::init_partial_inverse(const Index &/*nb*/, const Index &/*bsize*/)
{
    // need to stay in init_partial_inverse (called before inverse)
    H.clear();

}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::init_partial_solve()
{

    constexpr Index bsize = Matrix::getSubMatrixDim();
    const Index nb = this->l_linearSystem->getRHSVector()->size() / bsize;

    //TODO => optimisation ??
    bwdContributionOnLH.clear();
    bwdContributionOnLH.resize(nb*bsize);
    fwdContributionOnRH.clear();
    fwdContributionOnRH.resize(nb*bsize);


    _rh_buf.resize(nb*bsize);
    _acc_rh_bloc=0;
    _acc_rh_bloc.resize(bsize);
    _acc_lh_bloc=0;
    _acc_lh_bloc.resize(bsize);

    // Block that is currently being proceed => start from the end (so that we use step2 bwdAccumulateLHGlobal and accumulate potential initial forces)
    current_bloc = nb-1;


    // DF represents the variation of the right hand side of the equation (Force in mechanics)
    Vec_dRH.resize(nb);
    for (Index i=0; i<nb; i++)
    {
        Vec_dRH[i]=0;
        Vec_dRH[i].resize(bsize);
        _rh_buf.asub(i,bsize) = this->l_linearSystem->getRHSVector()->asub(i,bsize) ;

    }




}


////// STEP 1


template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::bwdAccumulateRHinBloc(Index indMaxBloc)
{
    constexpr Index bsize = Matrix::getSubMatrixDim();
    const bool showProblem = d_problem.getValue();

    Index b=indMaxBloc;

    //debug
    if (indMaxBloc <  current_bloc)
    {
        dmsg_warning() <<"indMaxBloc = "<<indMaxBloc <<" <  "<<" current_bloc = "<<current_bloc ;
    }

    SubVector RHbloc;
    RHbloc.resize(bsize);

    _acc_lh_bloc= bwdContributionOnLH.asub(b,bsize);


    while(b > current_bloc )
    {

        // evaluate the Right Hand Term for the block b
        RHbloc = this->l_linearSystem->getRHSVector()->asub(b,bsize) ;

        // compute the contribution on LH created by RH
        _acc_lh_bloc  += Minv.asub(b,b,bsize,bsize) * RHbloc;

        b--;
        // accumulate this contribution on LH on the lower blocks
        _acc_lh_bloc =  -(lambda[b]*_acc_lh_bloc);

        dmsg_info_when(showProblem) << "bwdLH[" << b << "] = H[" << b << "][" << b + 1 << "] * (Minv[" << b + 1 << "][" << b + 1 << "] * RH[" << b + 1 << "] + bwdLH[" << b + 1 << "])";

        // store the contribution as bwdContributionOnLH
        bwdContributionOnLH.asub(b,bsize) = _acc_lh_bloc;

    }

    b = current_bloc;
    // compute the block which indice is current_bloc
    this->l_linearSystem->getSolutionVector()->asub(b,bsize) = Minv.asub( b, b ,bsize,bsize) * ( fwdContributionOnRH.asub(b, bsize) + this->l_linearSystem->getRHSVector()->asub(b,bsize) ) +
            bwdContributionOnLH.asub(b, bsize);

    dmsg_info_when(showProblem) << "LH[" << b << "] = Minv[" << b << "][" << b << "] * (fwdRH(" << b << ") + RH(" << b << ")) + bwdLH(" << b << ")";


    // here b==current_bloc
}



////// STEP 2

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::bwdAccumulateLHGlobal( )
{
    constexpr Index bsize = Matrix::getSubMatrixDim();
    _acc_lh_bloc =  bwdContributionOnLH.asub(current_bloc, bsize);

    const bool showProblem = d_problem.getValue();

    while( current_bloc > 0)
    {
        dmsg_info_when(showProblem) << "bwdLH[" << current_bloc - 1 << "] = H[" << current_bloc - 1 << "][" << current_bloc << "] *( bwdLH[" << current_bloc << "] + Minv[" << current_bloc << "][" << current_bloc << "] * RH[" << current_bloc << "])";

        // BwdLH += Minv*RH
        _acc_lh_bloc +=  Minv.asub(current_bloc,current_bloc,bsize,bsize) * this->l_linearSystem->getRHSVector()->asub(current_bloc,bsize) ;

        current_bloc--;
        // BwdLH(n-1) = H(n-1)(n)*BwdLH(n)
        _acc_lh_bloc = -(lambda[current_bloc]*_acc_lh_bloc);

        bwdContributionOnLH.asub(current_bloc, bsize) = _acc_lh_bloc;


    }

    // at this point, current_bloc must be equal to 0

    // all the forces from RH were accumulated through bwdAccumulation:
    _indMaxNonNullForce = 0;

    // need to update all the value of LH during forward
    _indMaxFwdLHComputed = 0;

    // init fwdContribution
    fwdContributionOnRH.asub(0, bsize) = 0;


}


/////// STEP 3

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::fwdAccumulateRHGlobal(Index indMinBloc)
{
    constexpr Index bsize = Matrix::getSubMatrixDim();
    _acc_rh_bloc =fwdContributionOnRH.asub(current_bloc, bsize);

    const bool showProblem = d_problem.getValue();

    while( current_bloc< indMinBloc)
    {

        // fwdRH(n) += RH(n)
        _acc_rh_bloc += this->l_linearSystem->getRHSVector()->asub(current_bloc,bsize);

        // fwdRH(n+1) = H(n+1)(n) * fwdRH(n)
        _acc_rh_bloc = -(lambda[current_bloc].t() * _acc_rh_bloc);
        current_bloc++;

        fwdContributionOnRH.asub(current_bloc, bsize) = _acc_rh_bloc;

        dmsg_info_when(showProblem) << "fwdRH[" << current_bloc << "] = H[" << current_bloc << "][" << current_bloc - 1 << "] * (fwdRH[" << current_bloc - 1 << "] + RH[" << current_bloc - 1 << "])";
    }

    _indMaxFwdLHComputed = current_bloc;


    Index b = current_bloc;
    // compute the block which indice is _indMaxFwdLHComputed
    this->l_linearSystem->getSolutionVector()->asub(b,bsize) = Minv.asub( b, b ,bsize,bsize) * ( fwdContributionOnRH.asub(b, bsize) + this->l_linearSystem->getRHSVector()->asub(b,bsize) ) +
            bwdContributionOnLH.asub(b, bsize);

    dmsg_info_when(showProblem) << "LH[" << b << "] = Minv[" << b << "][" << b << "] * (fwdRH(" << b << ") + RH(" << b << ")) + bwdLH(" << b << ")";

}


/////// STEP 4

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::fwdComputeLHinBloc(Index indMaxBloc)
{

    constexpr Index bsize = Matrix::getSubMatrixDim();
    const bool showProblem = d_problem.getValue();

    Index b;

    while(_indMaxFwdLHComputed < indMaxBloc )
    {

        b = _indMaxFwdLHComputed;

        if(b>=0)
        {
            dmsg_info_when(showProblem) << " fwdRH[" << b + 1 << "] = H[" << b + 1 << "][" << b << "] * (fwdRH(" << b << ") + RH(" << b << "))";
            // fwdRH(n+1) = H(n+1)(n) * (fwdRH(n) + RH(n))
            fwdContributionOnRH.asub(b+1, bsize) = (-lambda[b].t())* ( fwdContributionOnRH.asub(b, bsize) + this->l_linearSystem->getRHSVector()->asub(b,bsize) ) ;
        }

        _indMaxFwdLHComputed++; b++;

        // compute the block which indice is _indMaxFwdLHComputed
        this->l_linearSystem->getSolutionVector()->asub(b,bsize) = Minv.asub( b, b ,bsize,bsize) * ( fwdContributionOnRH.asub(b, bsize) + this->l_linearSystem->getRHSVector()->asub(b,bsize) ) +
                bwdContributionOnLH.asub(b, bsize);

        dmsg_info_when(showProblem) << "LH["<<b<<"] = Minv["<<b<<"]["<<b<<"] * (fwdRH("<<b<< ") + RH("<<b<<")) + bwdLH("<<b<<")";

    }




}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::partial_solve(ListIndex&  Iout, ListIndex&  Iin , bool NewIn)  ///*Matrix& M, Vector& result, Vector& rh, */
{

    const bool showProblem = d_problem.getValue();

    Index MinIdBloc_OUT = Iout.front();
    Index MaxIdBloc_OUT = Iout.back();
    if( NewIn)
    {

        Index MinIdBloc_IN = Iin.front(); //  Iin needs to be sorted
        Index MaxIdBloc_IN = Iin.back();  //
        //debug
        dmsg_info_when(showProblem) << "STEP1: new force on block between dofs "<< MinIdBloc_IN<< "  and "<<MaxIdBloc_IN;

        if (MaxIdBloc_IN > this->_indMaxNonNullForce)
            this->_indMaxNonNullForce = MaxIdBloc_IN;

        //step 1:
        bwdAccumulateRHinBloc(this->_indMaxNonNullForce );

        // now the fwdLH begins to be wrong when > to the indice of MinIdBloc_IN (need to be updated in step 3 or 4)
        this->_indMaxFwdLHComputed = MinIdBloc_IN;
    }

    if (current_bloc > MinIdBloc_OUT)
    {
        //debug
        dmsg_info_when(showProblem) << "STEP2 (bwd GLOBAL on structure) : current_bloc ="<<current_bloc<<" > to  MinIdBloc_OUT ="<<MinIdBloc_OUT;

        // step 2:
        bwdAccumulateLHGlobal();

        //debug
        dmsg_info_when(showProblem) << " new current_bloc = " << current_bloc;
    }


    if (current_bloc < MinIdBloc_OUT)
    {
        //debug
        dmsg_info_when(showProblem) << "STEP3 (fwd GLOBAL on structure) : current_bloc =" << current_bloc << " < to  MinIdBloc_OUT =" << MinIdBloc_OUT;

        //step 3:
        fwdAccumulateRHGlobal(MinIdBloc_OUT);

        // debug
        if (d_problem.getValue())
            dmsg_info_when(showProblem) << " new current_bloc = " << current_bloc;
    }

    if ( _indMaxFwdLHComputed < MaxIdBloc_OUT)
    {
        //debug
        dmsg_info_when(showProblem) << " STEP 4 :_indMaxFwdLHComputed = " << _indMaxFwdLHComputed << " < " << "MaxIdBloc_OUT = " << MaxIdBloc_OUT << "  - verify that current_bloc=" << current_bloc << " == " << " MinIdBloc_OUT =" << MinIdBloc_OUT;

        fwdComputeLHinBloc(MaxIdBloc_OUT );
        //debug
        dmsg_info_when(showProblem) << "  new _indMaxFwdLHComputed = " << _indMaxFwdLHComputed;
    }
    // debug: test
    if (d_verification.getValue())
    {
        constexpr Index bsize = Matrix::getSubMatrixDim();
        Vector *Result_partial_Solve = new Vector();
        (*Result_partial_Solve) = (*this->l_linearSystem->getSolutionVector());

        solve(*this->l_linearSystem->getSystemMatrix(),*this->l_linearSystem->getSolutionVector(), *this->l_linearSystem->getRHSVector());

        Vector *Result = new Vector();
        (*Result) = (*this->l_linearSystem->getSolutionVector());

        Vector *DR = new Vector();
        (*DR) = (*Result);
        (*DR) -= (*Result_partial_Solve);


        double normDR = 0.0;
        double normR = 0.0;
        for (Index i=MinIdBloc_OUT; i<=MaxIdBloc_OUT; i++)
        {
            normDR += (DR->asub(i,bsize)).norm();
            normR += (Result->asub(i,bsize)).norm();
        }

        if (normDR > ((1.0e-7)*normR + 1.0e-20) )
        {
            std::ostringstream oss;
            oss <<"++++++++++++++++ WARNING +++++++++++\n \n Found solution for block OUT :";
            for (Index i=MinIdBloc_OUT; i<=MaxIdBloc_OUT; i++)
            {
                oss <<"     ["<<i<<"] "<< Result_partial_Solve->asub(i,bsize);
            }
            oss << "\n";

            oss <<" after complete resolution OUT :";
            for (Index i=MinIdBloc_OUT; i<=MaxIdBloc_OUT; i++)
            {
                oss <<"     ["<<i<<"] "<<Result->asub(i,bsize);
            }
            oss << "\n";
            msg_warning() << oss.str();
        }

        delete(Result_partial_Solve);
        delete(Result);
        delete(DR);
        return;
    }
}

template<class Matrix, class Vector>
template<class RMatrix, class JMatrix>
bool BTDLinearSolver<Matrix,Vector>::addJMInvJt(RMatrix& result, JMatrix& J, double fact)
{
    const Index Jcols = J.colSize();
    if (Jcols != Minv.rowSize())
    {
        msg_error() << "AddJMInvJt: incompatible J matrix size.";
        return false;
    }

    if (this->notMuted())
    {
        std::stringstream tmpStr;
        tmpStr<< "C = ["<<msgendl;
        for  (Index mr=0; mr<Minv.rowSize(); mr++)
        {
            tmpStr<<" "<<msgendl;
            for (Index mc=0; mc<Minv.colSize(); mc++)
            {
                tmpStr<<" "<< getMinvElement(mr,mc);
            }
        }
        tmpStr << "];"<<msgendl;

        tmpStr<< "J = ["<<msgendl;
        for  (Index jr=0; jr<J.rowSize(); jr++)
        {
            tmpStr<<" "<<msgendl;
            for (Index jc=0; jc<J.colSize(); jc++)
            {
                tmpStr<<" "<< J.element(jr, jc) ;
            }
        }
        tmpStr << "];";
        msg_info() << tmpStr.str();
    }

    std::stringstream tmpStr2;
    const typename JMatrix::LineConstIterator jitend = J.end();
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != jitend; ++jit1)
    {
        Index row1 = jit1->first;
        for (typename JMatrix::LineConstIterator jit2 = jit1; jit2 != jitend; ++jit2)
        {
            Index row2 = jit2->first;
            double acc = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(), i1end = jit1->second.end(); i1 != i1end; ++i1)
            {
                const Index col1 = i1->first;
                const double val1 = i1->second;
                for (typename JMatrix::LElementConstIterator i2 = jit2->second.begin(), i2end = jit2->second.end(); i2 != i2end; ++i2)
                {
                    const Index col2 = i2->first;
                    const double val2 = i2->second;
                    acc += val1 * getMinvElement(col1,col2) * val2;
                }
            }

            if (this->notMuted())
            {
                tmpStr2 << "W("<<row1<<","<<row2<<") += "<<acc<<" * "<<fact<<msgendl;
            }

            acc *= fact;
            result.add(row1,row2,acc);
            if (row1!=row2)
                result.add(row2,row1,acc);
        }
    }

    return true;
}

} //namespace sofa::component::linearsolver::direct
