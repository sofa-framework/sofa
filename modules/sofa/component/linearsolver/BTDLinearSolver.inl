#ifndef SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_INL
#define SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_INL

#include "BTDLinearSolver.h"


namespace sofa
{

namespace component
{

namespace linearsolver
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
void BTDLinearSolver<Matrix,Vector>::my_identity(SubMatrix& Id, const int size_id)
{
    Id.resize(size_id,size_id);
    for (int i=0; i<size_id; i++)
        Id.set(i,i,1.0);
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::invert(SubMatrix& Inv, const BlocType& m)
{
    SubMatrix M;
    M = m;
    // Check for diagonal matrices
    unsigned int i0 = 0;
    const unsigned int n = M.Nrows();
    Inv.resize(n,n);
    while (i0 < n)
    {
        unsigned int j0 = i0+1;
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
    const bool verbose  = this->f_verbose.getValue() || this->f_printLog.getValue();

    if( verbose )
    {
        serr<<"BTDLinearSolver, invert Matrix = "<< M <<sendl;
    }

    const int bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    const int nb = M.rowSize() / bsize;
    if (nb == 0) return;
    //alpha.resize(nb);
    alpha_inv.resize(nb);
    lambda.resize(nb-1);
    B.resize(nb);

    /////////////////////////// subpartSolve init ////////////

    if(subpartSolve.getValue() )
    {
        H.clear();
        //_acc_result=0;
        _acc_result.resize(nb*bsize);
        //_rh_buf = 0;
        _rh_buf.resize(nb*bsize);
        //_df_buf = 0;
        //_df_buf.resize(nb*bsize);
        _acc_rh_current_block=0;
        _acc_rh_current_block.resize(bsize);
        _acc_lh_current_block=0;
        _acc_lh_current_block.resize(bsize);
        current_block = nb-1;

        Vec_df.resize(nb);
        for (int i=0; i<nb; i++)
        {
            Vec_df[i]=0;
            Vec_df[i].resize(bsize);
        }


    }

    SubMatrix A, C;
    //int ndiag = 0;
    M.getAlignedSubMatrix(0,0,bsize,bsize,A);
    //if (verbose) sout << "A[0] = " << A << sendl;
    M.getAlignedSubMatrix(0,1,bsize,bsize,C);
    //if (verbose) sout << "C[0] = " << C << sendl;
    //alpha[0] = A;
    invert(alpha_inv[0],A);
    if (verbose) sout << "alpha_inv[0] = " << alpha_inv[0] << sendl;
    lambda[0] = alpha_inv[0]*C;
    if (verbose) sout << "lambda[0] = " << lambda[0] << sendl;
    //if (verbose) sout << "C[0] = alpha[0]*lambda[0] = " << alpha[0]*lambda[0] << sendl;


    for (int i=1; i<nb; ++i)
    {
        M.getAlignedSubMatrix((i  ),(i  ),bsize,bsize,A);
        //if (verbose) sout << "A["<<i<<"] = " << A << sendl;
        M.getAlignedSubMatrix((i  ),(i-1),bsize,bsize,B[i]);
        //if (verbose) sout << "B["<<i<<"] = " << B[i] << sendl;
        //alpha[i] = (A - B[i]*lambda[i-1]);


        BlocType Temp1= B[i]*lambda[i-1];
        BlocType Temp2= A - Temp1;
        invert(alpha_inv[i], Temp2);


        //if(subpartSolve.getValue() ) {
        //	helper::vector<SubMatrix> nHn_1; // bizarre: pb compilation avec SubMatrix nHn_1 = B[i] *alpha_inv[i];
        //	nHn_1.resize(1);
        //	nHn_1[0] = B[i] *alpha_inv[i-1];
        //	H.insert(make_pair(IndexPair(i,i-1),nHn_1[0])); //IndexPair(i+1,i) ??
        //	serr<<" Add pair ("<<i<<","<<i-1<<")"<<sendl;
        //}

        if (verbose) sout << "alpha_inv["<<i<<"] = " << alpha_inv[i] << sendl;
        //if (verbose) sout << "A["<<i<<"] = B["<<i<<"]*lambda["<<i-1<<"]+alpha["<<i<<"] = " << B[i]*lambda[i-1]+alpha[i] << sendl;
        if (i<nb-1)
        {
            M.getAlignedSubMatrix((i  ),(i+1),bsize,bsize,C);
            //if (verbose) sout << "C["<<i<<"] = " << C << sendl;
            lambda[i] = alpha_inv[i]*C;
            if (verbose) sout << "lambda["<<i<<"] = " << lambda[i] << sendl;
            //if (verbose) sout << "C["<<i<<"] = alpha["<<i<<"]*lambda["<<i<<"] = " << alpha[i]*lambda[i] << sendl;
        }
    }
    nBlockComputedMinv.resize(nb);
    for (int i=0; i<nb; ++i)
        nBlockComputedMinv[i] = 0;

    // WARNING : cost of resize here : ???
    Minv.resize(nb*bsize,nb*bsize);
    Minv.setAlignedSubMatrix((nb-1),(nb-1),bsize,bsize,alpha_inv[nb-1]);

    //std::cout<<"Minv.setSubMatrix call for block number"<<(nb-1)<<std::endl;

    nBlockComputedMinv[nb-1] = 1;

    if(subpartSolve.getValue() )
    {
        SubMatrix iHi; // bizarre: pb compilation avec SubMatrix nHn_1 = B[i] *alpha_inv[i];
        my_identity(iHi, bsize);
        H.insert( make_pair(  IndexPair(nb-1, nb-1), iHi  ) );

        // on calcule les blocks diagonaux jusqu'au bout!!
        // TODO : ajouter un compteur "first_block" qui évite de descendre les déplacements jusqu'au block 0 dans partial_solve si ce block n'a pas été appelé
        computeMinvBlock(0, 0);
    }

    //sout << "BTDLinearSolver: "<<ndiag<<"/"<<nb<<"diagonal blocs."<<sendl;
}



///
///                    [ inva0-l0(Minv10)     Minv10t          Minv20t      Minv30t ]
/// Minv = Uinv Linv = [  (Minv11)(-l0t)  inva1-l1(Minv21)     Minv21t      Minv31t ]
///                    [  (Minv21)(-l0t)   (Minv22)(-l1t)  inva2-l2(Minv32) Minv32t ]
///                    [  (Minv31)(-l0t)   (Minv32)(-l1t)   (Minv33)(-l2t)   inva3  ]
///

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::computeMinvBlock(int i, int j)
{
    //serr<<"computeMinvBlock("<<i<<","<<j<<")"<<sendl;

    if (i < j)
    {
        // lower diagonal
        int t = i; i = j; j = t;
    }
    if (nBlockComputedMinv[i] > i-j) return; // the block was already computed


    // the block is computed now :
    // 1. all the diagonal block between N and i need to be computed
    const int bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    int i0 = i;
    while (nBlockComputedMinv[i0]==0)
        ++i0;
    // i0 is the first block of the diagonal that is computed
    while (i0 > i)
    {
        //serr<<"i0 ="<<i0<<"nBlockComputedMinv[i0]="<<nBlockComputedMinv[i0]<<sendl;
        if (nBlockComputedMinv[i0] == 1)
        {
            // compute bloc (i0,i0-1)
            Minv.asub((i0  ),(i0-1),bsize,bsize) = Minv.asub((i0  ),(i0  ),bsize,bsize)*(-(lambda[i0-1].t()));
            ++nBlockComputedMinv[i0];

            if(subpartSolve.getValue() )
            {
                helper::vector<SubMatrix> iHi_1; // bizarre: pb compilation avec SubMatrix nHn_1 = B[i] *alpha_inv[i];
                iHi_1.resize(1);
                iHi_1[0] = - lambda[i0-1].t();
                H.insert( make_pair(  IndexPair(i0, i0-1), iHi_1[0]  ) );
                //serr<<" Add pair H("<<i0<<","<<i0-1<<")"<<sendl;
                // compute bloc (i0,i0-1)
                Minv.asub((i0-1),(i0),bsize,bsize) = -lambda[i0-1] * Minv.asub((i0  ),(i0  ),bsize,bsize);
            }

        }
        // compute bloc (i0-1,i0-1)
        Minv.asub((i0-1),(i0-1),bsize,bsize) = alpha_inv[i0-1] - lambda[i0-1]*Minv.asub((i0  ),(i0-1),bsize,bsize);

        if(subpartSolve.getValue() )
        {
            SubMatrix iHi; // bizarre: pb compilation avec SubMatrix nHn_1 = B[i] *alpha_inv[i];
            my_identity(iHi, bsize);
            H.insert( make_pair(  IndexPair(i0-1, i0-1), iHi  ) );
            //serr<<" Add pair ("<<i0-1<<","<<i0-1<<")"<<sendl;
        }

        ++nBlockComputedMinv[i0-1];
        --i0;
    }

    //serr<<"here i0 ="<<i0<<" should be equal to i ="<<i<<sendl;

    //2. all the block on the lines of block i between the diagonal and the block j are computed
    int j0 = i-nBlockComputedMinv[i];


    /////////////// ADD : Calcul pour faire du partial_solve //////////
    SubMatrix iHj ;
    if(subpartSolve.getValue() )
    {

        //if (i<current_block){
        //	current_block=i;
        //	first_block=i;
        //	}

        H_it = H.find( IndexPair(i0,j0+1) );
        //serr<<" find pair ("<<i<<","<<j0+1<<")"<<sendl;

        if (H_it == H.end()) // ? si jamais l'élément qu'on cherche est justement H.end() ??
        {
            my_identity(iHj, bsize);
            if (i0!=j0+1)
                serr<<"WARNING !! element("<<i0<<","<<j0+1<<") not found : nBlockComputedMinv[i] = "<<nBlockComputedMinv[i]<<sendl;
        }
        else
        {
            //serr<<"element("<<i0<<","<<j0+1<<")  found )!"<<sendl;
            iHj = H_it->second;
        }

    }
    /////////////////////////////////////////////////////////////////////

    while (j0 >= j)
    {
        // compute bloc (i0,j0)
        Minv.asub((i0  ),(j0  ),bsize,bsize) = Minv.asub((i0  ),(j0+1),bsize,bsize)*(-lambda[j0].t());
        if(subpartSolve.getValue() )
        {
            iHj = iHj * -lambda[j0].t();
            H.insert(make_pair(IndexPair(i0,j0),iHj));
            // compute bloc (i0,j0)
            Minv.asub((j0  ),(i0  ),bsize,bsize) = -lambda[j0]*Minv.asub((j0+1),(i0),bsize,bsize);
            //serr<<" Add pair ("<<i<<","<<j0<<")"<<sendl;
        }
        ++nBlockComputedMinv[i0];
        --j0;
    }
}

template<class Matrix, class Vector>
double BTDLinearSolver<Matrix,Vector>::getMinvElement(int i, int j)
{
    const int bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    if (i < j)
    {
        // lower diagonal
        int t = i; i = j; j = t;
    }
    computeMinvBlock(i/bsize, j/bsize);
    return Minv.element(i,j);
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::solve (Matrix& /*M*/, Vector& x, Vector& b)
{
    const bool verbose  = this->f_verbose.getValue() || this->f_printLog.getValue();

    if( verbose )
    {
        serr<<"BTDLinearSolver, b = "<< b <<sendl;
    }

    //invert(M);

    const int bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    const int nb = b.size() / bsize;
    if (nb == 0) return;

    //if (verbose) sout << "D["<<0<<"] = " << b.asub(0,bsize) << sendl;
    x.asub(0,bsize) = alpha_inv[0] * b.asub(0,bsize);
    //if (verbose) sout << "Y["<<0<<"] = " << x.asub(0,bsize) << sendl;
    for (int i=1; i<nb; ++i)
    {
        //if (verbose) sout << "D["<<i<<"] = " << b.asub(i,bsize) << sendl;
        x.asub(i,bsize) = alpha_inv[i]*(b.asub(i,bsize) - B[i]*x.asub((i-1),bsize));
        //if (verbose) sout << "Y["<<i<<"] = " << x.asub(i,bsize) << sendl;
    }
    //x.asub((nb-1),bsize) = Y.asub((nb-1),bsize);
    //if (verbose) sout << "x["<<nb-1<<"] = " << x.asub((nb-1),bsize) << sendl;
    for (int i=nb-2; i>=0; --i)
    {
        x.asub(i,bsize) /* = Y.asub(i,bsize)- */ -= lambda[i]*x.asub((i+1),bsize);
        //if (verbose) sout << "x["<<i<<"] = " << x.asub(i,bsize) << sendl;
    }

    // x is the solution of the system
    if( verbose )
    {
        serr<<"BTDLinearSolver::solve, solution = "<<x<<sendl;
    }
}

template<class Matrix, class Vector>
bool BTDLinearSolver<Matrix,Vector>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
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
    else if (defaulttype::BaseMatrix* r = result)
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

/////// NEW : partial solve :
// b is accumulated
// db is a sparse vector that is added to b
// partial_x is a sparse vector (with sparse map given) that provide the result of M x = b+db
// Solve Mx=b
// Iin donne un block en entrée (dans rh) => derniers blocks dont on a modifié la valeur: on verifie que cette valeur a réellement changé (TODO: éviter en introduisant un booléen)
// Iout donne les block en sortie (dans result)
// ils sont tous les deux tries en ordre croissant
template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::partial_solve(ListIndex&  Iout, ListIndex&  Iin , bool NewIn)  ///*Matrix& M, Vector& result, Vector& rh, */
{
    std::cerr<<" partial_solve......";



    // debug: test
    if (verification.getValue())
    {
        solve(*this->currentGroup->systemMatrix,*this->currentGroup->systemLHVector, *this->currentGroup->systemRHVector);
        return;
    }
    std::cerr<<" step 0"<<std::endl;


    const int bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());

    std::list<int>::const_iterator block_it;
    //SubMatrix iHj;



    //debug

    if(Iin.size() > 0)
    {
        std::cout<<"partial_solve block (in : "<<*Iin.begin()<<")  OUT : "<<*Iout.begin()<<"current_block (should be equal to in) = "<<current_block<<std::endl;
    }
    else
    {
        std::cout<<"partial_solve block (in is NULL) =>  OUT : "<<*Iout.begin()<<"current_block = "<<current_block<<std::endl;
    }


    std::cerr<<" step 1:"<<std::endl;

    /////////////////////////  step 1 .changement des forces en entrée /////////////////////////
    // debug
    //test_perf.getValue() ||
    bool new_forces = false;
    if(test_perf.getValue() || NewIn)
    {


        //on regarde si la force a changé sur les block en entrée
        // si le block actuel == bock en entrée => on accumule ces forces dans _acc_rh_current_block
        // si le block actuel > block en entrée => pb ne devrait pas arriver... pour des forces actives !
        // si le block actuel < block en entrée => on accumule les déplacements entre le block en entrée et le block actuel	+ on stocke la force actuelle pour qu'elle soit prise en compte lors de la prochaine remontée

        for(block_it=Iin.begin(); block_it!=Iin.end(); block_it++)
        {
            int block = *block_it;

            //// computation of DF
            SubVector DF;
            DF.resize(bsize);
            DF += this->currentGroup->systemRHVector->asub(block,bsize) - _rh_buf.asub(block,bsize);
            _rh_buf.asub(block,bsize) = this->currentGroup->systemRHVector->asub(block,bsize) ;
            ////


            if (DF.norm() > 0.0)
            {

                // debug //
                new_forces = true;
                if (current_block< block)
                {

                    SubVector DU;
                    DU.resize(bsize);
                    DU =  Minv.asub(block,block,bsize,bsize) * DF;


                    //std::cout<<"Vec_df["<<block<<"]"<<Vec_df[block] ;
                    Vec_df[block] += DF;
                    //std::cout<<"Vec_df["<<block<<"] += DF "<<Vec_df[block]<<std::endl;
                    // Un += DUacc
                    //_acc_result.asub(block,bsize)  += DU;		 // NON ! DU n'est ajouté que pour les blocks [current_block block[
                    // dans les calculs ultérieur.. pour les blocks [block N[ le calcul se dans le step 4 avec Vec_df
                    // jusqu'�  ce que current_block== block dans ce cas, DF étant déj�  dans this->currentGroup->systemRHVector->asub(block,bsize) il est définitivement pris en compte
                    //std::cout<<"la force sur le block en entrée vient du block "<<block<<" et le block courant est"<<current_block<<" ... on remonte le déplacement engendré "<<DU<<std::endl;
                    while( block > current_block)
                    {
                        block--;
                        // DUacc = Hn,n+1 * DUacc
                        DU = -(lambda[block]*DU);

                        // Un += DUacc
                        _acc_result.asub(block,bsize)  += DU;

                    }
                }
                else
                {

                    if (current_block > block)
                        serr<<"WARNING step1 forces en entrée: current_block= "<<current_block<<" should be inferior or equal to  block= "<<block<<" problem with sort in Iin"<<sendl;
                    else
                    {
                        //std::cout<<"la force sur le block en entrée vient du block "<<block<<" et le block courant est"<<current_block<<" ajout �  _acc_rh_current_block"<<std::endl;
                        _acc_rh_current_block +=  DF;  // current_block==block
                    }
                    /*
                     if(current_block == block)
                     my_identity(iHj, bsize);
                     else
                     {
                     H_it = H.find( IndexPair(current_block,block) );
                     iHj=H_it->second;
                     if (H_it == H.end())
                     {
                     my_identity(iHj, bsize);
                     serr<<"WARNING !! element("<<current_block<<","<<block<<") not found "<<sendl;
                     }
                     }
                     */
                }
            }
        }
    }


    if (NewIn && !new_forces)
        std::cout<<"problem : newIn is true but should be false"<<std::endl;

    // debug
    /*
    if (new_forces)
        std::cout<<"Nouvelles forces détectées et ajoutées"<<std::endl;
    */



    // accumulate DF jusqu'au block d'ordre le plus élevé dans Iout
    // on accumule les forces en parcourant la structure par ordre croissant
    // si la valeur max du "out" est plus petite que la valeur du block courant, c'est qu'on a fini de parcourir la strucure => on remonte jusqu'�  "first_block" (pour l'instant, jusqu'�  0 pour debug)

    int block_out = *Iout.begin();

    std::cerr<<" step 2:"<<std::endl;
    ///////////////////////// step2 parcours de la structure pour descendre les déplacements	/////////////////////////
    if (block_out< current_block)
    {

        //debug
        //std::cout<<" on remonte la structure : block_out= "<<block_out<<"  current_block = "<<current_block<<std::endl;

        //// on inverse le dernier block
        //debug
        //std::cout<<"Un = Kinv(n,n)*(accF + Fn) // accF="<<_acc_rh_current_block<<"   - Fn= "<< this->currentGroup->systemRHVector->asub(current_block,bsize)<<std::endl;
        /// Un = Kinv(n,n)*(accF + Fn)

        //_acc_result.asub(current_block,bsize) =  Minv.asub(current_block,current_block*bsize,bsize,bsize) * (  _acc_rh_current_block +  this->currentGroup->systemRHVector->asub(current_block,bsize) );

        /// Uacc = Kinv(n,n) * (accF+ Fn)
        _acc_lh_current_block =  Minv.asub(current_block,current_block,bsize,bsize) *  this->currentGroup->systemRHVector->asub(current_block,bsize);
        Vec_df[ current_block ] =  this->currentGroup->systemRHVector->asub(current_block,bsize);
        //debug
        //std::cout<<"Uacc = Kinv("<<current_block<<","<<current_block<<")*Fn = "<<_acc_lh_current_block<<std::endl;




        while (current_block> 0)
        {
            current_block--;
            //std::cout<<"descente des déplacements  : current_block = "<<current_block;
            // Uacc += Hn,n+1 * Uacc
            _acc_lh_current_block = -(lambda[current_block]*_acc_lh_current_block);

            // Un = Uacc
            _acc_result.asub(current_block,bsize)  = _acc_lh_current_block;

            // debug
            SubVector Fn;
            Fn =this->currentGroup->systemRHVector->asub(current_block,bsize);
            if (Fn.norm()>0.0)
            {
                Vec_df[ current_block ] =  this->currentGroup->systemRHVector->asub(current_block,bsize);
                //std::cout<<"non null force detected on block "<<current_block<<" : Fn= "<< Fn;
                // Uacc += Kinv* Fn
                _acc_lh_current_block += Minv.asub(current_block,current_block,bsize,bsize) * this->currentGroup->systemRHVector->asub(current_block,bsize) ;
            }


            //std::cout<<std::endl;



        }


        //debug
        //std::cout<<"VERIFY : current_block = "<<current_block<<"  must be 0"<<std::endl;

        //facc=f0;
        _acc_rh_current_block = this->currentGroup->systemRHVector->asub(0,bsize);


        // debug
        SubVector DF;
        DF = Vec_df[0];
        if (DF.norm()> 0.0)
            serr<<"WARNING: Vec_df added on block 0... strange..."<<sendl;


        //_acc_result.asub(0, bsize) += alpha_inv[0] * this->currentGroup->systemRHVector->asub(0,bsize);
//			_rh_buf.asub(0,bsize)  =  this->currentGroup->systemRHVector->asub(0,bsize);

        // accumulation of right hand term is reinitialized
//			_acc_rh_current_block= this->currentGroup->systemRHVector->asub(0,bsize);
    }

    std::cerr<<" step 3:"<<std::endl;

    ///////////////////////// step3 parcours de la structure pour remonter les forces /////////////////////////
    while(current_block<block_out)
    {
        //std::cout<<"remontée des forces  : current_block = "<<current_block<<std::endl;


        // Fbuf = Fn
        //serr<<"Fbuf = Fn"<<sendl;
        // la contribution du block [current_block+1] est prise en compte dans le mouvement actuel : ne sert �  rien ?? = _rh_buf n'est utilisé que pour calculer DF
        //_rh_buf.asub((current_block+1),bsize)  =  this->currentGroup->systemRHVector->asub((current_block+1),bsize) ;

        // Facc = Hn+1,n * Facc
        //serr<<"Facc = Hn+1,n * Facc"<<sendl;
        // on accumule les forces le long de la structure
        /*
        H_it = H.find( IndexPair(current_block+1,current_block) );
        if (H_it==H.end())
        {
                            serr<<"WARNING : H["<<current_block+1<<"]["<<current_block<<"] not found"<<sendl;
        }
        iHj=H_it->second;
        // debug
        Vector test;
        test = _acc_rh_current_block;
        _acc_rh_current_block = iHj * _acc_rh_current_block;
        test = -lambda[current_block].t() * test;

        test -= _acc_rh_current_block;

        if (test.norm()>0.0000000001*_acc_rh_current_block.norm())
        {
                            serr<<"WARNING matrix iHj = \n"<<iHj<<"\n and lambda["<<current_block<<"].t() =\n"<<lambda[current_block].t()<<"\n are not equal !!!"<<sendl;

        }
        */

        _acc_rh_current_block = -(lambda[current_block].t() * _acc_rh_current_block);

        current_block++;

        // debug: Facc+=Fn
        SubVector subV;
        subV =  this->currentGroup->systemRHVector->asub(current_block,bsize);
        _acc_rh_current_block += subV;
        //std::cout<<"step3 : Facc+= F["<<current_block<<"] : result : Facc ="<<_acc_rh_current_block<<std::endl;

        // df of current block is now included in _acc_rh_current_block
        Vec_df[current_block] = 0;
        //std::cout<<"Vec_df["<<current_block<<"] is set to zero: "<< Vec_df[current_block] <<std::endl;

    }



    ///////////////////////// now current_block == block_out : on calcule le déplacement engendré ////////
    //std::cout<<"VERIFY : current_block = "<<current_block<<"  must be equal to block_out :"<<block_out<<std::endl;


    //debug:
    //bool show_result = false;


    std::cerr<<" step 4:"<<std::endl;
    ////////////////////////// step 4 on calcule le déplacement engendré sur les blocks en sortie ////////////////////////

    for(block_it=Iout.begin(); block_it!=Iout.end(); block_it++)
    {
        int block = *block_it;
        // debug
        if (current_block>block)
            serr<<"WARNING : step 4 : blocks en sortie : current_block= "<<current_block<<" must be inferior or equal to  block= "<<block<<" problem with sort in Iout"<<sendl;

        SubVector LH_block;
        LH_block.resize(bsize);

        // un = Forces from
        SubVector PreviousU; // displacement of LH_block due to forces from on other blocks > block (from step 2)
        PreviousU =  _acc_result.asub(block,bsize);
        LH_block = Minv.asub( block, current_block *bsize,bsize,bsize) * _acc_rh_current_block + PreviousU;



        for (int b=current_block; b<block; b++)
        {
            SubVector DF ;
            DF = Vec_df[b+1];
            if (DF.norm())
            {
                //std::cout<<"step 4. Vec_df["<<b+1<<"] in NOT 0: "<<DF<<"   -> calcul du déplacement sur "<<block<<std::endl;
                LH_block += Minv.asub( block, (b+1),bsize,bsize) * DF;
            }
            else
            {
                //std::cout<<"step4. Vec_df["<<b+1<<"] is null  :"<<DF<<std::endl;
            }
        }

        /*
        if (LH_block.norm()>0.0)
        {
            show_result=true;
            std::cout<< " LH_block ["<<block<<"] = "<<LH_block<<" previousU = "<< PreviousU <<" _acc_rh_current_block = "<<_acc_rh_current_block<<std::endl;
        }
        else
        {
            std::cout<< " LH_block ["<<block<<"] is null "<<std::endl;

        }
        */


        if (verification.getValue())
        {
            SubVector LH_block2;
            LH_block2.resize(bsize);
            LH_block2 = this->currentGroup->systemLHVector->asub(block,bsize);
            //std::cout<< " solution ["<<block<<"] = "<<LH_block2<<std::endl;

            SubVector delta_result ;
            delta_result= LH_block - LH_block2;

            if (delta_result.norm() > 0.0001 * LH_block.norm() )
            {
                std::cout<<"++++++++++++++++++++++++++++++++ Problem : delta_result = "<<delta_result<<" +++++++++++++++++++++++++++++++++"<<std::endl;
                // pour faire un seg fault:
                delta_result +=  Minv.asub(0, 0,bsize+1,bsize) *delta_result ;


            }
        }


        // apply the result on "this->currentGroup->systemLHVector"

        this->currentGroup->systemLHVector->asub(block,bsize) = LH_block;



    }




    std::cerr<<"end "<<std::endl;

}





template<class Matrix, class Vector>
template<class RMatrix, class JMatrix>
bool BTDLinearSolver<Matrix,Vector>::addJMInvJt(RMatrix& result, JMatrix& J, double fact)
{
    //const int Jrows = J.rowSize();
    const unsigned int Jcols = J.colSize();
    if (Jcols != Minv.rowSize())
    {
        serr << "BTDLinearSolver::addJMInvJt ERROR: incompatible J matrix size." << sendl;
        return false;
    }


#if 0
// WARNING !!!
    //Getting all elements of Minv modifies the obtained Matrix "result"!!
    // It seems that result is computed more accurately.
    // There is a BUG to find here...
    if (!problem.getValue())
    {
        for  (int mr=0; mr<Minv.rowSize(); mr++)
        {
            for (int mc=0; mc<Minv.colSize(); mc++)
            {
                getMinvElement(mr,mc);
            }
        }
    }
////////////////////////////////////////////
#endif
    if (f_verbose.getValue())
    {
// debug christian: print of the inverse matrix:
        sout<< "C = ["<<sendl;
        for  (unsigned int mr=0; mr<Minv.rowSize(); mr++)
        {
            sout<<" "<<sendl;
            for (unsigned int mc=0; mc<Minv.colSize(); mc++)
            {
                sout<<" "<< getMinvElement(mr,mc);
            }
        }
        sout<< "];"<<sendl;

// debug christian: print of matrix J:
        sout<< "J = ["<<sendl;
        for  (unsigned int jr=0; jr<J.rowSize(); jr++)
        {
            sout<<" "<<sendl;
            for (unsigned int jc=0; jc<J.colSize(); jc++)
            {
                sout<<" "<< J.element(jr, jc) ;
            }
        }
        sout<< "];"<<sendl;
    }


    const typename JMatrix::LineConstIterator jitend = J.end();
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != jitend; ++jit1)
    {
        int row1 = jit1->first;
        for (typename JMatrix::LineConstIterator jit2 = jit1; jit2 != jitend; ++jit2)
        {
            int row2 = jit2->first;
            double acc = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(), i1end = jit1->second.end(); i1 != i1end; ++i1)
            {
                int col1 = i1->first;
                double val1 = i1->second;
                for (typename JMatrix::LElementConstIterator i2 = jit2->second.begin(), i2end = jit2->second.end(); i2 != i2end; ++i2)
                {
                    int col2 = i2->first;
                    double val2 = i2->second;
                    acc += val1 * getMinvElement(col1,col2) * val2;
                }
            }
            //sout << "W("<<row1<<","<<row2<<") += "<<acc<<" * "<<fact<<sendl;
            acc *= fact;
            result.add(row1,row2,acc);
            if (row1!=row2)
                result.add(row2,row1,acc);
        }
    }
    return true;
}




} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
