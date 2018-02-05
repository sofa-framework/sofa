/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_DEFAULTMULTIMATRIXACCESSOR_H
#define SOFA_COMPONENT_LINEARSOLVER_DEFAULTMULTIMATRIXACCESSOR_H
#include "config.h"

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/objectmodel/Data.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace linearsolver
{

/* DefaultMultiMatrixAccessor is a simplest class managing the global matrix setup.
 * This class allow only cases where there are several Mechanical State on simulation scene
 * and interactions between them.
 *
 * CRSMultiMatrixAccessor is a more powerfull class managing the global system matrix.
 * This class allow the scene where there are mappings so mapped Mechanical State. It compute
 * The contribution of stiffness on mapped Mechanical State to the root State related by mapping.
 *
 * see Sofa/doc/modules/linearsolver/LinearSolver.tex
 * see all scenes in example/Component/linearsolver/MatrixContribution*  for tests
 *
 * */




class SOFA_BASE_LINEAR_SOLVER_API DefaultMultiMatrixAccessor : public sofa::core::behavior::MultiMatrixAccessor
{
public:
    DefaultMultiMatrixAccessor();
    virtual ~DefaultMultiMatrixAccessor();

    virtual void clear();

    // setting the global matrix for the system. Its size must have the sum of all real Mechanical state
    virtual void setGlobalMatrix(defaulttype::BaseMatrix* matrix);

    // When a real MS is visited by the visitor, it must be registed in a local data here (realStateOffsets)
    // the global size of the system must be ajusted.
    virtual void addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);

    // When a mapping is visited by the visitor, satisfying that is a mechanical mapping
    // and having implemented getJ, this mapping must be registed in a local data here (mappingList)
    virtual void addMechanicalMapping(sofa::core::BaseMapping* mapping);

    //do nothing for instance
    virtual void addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);

    //Read all Real Mechanical State
    virtual void setupMatrices();

    //give the sum of size of all Real Mechanical State in ordre to set the global matrix dimension
    virtual int getGlobalDimension() const;

    //give position in global matrix of the blog related to a given Mechanical State
    virtual int getGlobalOffset(const sofa::core::behavior::BaseMechanicalState* mstate) const;

    //give the Matrix Reference (Matrix and Offset) related to a given Mechanical State
    virtual MatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;

    //give the Matrix Reference (Matrix and Offset) related to a interactionForceField (between 2 Mechanical State)
    virtual InteractionMatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const;

    //Compute the global system matrix
    //If there are no mapping, do nothing
    //If there are mappings, compute the contribution
    virtual void computeGlobalMatrix();


    //Matrix creating is only call when there are mapped state,
    //the stiffness and interaction stiffness of this state couldn't directly described on the principal matrix
    //then it demande to create a new matrix
    static defaulttype::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2);

protected:

    defaulttype::BaseMatrix* globalMatrix;
    unsigned int globalDim;

    //           case1                                           case2
    //      |               |                                  |       |
    //     MS1             MS2                                MS1     MS2
    //      |               |                                 /      /
    //     mapA            mapB                             map   Inter
    //      |               |                                 \   /
    //     MS3 ---Inter--  MS4                                MS3/
    //
    //    K11 += JAt * K33 * JA                         K11 += Jt * K33 * J
    //    K22 += JBt * K44 * JB                         K12 += Jt * I32
    //    K12 += JAt * I34 * JB                         K21 +=      I32 * J
    //    K21 += JBt * I43 * JA
    //
    // using matrix buffer in case of interaction between mapped model

    //map used for real mechanical state (non-mapped only)
    std::map< const sofa::core::behavior::BaseMechanicalState*, int > realStateOffsets;

    //map used only for mapped mechanical state
    //a mapped state is added here if and only if its stiffness matrix is guessed by other component (forcefield)
    //by method "getMatrix" in order to fill its matrix
    mutable std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* > mappedMatrices;

    //The data structure included mapped and on mapped state, the diagonal stiffness bloc and interaction stiffnessbloc
    mutable std::map< const sofa::core::behavior::BaseMechanicalState*, MatrixRef > diagonalStiffnessBloc;//todo remove
    mutable std::map< std::pair<const sofa::core::behavior::BaseMechanicalState*, const sofa::core::behavior::BaseMechanicalState*>, InteractionMatrixRef > interactionStiffnessBloc;//todo remove

    //The list of validated mapping in the order of visitor, to be read in the inverted direction for propagation contribution
    std::vector<sofa::core::BaseMapping*> mappingList;
};



#ifdef SOFA_SUPPORT_CRS_MATRIX
//TODO separating in other file
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* CRSMultiMatrixAccessor do the same thing as DefaultMultiMatrixAccessor but optimal.
 * The different is instead of creating a standard full matrix for mapped and mapping, CRSMultiMatrixAccessor works
 * with CompressedRowSparseMatrix.
 *
 * To be able to creat  CompressedRowSparseMatrix, it is needle to know about block format of relied to the
 * size of DOF of mapped Mechanical state and input-output Mechanical State of the mapping
 * */
class SOFA_BASE_LINEAR_SOLVER_API CRSMultiMatrixAccessor : public DefaultMultiMatrixAccessor
{
public:
    CRSMultiMatrixAccessor() : DefaultMultiMatrixAccessor() {}
    ~CRSMultiMatrixAccessor() {	this->clear();}

    virtual void addMechanicalMapping(sofa::core::BaseMapping* mapping);

    //Creating the stiffness matrix for pair of Mechanical State when they are not all real state
    static defaulttype::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2);

    //Compute the contribution of all new created matrix to the the global system matrix
    virtual void computeGlobalMatrix();
};




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////  SWITCH COMPILATION PROBLEM TO RUNTIME PROBLEM  /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<int blocRsize, int blocCsize, class elementType>
inline defaulttype::BaseMatrix* createBlocSparseMatrixT(int nbRowBloc, int nbColBloc, bool _debug)
{
    typedef CompressedRowSparseMatrix< defaulttype::Mat<blocRsize, blocCsize, elementType> > BlocMatrix;
    BlocMatrix* m =	new BlocMatrix;
    m->resizeBloc(nbRowBloc,nbColBloc);
    if(_debug)
        std::cout<<"				++createBlocSparseMatrix : _"<< nbRowBloc <<"x"<<nbColBloc<<"_ of blocs _["<<blocRsize<<"x"<<blocCsize<<"]"<<std::endl;
    return m;
}

template<int blocRsize, int blocCsize>
inline defaulttype::BaseMatrix* createBlocSparseMatrixTReal(int elementSize, int nbRowBloc, int nbColBloc, bool _debug)
{
    switch(elementSize)
    {
    case sizeof(float) : return createBlocSparseMatrixT<blocRsize,blocCsize, float>(nbRowBloc,nbColBloc, _debug);
    case sizeof(double): return createBlocSparseMatrixT<blocRsize,blocCsize,double>(nbRowBloc,nbColBloc, _debug);
    default            : return createBlocSparseMatrixT<blocRsize,blocCsize, SReal>(nbRowBloc,nbColBloc, _debug);
    }
}

template<int blocRsize>
inline defaulttype::BaseMatrix* createBlocSparseMatrixTRow(int blocCsize, int elementSize, int nbRowBloc, int nbColBloc, bool _debug)
{
    switch(blocCsize)
    {
    case 1: return createBlocSparseMatrixTReal<blocRsize,1>(elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 2: return createBlocSparseMatrixTReal<blocRsize,2>(elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 3: return createBlocSparseMatrixTReal<blocRsize,3>(elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 4: return createBlocSparseMatrixTReal<blocRsize,4>(elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 5: return createBlocSparseMatrixTReal<blocRsize,5>(elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 6: return createBlocSparseMatrixTReal<blocRsize,6>(elementSize  ,nbRowBloc,nbColBloc, _debug);
    default: return createBlocSparseMatrixTReal<blocRsize,1>(elementSize  ,nbRowBloc,nbColBloc, _debug);
    }
}

inline defaulttype::BaseMatrix* createBlocSparseMatrix(int blocRsize, int blocCsize, int elementSize, int nbRowBloc, int nbColBloc, bool _debug=false)
{
    switch(blocRsize)
    {
    case 1: return createBlocSparseMatrixTRow<1>(blocCsize, elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 2: return createBlocSparseMatrixTRow<2>(blocCsize, elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 3: return createBlocSparseMatrixTRow<3>(blocCsize, elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 4: return createBlocSparseMatrixTRow<4>(blocCsize, elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 5: return createBlocSparseMatrixTRow<5>(blocCsize, elementSize  ,nbRowBloc,nbColBloc, _debug);
    case 6: return createBlocSparseMatrixTRow<6>(blocCsize, elementSize  ,nbRowBloc,nbColBloc, _debug);
    default: return createBlocSparseMatrixTRow<1>(blocCsize, elementSize  ,nbRowBloc,nbColBloc, _debug);
    }
}


template<int JblocRsize, int JblocCsize, int MblocCsize, class JelementType, class MelementType>
inline bool opAddMulJTM_TBloc(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* J, defaulttype::BaseMatrix* stiffMatrix2, int offsetRow, int offsetCol, bool _debug)
{
    // Notice : in case where stiffMatrix2 are self-stiffness matrix,
    // we have JblocRsize = MblocCsize
    typedef defaulttype::Mat<JblocRsize, JblocCsize, JelementType> JBloc;
    typedef CompressedRowSparseMatrix<JBloc>                       JMatrix;
    typedef typename JMatrix::ColBlockConstIterator                JBColConstIterator;
    typedef typename JMatrix::BlockConstAccessor                   JBlocConstAccessor;

    typedef defaulttype::Mat<JblocRsize, MblocCsize, MelementType> MBloc;
    typedef CompressedRowSparseMatrix<MBloc>                       MMatrix;
    typedef typename MMatrix::ColBlockConstIterator                MBColConstIterator;
    typedef typename MMatrix::BlockConstAccessor                   MBlocConstAccessor;

    typedef defaulttype::Mat<JblocCsize, MblocCsize, MelementType> OutBloc;
    typedef CompressedRowSparseMatrix<OutBloc>                     OutMatrix;

    JMatrix* Jmatrix = dynamic_cast<JMatrix*>(J);
    MMatrix* Mmatrix = dynamic_cast<MMatrix*>(stiffMatrix2);
    OutMatrix* Outmatrix = dynamic_cast<OutMatrix*>(out);

    if (!Jmatrix || !Mmatrix) return false;
    if(_debug)
    {
        std::cout<<"    **opAddMulJTM_TBloc: === "
                <<"K1["<< out->rowSize() <<"x"<< out->colSize()<< "]:_"
                << out->bRowSize() << "x"<< out->bColSize() <<"_ of blocs _["
                << out->getBlockRows() << "x"<< out->getBlockCols() <<"]_"
                <<"    +=    Jt["<< J->colSize()<<"x"<< J->rowSize() << "]:_"
                << J->bColSize() << "x"<<  J->bRowSize()<<"_ of blocs _["
                <<J->getBlockCols() << "x"<< J->getBlockRows()  <<"]_"
                <<"*K2["<< stiffMatrix2->rowSize() <<"x"<< stiffMatrix2->colSize()<< "]:_"
                << stiffMatrix2->bRowSize() << "x"<< stiffMatrix2->bColSize() <<"_ of blocs _["
                << stiffMatrix2->getBlockRows() << "x"<< stiffMatrix2->getBlockCols() <<"]_"
                <<"   =============================="<<std::endl<<std::endl;
    }

    // We can compute stiffMatrix 1 += Jt M
    if (!Outmatrix || (offsetRow % JblocCsize != 0) || (offsetCol % MblocCsize != 0))
    {
        // optimized multiplication, but generic output

        JBloc Jblocbuffer;
        MBloc Mblocbuffer;


        for (int JBRowIndex = 0; JBRowIndex < Jmatrix->nBlocRow; JBRowIndex++)
        {
            //through X, must take each row  (but test can be added)
            for (JBColConstIterator JBColIter = Jmatrix->bRowBegin(JBRowIndex); JBColIter < Jmatrix->bRowEnd(JBRowIndex); JBColIter++)
            {
                //take non zero blocks in row, determines the row in K)
                JBlocConstAccessor Jbloc = JBColIter.bloc();
                const JBloc& JblocData = *(const JBloc*)Jbloc.elements(Jblocbuffer.ptr());
                int JBColIndex = Jbloc.getCol();
                for (MBColConstIterator MBColIter = Mmatrix->bRowBegin(JBRowIndex); MBColIter < Mmatrix->bRowEnd(JBRowIndex); MBColIter++)
                {
                    MBlocConstAccessor Mbloc = MBColIter.bloc();
                    const MBloc& MblocData = *(const MBloc*)Mbloc.elements(Mblocbuffer.ptr());
                    int MBColIndex = Mbloc.getCol();
                    OutBloc tempBlockData(0.0);
                    //multiply the block, could be done more efficiently
                    for (int i = 0; i < JblocCsize; i++)
                    {
                        for (int j = 0; j < MblocCsize; j++)
                        {
                            for (int k = 0; k < JblocRsize; k++)
                            {
                                tempBlockData(i,j) += (MelementType)JblocData(k,i) * MblocData(k,j);
                                out->add(offsetRow + JblocRsize*JBColIndex+i, offsetCol + MblocCsize*MBColIndex+j, tempBlockData(i,j));
                                if(_debug)
                                {
                                    std::cout<<" (i."<<offsetRow + JblocRsize*JBColIndex+i<<","
                                            <<"j."<<offsetCol + MblocCsize*MBColIndex+j<<")"
                                            <<"    JBRowIndex:" <<JBRowIndex <<" JBColIndex:" <<JBColIndex
                                            <<"      i:" <<i <<" j:"<<j <<"  k:" <<k <<std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // optimized multiplication and output
        int offsetBRow = offsetRow / JblocCsize;
        int offsetBCol = offsetCol / MblocCsize;

        JBloc Jblocbuffer;
        MBloc Mblocbuffer;

        for (int JBRowIndex = 0; JBRowIndex < Jmatrix->nBlocRow; JBRowIndex++)
        {
            //through X, must take each row  (but test can be added)
            for (JBColConstIterator JBColIter = Jmatrix->bRowBegin(JBRowIndex); JBColIter < Jmatrix->bRowEnd(JBRowIndex); JBColIter++)
            {
                //take non zero blocks in row, determines the row in K)
                JBlocConstAccessor Jbloc = JBColIter.bloc();
                const JBloc& JblocData = *(const JBloc*)Jbloc.elements(Jblocbuffer.ptr());
                int JBColIndex = Jbloc.getCol();
                for (MBColConstIterator MBColIter = Mmatrix->bRowBegin(JBRowIndex); MBColIter < Mmatrix->bRowEnd(JBRowIndex); MBColIter++)
                {
                    MBlocConstAccessor Mbloc = MBColIter.bloc();
                    const MBloc& MblocData = *(const MBloc*)Mbloc.elements(Mblocbuffer.ptr());
                    int MBColIndex = Mbloc.getCol();
                    OutBloc tempBlockData(0.0);
                    //multiply the block, could be done more efficiently
                    for (int i = 0; i < JblocCsize; i++)
                        for (int j = 0; j < MblocCsize; j++)
                            for (int k = 0; k < JblocRsize; k++)
                            {
                                tempBlockData(i,j) += (MelementType)JblocData(k,i) * MblocData(k,j);
                                if(_debug)
                                {
                                    std::cout<<" (bI."<<offsetBRow + JBColIndex<<","
                                            <<"bJ."<<offsetBCol + MBColIndex<<")"
                                            <<"    JBRowIndex:" <<JBRowIndex <<" JBColIndex:" <<JBColIndex
                                            <<"      i:" <<i <<" j:"<<j <<"  k:" <<k <<std::endl;
                                }
                            }
                    Outmatrix->blocAdd(offsetBRow + JBColIndex,offsetBCol + MBColIndex, tempBlockData.ptr());
                }
            }
        }
    }



    return true;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int JblocRsize, int JblocCsize, int MblocCsize, class JelementType>
inline bool opAddMulJTM_T4(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* J, defaulttype::BaseMatrix* stiffMatrix2,
        int offsetRow, int offsetCol, int MelementSize, bool _debug)
{
    switch(MelementSize)
    {
    case sizeof(float) :
        return opAddMulJTM_TBloc<JblocRsize, JblocCsize, MblocCsize, JelementType, float>(out,J,stiffMatrix2,offsetRow,offsetCol, _debug);
    case sizeof(double):
        return opAddMulJTM_TBloc<JblocRsize, JblocCsize, MblocCsize, JelementType, double>(out,J,stiffMatrix2,offsetRow,offsetCol, _debug);
    default            :
        return opAddMulJTM_TBloc<JblocRsize, JblocCsize, MblocCsize, JelementType, SReal>(out,J,stiffMatrix2,offsetRow,offsetCol, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
template<int JblocRsize, int JblocCsize, int MblocCsize>
inline bool opAddMulJTM_T3(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* J, defaulttype::BaseMatrix* stiffMatrix2,
        int offsetRow, int offsetCol, int JelementSize, int MelementSize, bool _debug)
{
    switch(JelementSize)
    {
    case sizeof(float) :
        return opAddMulJTM_T4<JblocRsize, JblocCsize, MblocCsize, float>(out,J,stiffMatrix2,offsetRow,offsetCol,MelementSize, _debug);
    case sizeof(double):
        return opAddMulJTM_T4<JblocRsize, JblocCsize, MblocCsize, double>(out,J,stiffMatrix2,offsetRow,offsetCol,MelementSize, _debug);
    default            :
        return opAddMulJTM_T4<JblocRsize, JblocCsize, MblocCsize, SReal>(out,J,stiffMatrix2,offsetRow,offsetCol,MelementSize, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
template<int JblocRsize, int JblocCsize>
inline bool opAddMulJTM_T2(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* J, defaulttype::BaseMatrix* stiffMatrix2,
        int offsetRow, int offsetCol, int _MblocCsize, int JelementSize, int MelementSize, bool _debug)
{
    switch(_MblocCsize)
    {
    case 1:	return opAddMulJTM_T3<JblocRsize, JblocCsize, 1>(out,J,stiffMatrix2,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 2:	return opAddMulJTM_T3<JblocRsize, JblocCsize, 2>(out,J,stiffMatrix2,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 3:	return opAddMulJTM_T3<JblocRsize, JblocCsize, 3>(out,J,stiffMatrix2,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 4:	return opAddMulJTM_T3<JblocRsize, JblocCsize, 4>(out,J,stiffMatrix2,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 5:	return opAddMulJTM_T3<JblocRsize, JblocCsize, 5>(out,J,stiffMatrix2,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 6:	return opAddMulJTM_T3<JblocRsize, JblocCsize, 6>(out,J,stiffMatrix2,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    default: return opAddMulJTM_T3<JblocRsize, JblocCsize, 1>(out,J,stiffMatrix2,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
template<int JblocRsize>
inline bool opAddMulJTM_T1(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* J, defaulttype::BaseMatrix* stiffMatrix2,
        int offsetRow, int offsetCol, int _JblocCsize, int _MblocCsize, int JelementSize, int MelementSize, bool _debug)
{
    switch(_JblocCsize)
    {
    case 1:	return opAddMulJTM_T2<JblocRsize, 1>(out,J,stiffMatrix2,offsetRow,offsetCol, _MblocCsize, JelementSize, MelementSize, _debug);
    case 2:	return opAddMulJTM_T2<JblocRsize, 2>(out,J,stiffMatrix2,offsetRow,offsetCol, _MblocCsize, JelementSize, MelementSize, _debug);
    case 3:	return opAddMulJTM_T2<JblocRsize, 3>(out,J,stiffMatrix2,offsetRow,offsetCol, _MblocCsize, JelementSize, MelementSize, _debug);
    case 4:	return opAddMulJTM_T2<JblocRsize, 4>(out,J,stiffMatrix2,offsetRow,offsetCol, _MblocCsize, JelementSize, MelementSize, _debug);
    case 5:	return opAddMulJTM_T2<JblocRsize, 5>(out,J,stiffMatrix2,offsetRow,offsetCol, _MblocCsize, JelementSize, MelementSize, _debug);
    case 6:	return opAddMulJTM_T2<JblocRsize, 6>(out,J,stiffMatrix2,offsetRow,offsetCol, _MblocCsize, JelementSize, MelementSize, _debug);
    default: return opAddMulJTM_T2<JblocRsize, 1>(out,J,stiffMatrix2,offsetRow,offsetCol, _MblocCsize, JelementSize, MelementSize, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
inline bool opAddMulJTM(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* J, defaulttype::BaseMatrix* stiffMatrix2,
        int offsetRow, int offsetCol, int _JblocRsize, int _JblocCsize, int _MblocCsize, int JelementSize, int MelementSize, bool _debug=false)
{
    switch(_JblocRsize)
    {
    case 1:	return opAddMulJTM_T1<1>(out,J,stiffMatrix2,offsetRow,offsetCol, _JblocCsize, _MblocCsize, JelementSize, MelementSize, _debug);
    case 2:	return opAddMulJTM_T1<2>(out,J,stiffMatrix2,offsetRow,offsetCol, _JblocCsize, _MblocCsize, JelementSize, MelementSize, _debug);
    case 3:	return opAddMulJTM_T1<3>(out,J,stiffMatrix2,offsetRow,offsetCol, _JblocCsize, _MblocCsize, JelementSize, MelementSize, _debug);
    case 4:	return opAddMulJTM_T1<4>(out,J,stiffMatrix2,offsetRow,offsetCol, _JblocCsize, _MblocCsize, JelementSize, MelementSize, _debug);
    case 5:	return opAddMulJTM_T1<5>(out,J,stiffMatrix2,offsetRow,offsetCol, _JblocCsize, _MblocCsize, JelementSize, MelementSize, _debug);
    case 6:	return opAddMulJTM_T1<6>(out,J,stiffMatrix2,offsetRow,offsetCol, _JblocCsize, _MblocCsize, JelementSize, MelementSize, _debug);
    default: return opAddMulJTM_T1<1>(out,J,stiffMatrix2,offsetRow,offsetCol, _JblocCsize, _MblocCsize, JelementSize, MelementSize, _debug);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<int MblocRsize, int MblocCsize, int JblocCsize, class JelementType, class MelementType>
inline bool opAddMulMJ_TBloc(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* stiffMatrix2,  defaulttype::BaseMatrix* J,int offsetRow, int offsetCol, bool _debug)
{
    typedef defaulttype::Mat<MblocCsize, JblocCsize, JelementType> JBloc;
    typedef CompressedRowSparseMatrix<JBloc>                       JMatrix;
    typedef typename JMatrix::ColBlockConstIterator                JBColConstIterator;
    typedef typename JMatrix::BlockConstAccessor                   JBlocConstAccessor;

    typedef defaulttype::Mat<MblocRsize, MblocCsize, MelementType> MBloc;
    typedef CompressedRowSparseMatrix<MBloc>                       MMatrix;
    typedef typename MMatrix::ColBlockConstIterator                MBColConstIterator;
    typedef typename MMatrix::BlockConstAccessor                   MBlocConstAccessor;

    typedef defaulttype::Mat<MblocRsize, JblocCsize, MelementType> OutBloc;
    typedef CompressedRowSparseMatrix<OutBloc>                     OutMatrix;

    JMatrix* Jmatrix = dynamic_cast<JMatrix*>(J);
    MMatrix* Mmatrix = dynamic_cast<MMatrix*>(stiffMatrix2);
    OutMatrix* Outmatrix = dynamic_cast<OutMatrix*>(out);

    if (!Jmatrix || !Mmatrix) return false;

    if(_debug)
    {
        std::cout<<"    **opAddMulMJ_TBloc: === "
                <<"K1["<< out->rowSize() <<"x"<< out->colSize()<< "]:_"
                << out->bRowSize() << "x"<< out->bColSize() <<"_ of blocs _["
                << out->getBlockRows() << "x"<< out->getBlockCols() <<"]_"
                <<"    +=    K2["<< stiffMatrix2->rowSize()       <<"x" << stiffMatrix2->colSize()      <<"]:_"
                <<                  stiffMatrix2->bRowSize()     << "x" << stiffMatrix2->bColSize()     <<"_ of blocs _["
                <<                  stiffMatrix2->getBlockRows() << "x" << stiffMatrix2->getBlockCols() <<"]_"
                <<" * J["<< J->rowSize()     << "x" << J->colSize()      <<"]:_"
                <<          J->bRowSize()    << "x" << J->bColSize()     <<"_ of blocs _["
                <<          J->getBlockRows()<< "x" << J->getBlockCols() <<"]_"
                <<"   =============================="<<std::endl<<std::endl;
    }

    // We can compute stiffMatrix 1 += Jt M
    if (!Outmatrix || (offsetRow % MblocRsize != 0) || (offsetCol % JblocCsize != 0))
    {
        // optimized multiplication, but generic output
        JBloc Jblocbuffer;
        MBloc Mblocbuffer;

        for (int MBRowIndex = 0; MBRowIndex < Mmatrix->nBlocRow; MBRowIndex++)
        {
            //through X, must take each row  (but test can be added)
            for (MBColConstIterator MBColIter = Mmatrix->bRowBegin(MBRowIndex); MBColIter < Mmatrix->bRowEnd(MBRowIndex); MBColIter++)
            {
                //take non zero blocks in row, determines the row in K)
                MBlocConstAccessor Mbloc = MBColIter.bloc();
                const MBloc& MblocData = *(const MBloc*)Mbloc.elements(Mblocbuffer.ptr());
                //int MBColIndex = Mbloc.getCol();
                for (JBColConstIterator JBColIter = Jmatrix->bRowBegin(MBRowIndex); JBColIter < Jmatrix->bRowEnd(MBRowIndex); JBColIter++)
                {
                    JBlocConstAccessor Jbloc = JBColIter.bloc();
                    const JBloc& JblocData = *(const JBloc*)Jbloc.elements(Jblocbuffer.ptr());
                    int JBColIndex = Jbloc.getCol();
                    OutBloc tempBlockData(0.0);
                    //multiply the block, could be done more efficiently
                    for (int i = 0; i < MblocRsize; i++)
                    {
                        for (int j = 0; j < JblocCsize; j++)
                        {
                            for (int k = 0; k < MblocCsize; k++)
                            {
                                tempBlockData(i,j) += MblocData(i,k) * (MelementType)JblocData(k,j);
                                if(_debug)
                                {
                                    std::cout<<" (i."<<offsetRow + MblocRsize*MBRowIndex+i<<","
                                            <<"j."<<offsetCol + JblocCsize * JBColIndex+j<<")"
                                            <<"    MBRowIndex:" <<MBRowIndex <<" JBColIndex:" <<JBColIndex
                                            <<"      i:" <<i <<" j:"<<j <<"  k:" <<k <<std::endl;
                                }


                            }
                            out->add(offsetRow + MblocRsize*MBRowIndex+i, offsetCol + JblocCsize*JBColIndex+j, tempBlockData(i,j));
                        }
                    }
                }
            }
        }
    }
    else
    {
        // optimized multiplication and output
        int offsetBRow = offsetRow / MblocRsize;
        int offsetBCol = offsetCol / JblocCsize;

        JBloc Jblocbuffer;
        MBloc Mblocbuffer;

        for (int MBRowIndex = 0; MBRowIndex < Mmatrix->nBlocRow; MBRowIndex++)
        {
            //through X, must take each row  (but test can be added)
            for (MBColConstIterator MBColIter = Mmatrix->bRowBegin(MBRowIndex); MBColIter < Mmatrix->bRowEnd(MBRowIndex); MBColIter++)
            {
                //take non zero blocks in row, determines the row in K)
                MBlocConstAccessor Mbloc = MBColIter.bloc();
                const MBloc& MblocData = *(const MBloc*)Mbloc.elements(Mblocbuffer.ptr());
                //int MBColIndex = Mbloc.getCol();
                for (JBColConstIterator JBColIter = Jmatrix->bRowBegin(MBRowIndex); JBColIter < Jmatrix->bRowEnd(MBRowIndex); JBColIter++)
                {
                    JBlocConstAccessor Jbloc = JBColIter.bloc();
                    const JBloc& JblocData = *(const JBloc*)Jbloc.elements(Jblocbuffer.ptr());
                    int JBColIndex = Jbloc.getCol();
                    OutBloc tempBlockData(0.0);
                    //multiply the block, could be done more efficiently
                    for (int i = 0; i < MblocRsize; i++)
                        for (int j = 0; j < JblocCsize; j++)
                            for (int k = 0; k < MblocCsize; k++)
                            {
                                if(_debug)
                                {
                                    std::cout<<" (bI."<<offsetBRow + MBRowIndex<<","
                                            <<"bJ."<<offsetBCol + JBColIndex<<")"
                                            <<"    MBRowIndex:" <<MBRowIndex <<" JBColIndex:" <<JBColIndex
                                            <<"      i:" <<i <<" j:"<<j <<"  k:" <<k <<std::endl;
                                }

                                tempBlockData(i,j) += MblocData(i,k) * (MelementType)JblocData(k,j);
                            }
                    Outmatrix->blocAdd(offsetBRow + MBRowIndex,offsetBCol + JBColIndex, tempBlockData.ptr());
                }
            }
        }
    }
    return true;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int MblocRsize, int MblocCsize, int JblocCsize, class JelementType>
inline bool opAddMulMJ_T4(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* stiffMatrix2, defaulttype::BaseMatrix* J,
        int offsetRow, int offsetCol, int MelementSize, bool _debug)
{
    switch(MelementSize)
    {
    case sizeof(float) :
        return opAddMulMJ_TBloc<MblocRsize, MblocCsize, JblocCsize, JelementType, float>(out,stiffMatrix2,J,offsetRow,offsetCol, _debug);
    case sizeof(double):
        return opAddMulMJ_TBloc<MblocRsize, MblocCsize, JblocCsize, JelementType, double>(out,stiffMatrix2,J,offsetRow,offsetCol, _debug);
    default            :
        return opAddMulMJ_TBloc<MblocRsize, MblocCsize, JblocCsize, JelementType, SReal>(out,stiffMatrix2,J,offsetRow,offsetCol, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
template<int MblocRsize, int MblocCsize, int JblocCsize>
inline bool opAddMulMJ_T3(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* stiffMatrix2, defaulttype::BaseMatrix* J,
        int offsetRow, int offsetCol, int JelementSize, int MelementSize, bool _debug)
{
    switch(JelementSize)
    {
    case sizeof(float) :
        return opAddMulMJ_T4<MblocRsize, MblocCsize, JblocCsize, float>(out,stiffMatrix2,J,offsetRow,offsetCol,MelementSize, _debug);
    case sizeof(double):
        return opAddMulMJ_T4<MblocRsize, MblocCsize, JblocCsize, double>(out,stiffMatrix2,J,offsetRow,offsetCol,MelementSize, _debug);
    default            :
        return opAddMulMJ_T4<MblocRsize, MblocCsize, JblocCsize, SReal>(out,stiffMatrix2,J,offsetRow,offsetCol,MelementSize, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
template<int MblocRsize, int MblocCsize>
inline bool opAddMulMJ_T2(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* stiffMatrix2, defaulttype::BaseMatrix* J,
        int offsetRow, int offsetCol, int _JblocCsize, int JelementSize, int MelementSize, bool _debug)
{
    switch(_JblocCsize)
    {
    case 1:	return opAddMulMJ_T3<MblocRsize, MblocCsize, 1>(out,stiffMatrix2,J,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 2:	return opAddMulMJ_T3<MblocRsize, MblocCsize, 2>(out,stiffMatrix2,J,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 3:	return opAddMulMJ_T3<MblocRsize, MblocCsize, 3>(out,stiffMatrix2,J,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 4:	return opAddMulMJ_T3<MblocRsize, MblocCsize, 4>(out,stiffMatrix2,J,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 5:	return opAddMulMJ_T3<MblocRsize, MblocCsize, 5>(out,stiffMatrix2,J,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    case 6:	return opAddMulMJ_T3<MblocRsize, MblocCsize, 6>(out,stiffMatrix2,J,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    default: return opAddMulMJ_T3<MblocRsize, MblocCsize, 1>(out,stiffMatrix2,J,offsetRow,offsetCol, JelementSize, MelementSize, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
template<int MblocRsize>
inline bool opAddMulMJ_T1(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* stiffMatrix2, defaulttype::BaseMatrix* J,
        int offsetRow, int offsetCol, int _MblocCsize, int _JblocCsize, int JelementSize, int MelementSize, bool _debug)
{
    switch(_MblocCsize)
    {
    case 1:	return opAddMulMJ_T2<MblocRsize, 1>(out,stiffMatrix2,J,offsetRow,offsetCol, _JblocCsize, JelementSize, MelementSize, _debug);
    case 2:	return opAddMulMJ_T2<MblocRsize, 2>(out,stiffMatrix2,J,offsetRow,offsetCol, _JblocCsize, JelementSize, MelementSize, _debug);
    case 3:	return opAddMulMJ_T2<MblocRsize, 3>(out,stiffMatrix2,J,offsetRow,offsetCol, _JblocCsize, JelementSize, MelementSize, _debug);
    case 4:	return opAddMulMJ_T2<MblocRsize, 4>(out,stiffMatrix2,J,offsetRow,offsetCol, _JblocCsize, JelementSize, MelementSize, _debug);
    case 5:	return opAddMulMJ_T2<MblocRsize, 5>(out,stiffMatrix2,J,offsetRow,offsetCol, _JblocCsize, JelementSize, MelementSize, _debug);
    case 6:	return opAddMulMJ_T2<MblocRsize, 6>(out,stiffMatrix2,J,offsetRow,offsetCol, _JblocCsize, JelementSize, MelementSize, _debug);
    default: return opAddMulMJ_T2<MblocRsize, 1>(out,stiffMatrix2,J,offsetRow,offsetCol, _JblocCsize, JelementSize, MelementSize, _debug);
    }
}
//-------------------------------------------------------------------------------------------------
inline bool opAddMulMJ(defaulttype::BaseMatrix* out, defaulttype::BaseMatrix* stiffMatrix2, defaulttype::BaseMatrix* J,
        int offsetRow, int offsetCol, int _MblocRsize, int _MblocCsize, int _JblocCsize, int JelementSize, int MelementSize, bool _debug=false)
{
    switch(_MblocRsize)
    {
    case 1:	return opAddMulMJ_T1<1>(out,stiffMatrix2,J,offsetRow,offsetCol, _MblocCsize, _JblocCsize, JelementSize, MelementSize, _debug);
    case 2:	return opAddMulMJ_T1<2>(out,stiffMatrix2,J,offsetRow,offsetCol, _MblocCsize, _JblocCsize, JelementSize, MelementSize, _debug);
    case 3:	return opAddMulMJ_T1<3>(out,stiffMatrix2,J,offsetRow,offsetCol, _MblocCsize, _JblocCsize, JelementSize, MelementSize, _debug);
    case 4:	return opAddMulMJ_T1<4>(out,stiffMatrix2,J,offsetRow,offsetCol, _MblocCsize, _JblocCsize, JelementSize, MelementSize, _debug);
    case 5:	return opAddMulMJ_T1<5>(out,stiffMatrix2,J,offsetRow,offsetCol, _MblocCsize, _JblocCsize, JelementSize, MelementSize, _debug);
    case 6:	return opAddMulMJ_T1<6>(out,stiffMatrix2,J,offsetRow,offsetCol, _MblocCsize, _JblocCsize, JelementSize, MelementSize, _debug);
    default: return opAddMulMJ_T1<1>(out,stiffMatrix2,J,offsetRow,offsetCol, _MblocCsize, _JblocCsize, JelementSize, MelementSize, _debug);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif













} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
