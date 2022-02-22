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
#include <sofa/core/behavior/DefaultMultiMatrixAccessor.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

namespace sofa::core::behavior
{


/* CRSMultiMatrixAccessor do the same thing as DefaultMultiMatrixAccessor but optimal.
 * The different is instead of creating a standard full matrix for mapped and mapping, CRSMultiMatrixAccessor works
 * with linearalgebra::CompressedRowSparseMatrix.
 *
 * To be able to creat  linearalgebra::CompressedRowSparseMatrix, it is needle to know about block format of relied to the
 * size of DOF of mapped Mechanical state and input-output Mechanical State of the mapping
 * */
class SOFA_CORE_API CRSMultiMatrixAccessor : public DefaultMultiMatrixAccessor
{
public:
    CRSMultiMatrixAccessor() : DefaultMultiMatrixAccessor() {}
    ~CRSMultiMatrixAccessor() override {	this->clear();}

    void addMechanicalMapping(sofa::core::BaseMapping* mapping) override;

    //Creating the stiffness matrix for pair of Mechanical State when they are not all real state
    static linearalgebra::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2);
    static linearalgebra::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2, bool doPrintInfo);

    //Compute the contribution of all new created matrix to the global system matrix
    void computeGlobalMatrix() override;
};

template<int blocRsize, int blocCsize, class elementType>
inline linearalgebra::BaseMatrix* createBlocSparseMatrixT(int nbRowBloc, int nbColBloc, bool _debug)
{
    typedef sofa::linearalgebra::CompressedRowSparseMatrix< type::Mat<blocRsize, blocCsize, elementType> > BlocMatrix;
    BlocMatrix* m =	new BlocMatrix;
    m->resizeBloc(nbRowBloc,nbColBloc);
    msg_info_when(_debug, "CRSMultiMatrixAccessor") << "				++createBlocSparseMatrix : _" << nbRowBloc << "x" << nbColBloc << "_ of blocs _[" << blocRsize << "x" << blocCsize << "]";
    return m;
}

template<int blocRsize, int blocCsize>
inline linearalgebra::BaseMatrix* createBlocSparseMatrixTReal(int elementSize, int nbRowBloc, int nbColBloc, bool _debug)
{
    switch(elementSize)
    {
        case sizeof(float) : return createBlocSparseMatrixT<blocRsize,blocCsize, float>(nbRowBloc,nbColBloc, _debug);
        case sizeof(double): return createBlocSparseMatrixT<blocRsize,blocCsize,double>(nbRowBloc,nbColBloc, _debug);
        default            : return createBlocSparseMatrixT<blocRsize,blocCsize, SReal>(nbRowBloc,nbColBloc, _debug);
    }
}

template<int blocRsize>
inline linearalgebra::BaseMatrix* createBlocSparseMatrixTRow(int blocCsize, int elementSize, int nbRowBloc, int nbColBloc, bool _debug)
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

inline linearalgebra::BaseMatrix* createBlocSparseMatrix(int blocRsize, int blocCsize, int elementSize, int nbRowBloc, int nbColBloc, bool _debug=false)
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
inline bool opAddMulJTM_TBloc(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* J, linearalgebra::BaseMatrix* stiffMatrix2, int offsetRow, int offsetCol, bool _debug)
{
    // Notice : in case where stiffMatrix2 are self-stiffness matrix,
    // we have JblocRsize = MblocCsize
    typedef type::Mat<JblocRsize, JblocCsize, JelementType> JBloc;
    typedef linearalgebra::CompressedRowSparseMatrix<JBloc>                       JMatrix;
    typedef typename JMatrix::ColBlockConstIterator                JBColConstIterator;
    typedef typename JMatrix::BlockConstAccessor                   JBlocConstAccessor;

    typedef type::Mat<JblocRsize, MblocCsize, MelementType> MBloc;
    typedef linearalgebra::CompressedRowSparseMatrix<MBloc>                       MMatrix;
    typedef typename MMatrix::ColBlockConstIterator                MBColConstIterator;
    typedef typename MMatrix::BlockConstAccessor                   MBlocConstAccessor;

    typedef type::Mat<JblocCsize, MblocCsize, MelementType> OutBloc;
    typedef linearalgebra::CompressedRowSparseMatrix<OutBloc>                     OutMatrix;

    JMatrix* Jmatrix = dynamic_cast<JMatrix*>(J);
    MMatrix* Mmatrix = dynamic_cast<MMatrix*>(stiffMatrix2);
    OutMatrix* Outmatrix = dynamic_cast<OutMatrix*>(out);

    if (!Jmatrix || !Mmatrix) return false;

    msg_info_when(_debug, "CRSMultiMatrixAccessor") << "    **opAddMulJTM_TBloc: === "
        << "K1[" << out->rowSize() << "x" << out->colSize() << "]:_"
        << out->bRowSize() << "x" << out->bColSize() << "_ of blocs _["
        << out->getBlockRows() << "x" << out->getBlockCols() << "]_"
        << "    +=    Jt[" << J->colSize() << "x" << J->rowSize() << "]:_"
        << J->bColSize() << "x" << J->bRowSize() << "_ of blocs _["
        << J->getBlockCols() << "x" << J->getBlockRows() << "]_"
        << "*K2[" << stiffMatrix2->rowSize() << "x" << stiffMatrix2->colSize() << "]:_"
        << stiffMatrix2->bRowSize() << "x" << stiffMatrix2->bColSize() << "_ of blocs _["
        << stiffMatrix2->getBlockRows() << "x" << stiffMatrix2->getBlockCols() << "]_"
        << "   ==============================";

    // We can compute stiffMatrix 1 += Jt M
    if (!Outmatrix || (offsetRow % JblocCsize != 0) || (offsetCol % MblocCsize != 0))
    {
        // optimized multiplication, but generic output

        JBloc Jblocbuffer;
        MBloc Mblocbuffer;


        for (int JBRowIndex = 0; JBRowIndex < Jmatrix->nBlockRow; JBRowIndex++)
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
                                msg_info_when(_debug, "CRSMultiMatrixAccessor") << " (i." << offsetRow + JblocRsize * JBColIndex + i << ","
                                    << "j." << offsetCol + MblocCsize * MBColIndex + j << ")"
                                    << "    JBRowIndex:" << JBRowIndex << " JBColIndex:" << JBColIndex
                                    << "      i:" << i << " j:" << j << "  k:" << k;
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

        for (int JBRowIndex = 0; JBRowIndex < Jmatrix->nBlockRow; JBRowIndex++)
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
                                msg_info_when(_debug, "CRSMultiMatrixAccessor") << " (bI." << offsetBRow + JBColIndex << ","
                                    << "bJ." << offsetBCol + MBColIndex << ")"
                                    << "    JBRowIndex:" << JBRowIndex << " JBColIndex:" << JBColIndex
                                    << "      i:" << i << " j:" << j << "  k:" << k;
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
inline bool opAddMulJTM_T4(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* J, linearalgebra::BaseMatrix* stiffMatrix2,
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
inline bool opAddMulJTM_T3(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* J, linearalgebra::BaseMatrix* stiffMatrix2,
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
inline bool opAddMulJTM_T2(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* J, linearalgebra::BaseMatrix* stiffMatrix2,
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
inline bool opAddMulJTM_T1(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* J, linearalgebra::BaseMatrix* stiffMatrix2,
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
inline bool opAddMulJTM(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* J, linearalgebra::BaseMatrix* stiffMatrix2,
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
inline bool opAddMulMJ_TBloc(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* stiffMatrix2,  linearalgebra::BaseMatrix* J,int offsetRow, int offsetCol, bool _debug)
{
    typedef type::Mat<MblocCsize, JblocCsize, JelementType> JBloc;
    typedef linearalgebra::CompressedRowSparseMatrix<JBloc>                       JMatrix;
    typedef typename JMatrix::ColBlockConstIterator                JBColConstIterator;
    typedef typename JMatrix::BlockConstAccessor                   JBlocConstAccessor;

    typedef type::Mat<MblocRsize, MblocCsize, MelementType> MBloc;
    typedef linearalgebra::CompressedRowSparseMatrix<MBloc>                       MMatrix;
    typedef typename MMatrix::ColBlockConstIterator                MBColConstIterator;
    typedef typename MMatrix::BlockConstAccessor                   MBlocConstAccessor;

    typedef type::Mat<MblocRsize, JblocCsize, MelementType> OutBloc;
    typedef linearalgebra::CompressedRowSparseMatrix<OutBloc>                     OutMatrix;

    JMatrix* Jmatrix = dynamic_cast<JMatrix*>(J);
    MMatrix* Mmatrix = dynamic_cast<MMatrix*>(stiffMatrix2);
    OutMatrix* Outmatrix = dynamic_cast<OutMatrix*>(out);

    if (!Jmatrix || !Mmatrix) return false;

    msg_info_when(_debug, "CRSMultiMatrixAccessor") << "    **opAddMulMJ_TBloc: === "
        << "K1[" << out->rowSize() << "x" << out->colSize() << "]:_"
        << out->bRowSize() << "x" << out->bColSize() << "_ of blocs _["
        << out->getBlockRows() << "x" << out->getBlockCols() << "]_"
        << "    +=    K2[" << stiffMatrix2->rowSize() << "x" << stiffMatrix2->colSize() << "]:_"
        << stiffMatrix2->bRowSize() << "x" << stiffMatrix2->bColSize() << "_ of blocs _["
        << stiffMatrix2->getBlockRows() << "x" << stiffMatrix2->getBlockCols() << "]_"
        << " * J[" << J->rowSize() << "x" << J->colSize() << "]:_"
        << J->bRowSize() << "x" << J->bColSize() << "_ of blocs _["
        << J->getBlockRows() << "x" << J->getBlockCols() << "]_"
        << "   ==============================";

    // We can compute stiffMatrix 1 += Jt M
    if (!Outmatrix || (offsetRow % MblocRsize != 0) || (offsetCol % JblocCsize != 0))
    {
        // optimized multiplication, but generic output
        JBloc Jblocbuffer;
        MBloc Mblocbuffer;

        for (int MBRowIndex = 0; MBRowIndex < Mmatrix->nBlockRow; MBRowIndex++)
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
                                msg_info_when(_debug, "CRSMultiMatrixAccessor") << " (i." << offsetRow + MblocRsize * MBRowIndex + i << ","
                                    << "j." << offsetCol + JblocCsize * JBColIndex + j << ")"
                                    << "    MBRowIndex:" << MBRowIndex << " JBColIndex:" << JBColIndex
                                    << "      i:" << i << " j:" << j << "  k:" << k;
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

        for (int MBRowIndex = 0; MBRowIndex < Mmatrix->nBlockRow; MBRowIndex++)
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
                                msg_info_when(_debug, "CRSMultiMatrixAccessor") << " (bI." << offsetBRow + MBRowIndex << ","
                                    << "bJ." << offsetBCol + JBColIndex << ")"
                                    << "    MBRowIndex:" << MBRowIndex << " JBColIndex:" << JBColIndex
                                    << "      i:" << i << " j:" << j << "  k:" << k;

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
inline bool opAddMulMJ_T4(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* stiffMatrix2, linearalgebra::BaseMatrix* J,
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
inline bool opAddMulMJ_T3(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* stiffMatrix2, linearalgebra::BaseMatrix* J,
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
inline bool opAddMulMJ_T2(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* stiffMatrix2, linearalgebra::BaseMatrix* J,
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
inline bool opAddMulMJ_T1(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* stiffMatrix2, linearalgebra::BaseMatrix* J,
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
inline bool opAddMulMJ(linearalgebra::BaseMatrix* out, linearalgebra::BaseMatrix* stiffMatrix2, linearalgebra::BaseMatrix* J,
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


} // namespace sofa::core::behavior
