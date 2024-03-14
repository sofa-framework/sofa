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
#include <sofa/core/behavior/CRSMultiMatrixAccessor.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/BaseMapping.h>

namespace sofa::core::behavior
{

using sofa::core::behavior::BaseMechanicalState;

void CRSMultiMatrixAccessor::addMechanicalMapping(sofa::core::BaseMapping* mapping)
{
    const sofa::linearalgebra::BaseMatrix* jmatrix = mapping->getJ();

    if ((jmatrix != nullptr) && (mapping->isMechanical()) && (mapping->areMatricesMapped()))
    {
        const BaseMechanicalState* mappedState  = const_cast<const BaseMechanicalState*>(mapping->getMechTo()[0]);
        linearalgebra::BaseMatrix* mappedstiffness;
        mappedstiffness = mapping->createMappedMatrix(mappedState,mappedState,&CRSMultiMatrixAccessor::createMatrix);
        mappedMatrices[mappedState]=mappedstiffness;

        mappingList.push_back(mapping);
        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "Mapping Visitor : adding validated MechanicalMapping " << mapping->getName()
            << " with J[" << jmatrix->rowSize() << "." << jmatrix->colSize() << "]";
    }
    else
    {
        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	-- Warning DefaultMultiMatrixAccessor : mapping " << mapping->getName() << " do not build matrices ";
    }
}

linearalgebra::BaseMatrix* CRSMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1,
                                                              const sofa::core::behavior::BaseMechanicalState* mstate2)
{
    return createMatrix(mstate1, mstate2, false);
}

linearalgebra::BaseMatrix* CRSMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1,
                                                              const sofa::core::behavior::BaseMechanicalState* mstate2,
                                                              bool doPrintInfo)
{
    // The auxiliar interaction matrix is added if and only if at least one of two state is not real state
    //assert(! (realStateOffsets.find(mstate1) != realStateOffsets.end() && realStateOffsets.find(mstate2) != realStateOffsets.end()) );

    int nbDOFs1  = mstate1->getSize();
    int dofSize1 = mstate1->getDerivDimension();//getMatrixBlockSize();
    int nbDOFs2  = mstate2->getSize();
    int dofSize2 = mstate2->getDerivDimension();//getMatrixBlockSize();
    //int elementsize = globalMatrix->getElementSize();

    if (mstate1 == mstate2)
    {
        msg_info_when(doPrintInfo, "CRSMultiMatrixAccessor") << "			++ Creating matrix Mapped Mechanical State  : " << mstate1->getName()
            << " associated to K[" << mstate1->getMatrixSize() << "x" << mstate1->getMatrixSize() << "] in the format _"
            << nbDOFs1 << "x" << nbDOFs1 << "_ of blocs _["
            << dofSize1 << "x" << dofSize1 << "]_";
        return createBlocSparseMatrix(dofSize1,dofSize1,sizeof(SReal) /*elementsize*/,nbDOFs1,nbDOFs1,doPrintInfo);

    }
    else
    {
        msg_info_when(doPrintInfo, "CRSMultiMatrixAccessor") << "			++ Creating matrix Interaction: "
            << mstate1->getName() << " -- " << mstate2->getName()
            << " associated to K[" << mstate1->getMatrixSize() << "x" << mstate2->getMatrixSize() << "] in the format _"
            << nbDOFs1 << "x" << nbDOFs2 << "_ of blocs _["
            << dofSize1 << "x" << dofSize2 << "]_";
        return createBlocSparseMatrix(dofSize1,dofSize2,sizeof(SReal) /*elementsize*/,nbDOFs1,nbDOFs2,doPrintInfo);
    }
}

void CRSMultiMatrixAccessor::computeGlobalMatrix()
{
    msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "==========================     VERIFICATION BLOC MATRIX FORMATS    ========================";

    for (std::map< const BaseMechanicalState*, MatrixRef >::iterator it = diagonalStiffnessBloc.begin(), itend = diagonalStiffnessBloc.end(); it != itend; ++it)
    {
        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << " Mechanical State  : " << it->first->getName()
            << " associated to K[" << it->second.matrix->rowSize() << "x" << it->second.matrix->colSize() << "] in the format _"
            << it->second.matrix->bRowSize() << "x" << it->second.matrix->bColSize() << "_ of blocs _["
            << it->second.matrix->getBlockRows() << "x" << it->second.matrix->getBlockCols() << "]_";
    }

    std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator itBegin = interactionStiffnessBloc.begin();
    std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator itEnd = interactionStiffnessBloc.end();

    while(itBegin != itEnd)
    {
        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << " Interaction: "
            << itBegin->first.first->getName() << " -- " << itBegin->first.second->getName()
            << " associated to K[" << itBegin->second.matrix->rowSize() << "x" << itBegin->second.matrix->colSize() << "] in the format _"
            << itBegin->second.matrix->bRowSize() << "x" << itBegin->second.matrix->bColSize() << "_ of blocs _["
            << itBegin->second.matrix->getBlockRows() << "x" << itBegin->second.matrix->getBlockCols() << "]_";

        ++itBegin;
    }

    const int lastMappingId = mappingList.size() - 1;
    for(int id=lastMappingId; id>=0; --id)
    {
        sofa::core::BaseMapping* m_mapping = mappingList[id];
        const BaseMechanicalState* instate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechFrom()[0]);
        const BaseMechanicalState* outstate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechTo()[0]);
        const linearalgebra::BaseMatrix* matrixJ = m_mapping->getJ();

        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "  " << id << "-th Mapping : " << m_mapping->getName() << " associated to matrix J["
            << matrixJ->rowSize() << "x" << matrixJ->colSize() << "] in the format _"
            << matrixJ->bRowSize() << "x" << matrixJ->bColSize() << "_ of blocs _["
            << matrixJ->getBlockRows() << "x" << matrixJ->getBlockCols() << "]_";

        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "			inState  : " << instate->getName()
            << " associated to K11[" << instate->getMatrixSize() << "x" << instate->getMatrixSize() << "] in the format _"
            << instate->getSize() << "x" << instate->getSize() << "_ of blocs _["
            << instate->getDerivDimension() << "x" << instate->getDerivDimension() << "]_";

        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "			outState  : "<< outstate->getName()
                <<" associated to K11["<< outstate->getMatrixSize() <<"x"<< outstate->getMatrixSize() << "] in the format _"
                << outstate->getSize() << "x"<< outstate->getSize() <<"_ of blocs _["
                << outstate->getDerivDimension() << "x"<< outstate->getDerivDimension() <<"]_";
    }
    msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "=======================     CONTRIBUTION CONTRIBUTION CONTRIBUTION     ======================";
    
    for(int id=lastMappingId; id>=0; --id)
    {
        sofa::core::BaseMapping* m_mapping = mappingList[id];
        const BaseMechanicalState* instate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechFrom()[0]);
        const BaseMechanicalState* outstate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechTo()[0]);

        const linearalgebra::BaseMatrix* matrixJ = m_mapping->getJ();
        const unsigned int nbR_J = matrixJ->rowSize();
        const unsigned int nbC_J = matrixJ->colSize();

        msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "[CONTRIBUTION] of " << id << "-th mapping named : " << m_mapping->getName()
            << " with fromModel : " << instate->getName()
            << " and toModel : " << outstate->getName() << " -- with matrix J[" << nbR_J << "." << nbC_J << "]";

        //for toModel -----------------------------------------------------------
        if(mappedMatrices.find(outstate) == mappedMatrices.end())
        {
            msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	[Propa.Stiff] WARNING toModel : " << outstate->getName() << " dont have stiffness matrix";
        }



        if(diagonalStiffnessBloc.find(outstate) != diagonalStiffnessBloc.end())
        {
            //=================================
            //           K11 += Jt * K22 * J
            //=================================
            MatrixRef K1 = this->getMatrix(instate);
            MatrixRef K2 = this->getMatrix(outstate);

            const unsigned int offset1  = K1.offset;
            const unsigned int offset2  = K2.offset;
            const unsigned int sizeK1 = K1.matrix->rowSize() - offset1;
            const unsigned int sizeK2 = K2.matrix->rowSize() - offset2;

            linearalgebra::BaseMatrix* matrixJJ =const_cast<linearalgebra::BaseMatrix*>(matrixJ);

            int JblocRsize     = matrixJJ->getBlockRows();
            int JblocCsize     = matrixJJ->getBlockCols();
            int JnbBlocCol     = matrixJJ->bColSize();

            int K2blocCsize    = K2.matrix->getBlockCols();
            int K2nbBlocCol   = K2.matrix->bColSize();

            int JelementSize   = matrixJJ->getElementSize();
            int MelementSize   = K2.matrix->getElementSize();

            // creating a tempo matrix  tempoMatrix
            linearalgebra::BaseMatrix* tempoMatrix = createBlocSparseMatrix(JblocCsize, K2blocCsize, MelementSize, JnbBlocCol, K2nbBlocCol,m_doPrintInfo);
            // Matrix multiplication  tempoMatrix += Jt * K22
            opAddMulJTM(tempoMatrix,   matrixJJ, K2.matrix,       0,0      , JblocRsize, JblocCsize , K2blocCsize, JelementSize,MelementSize,m_doPrintInfo);
            // Matrix multiplication  K11         += tempoMatrix * J
            opAddMulMJ( K1.matrix  ,tempoMatrix,  matrixJJ, offset1,offset1, JblocCsize, K2blocCsize, JblocCsize , JelementSize,MelementSize,m_doPrintInfo);

            delete tempoMatrix;

            msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	[Propa.Stiff] propagating stiffness of : " << outstate->getName() << " to stifness " << instate->getName();
            msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	                    **multiplication of "
                << " K1[" << sizeK1 << "." << sizeK1 << "](" << offset1 << "," << offset1 << ")   =  "
                << " Jt[" << nbC_J << "." << nbR_J << "] * "
                << " K2[" << sizeK2 << "." << sizeK2 << "](" << offset2 << "," << offset2 << ") * "
                << " J[" << nbR_J << "." << nbC_J << "]";
        }

        std::vector<std::pair<const BaseMechanicalState*, const BaseMechanicalState*> > interactionList;
        for (auto it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        {
            if(it->first.first == outstate || it->first.second == outstate )
            {
                interactionList.push_back(it->first);
            }
        }


        const unsigned nbInteraction = interactionList.size();
        for(unsigned i=0; i< nbInteraction; i++)
        {
            msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	[Propa.Interac.Stiff] detected interaction between toModel : " << interactionList[i].first->getName()
                << " and : " << interactionList[i].second->getName();
            
            //                   |       |
            //                 MS1     MS2
            //                  /      /
            //                map    inter
            //                   \   /
            //                   MS3/
            //
            //           K_11 += Jt * K_33 * J
            //           I_12 += Jt * I_32
            //           I_21 +=      I_23 * J
            //

            if(interactionList[i].first == outstate)
            {
                InteractionMatrixRef I_32 = this->getMatrix( outstate , interactionList[i].second);
                InteractionMatrixRef I_12 = this->getMatrix( instate  , interactionList[i].second);
                //===========================
                //          I_12 += Jt * I_32
                //===========================
                const unsigned int offR_I_12  = I_12.offRow;                       //      row offset of I12 matrix
                const unsigned int offC_I_12  = I_12.offCol;                       //    colum offset of I12 matrix
                const unsigned int nbR_I_12   = I_12.matrix->rowSize() - offR_I_12;//number of rows   of I12 matrix
                const unsigned int nbC_I_12   = I_12.matrix->colSize() - offC_I_12;//number of colums of I12 matrix

                const unsigned int offR_I_32  = I_32.offRow;                     //      row offset of I32 matrix
                const unsigned int offC_I_32  = I_32.offCol;                     //    colum offset of I32 matrix
                const unsigned int nbR_I_32 = I_32.matrix->rowSize() - offR_I_32;//number of rows   of I32 matrix
                const unsigned int nbC_I_32 = I_32.matrix->colSize() - offC_I_32;//number of colums of I32 matrix

                msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	[Propa.Interac.Stiff] propagating interaction "
                    << outstate->getName() << "--" << interactionList[i].second->getName()
                    << "  to : "
                    << instate->getName() << "--" << interactionList[i].second->getName();

                msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	                    **multiplication of "
                    << " I12[" << nbR_I_12 << "." << nbC_I_12 << "](" << offR_I_12 << "," << offC_I_12 << ")  =  "
                    << " Jt[" << nbC_J << "." << nbR_J << "]  *  "
                    << " I32[" << nbR_I_32 << "." << nbC_I_32 << "](" << offR_I_32 << "," << offC_I_32 << ")";

                // Matrix multiplication   I_12 += Jt * I_32

                linearalgebra::BaseMatrix* matrixJJ =const_cast<linearalgebra::BaseMatrix*>(matrixJ);
                int JblocRsize     = matrixJJ->getBlockRows();
                int JblocCsize     = matrixJJ->getBlockCols();
                int I_32_blocCsize = I_32.matrix->getBlockCols();
                int JelementSize   = matrixJJ->getElementSize();
                int MelementSize   = I_32.matrix->getElementSize();

                opAddMulJTM(I_12.matrix,matrixJJ,I_32.matrix,offR_I_12,offC_I_12, JblocRsize, JblocCsize,I_32_blocCsize,JelementSize,MelementSize,m_doPrintInfo);
            }

            if(interactionList[i].second == outstate)
            {
                InteractionMatrixRef I_21 = this->getMatrix(interactionList[i].first,instate);
                InteractionMatrixRef I_23 = this->getMatrix(interactionList[i].first,outstate);
                //=========================================
                //          I_21 +=      I_23 * J
                //=========================================
                const unsigned int offR_I_21  = I_21.offRow;
                const unsigned int offC_I_21  = I_21.offCol;
                const unsigned int nbR_I_21 = I_21.matrix->rowSize() - offR_I_21;
                const unsigned int nbC_I_21 = I_21.matrix->colSize() - offC_I_21;

                const unsigned int offR_I_23  = I_23.offRow;
                const unsigned int offC_I_23  = I_23.offCol;
                const unsigned int nbR_I_23   = I_23.matrix->rowSize() - offR_I_23;
                const unsigned int nbC_I_23   = I_23.matrix->colSize() - offC_I_23;

                msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	[Propa.Interac.Stiff] propagating interaction "
                    << interactionList[i].first->getName() << "--" << outstate->getName()
                    << " to : " << interactionList[i].first->getName() << "--" << instate->getName();

                msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	                    **multiplication of "
                    << " I_21[" << nbR_I_21 << "." << nbC_I_21 << "](" << offR_I_21 << "," << offC_I_21 << ")  =  "
                    << " I23[" << nbR_I_23 << "." << nbC_I_23 << "](" << offR_I_23 << "," << offC_I_23 << ")  *  "
                    << " J[" << nbR_J << "." << nbC_J << "]";

                // Matrix multiplication  I_21 +=  I_23 * J
                linearalgebra::BaseMatrix* matrixJJ =const_cast<linearalgebra::BaseMatrix*>(matrixJ);
                int I_23_blocRsize = I_23.matrix->getBlockRows();
                int I_23_blocCsize = I_23.matrix->getBlockCols();
                int JblocCsize     = matrixJJ->getBlockCols();
                int JelementSize   = matrixJJ->getElementSize();
                int MelementSize   = I_23.matrix->getElementSize();

                opAddMulMJ(I_21.matrix,I_23.matrix,matrixJJ,offR_I_21,offC_I_21, I_23_blocRsize,I_23_blocCsize,JblocCsize,JelementSize,MelementSize,m_doPrintInfo);

            }

            //after propagating the interaction, we remove the older interaction
            interactionStiffnessBloc.erase( interactionStiffnessBloc.find(interactionList[i]) );
            msg_info_when(m_doPrintInfo, "CRSMultiMatrixAccessor") << "	--[Propa.Interac.Stiff] remove interaction of : " << interactionList[i].first->getName()
                << " and : " << interactionList[i].second->getName();

        }//end of interaction loop

    }//end of mapping loop
}

} //namespace sofa::core::behavior
