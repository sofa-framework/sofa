/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <SofaBaseLinearSolver/DefaultMultiMatrixAccessor.h>
#include <sofa/core/visual/VisualParams.h>

using sofa::core::behavior::BaseMechanicalState;

namespace sofa
{

namespace component
{

namespace linearsolver
{

#define MULTIMATRIX_VERBOSE 0

DefaultMultiMatrixAccessor::DefaultMultiMatrixAccessor()
    : globalMatrix(NULL)
    , globalDim(0)
{
}


DefaultMultiMatrixAccessor::~DefaultMultiMatrixAccessor()
{
    this->clear();
}

void DefaultMultiMatrixAccessor::clear()
{
    globalDim = 0;
    for (std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it = realStateOffsets.begin(), itend = realStateOffsets.end(); it != itend; ++it)
        it->second = -1;

    for (std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::iterator it = mappedMatrices.begin(), itend = mappedMatrices.end(); it != itend; ++it)
        if (it->second != NULL) delete it->second;
    mappedMatrices.clear();
    diagonalStiffnessBloc.clear();

    for (std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        if (it->second.matrix != NULL && it->second.matrix != globalMatrix) delete it->second.matrix;

    interactionStiffnessBloc.clear();
    mappingList.clear();

}


void DefaultMultiMatrixAccessor::setGlobalMatrix(defaulttype::BaseMatrix* matrix)
{
    this->globalMatrix = matrix;
}

void DefaultMultiMatrixAccessor::addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
    unsigned int dim = mstate->getMatrixSize();
    realStateOffsets[mstate] = globalDim;
    globalDim += dim;

    if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
    {
        std::cout << "Mechanical Visitor : adding Real MechanicalState " << mstate->getName()
                << " in global matrix ["<<dim<<"."<<dim
                <<"] at offset (" << realStateOffsets[mstate] <<","<< realStateOffsets[mstate]<<")"<< std::endl;
    }
}


void DefaultMultiMatrixAccessor::addMechanicalMapping(sofa::core::BaseMapping* mapping)
{
    const sofa::defaulttype::BaseMatrix* jmatrix = NULL;
    if (mapping->isMechanical() && mapping->areMatricesMapped())
        jmatrix = mapping->getJ();

    if (jmatrix)
    {

        const BaseMechanicalState* mappedState  = const_cast<const BaseMechanicalState*>(mapping->getMechTo()[0]);
        defaulttype::BaseMatrix* mappedstiffness;
        mappedstiffness = mapping->createMappedMatrix(mappedState,mappedState,&DefaultMultiMatrixAccessor::createMatrix);
        mappedMatrices[mappedState]=mappedstiffness;

        mappingList.push_back(mapping);

        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "Mapping Visitor : adding validated MechanicalMapping " << mapping->getName()
                    << " with J["<< jmatrix->rowSize()<<"."<<jmatrix->colSize()<<"]" <<std::endl;
        }
    }
    else
    {
        //std::cout << "	-- Warning DefaultMultiMatrixAccessor : mapping " << mapping->getName()<<" do not build matrices " << std::endl;
    }
}


void DefaultMultiMatrixAccessor::addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* /*mstate*/)
{
    // do not add the mapped mechanical state here because
    // a mapped mechanical state is added if and only if it has its own stiffness matrix and its mapping must satistify several conditions
    // so if and only if there are a forcefield or other component call getMatrix(mappedstate)
}

void DefaultMultiMatrixAccessor::setupMatrices()
{
    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it = realStateOffsets.begin(), itend = realStateOffsets.end();
    while (it != itend)
    {
        if (it->second < 0)
        {
            std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it2 = it;
            ++it;
            realStateOffsets.erase(it2);
        }
        else
        {
            if (globalMatrix)
            {
                MatrixRef& r = diagonalStiffnessBloc[it->first];
                r.matrix = globalMatrix;
                r.offset = it->second;
            }
            ++it;
        }
    }

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "Setup Global Matrix [" << globalDim << "." << globalDim << "] for " << realStateOffsets.size() << " real mechanical state" << std::endl;
    }


#if 0 // the following code was used to debug resize issues in CompressedRowSparseMatrix
    if (globalMatrix)
    {
        globalMatrix->resize(globalDim, globalDim);
        std::cout << "DefaultMultiMatrixAccessor: resize -> " << globalMatrix->rowSize() << "x" << globalMatrix->colSize() << " : " << globalMatrix->bRowSize() << "x" << globalMatrix->bColSize() << " blocks of size " << globalMatrix->getBlockRows() << "x"  << globalMatrix->getBlockCols() << "." << std::endl;
        globalMatrix->resize(globalDim, globalDim);
        std::cout << "DefaultMultiMatrixAccessor: resize -> " << globalMatrix->rowSize() << "x" << globalMatrix->colSize() << " : " << globalMatrix->bRowSize() << "x" << globalMatrix->bColSize() << " blocks of size " << globalMatrix->getBlockRows() << "x"  << globalMatrix->getBlockCols() << "." << std::endl;
    }
#endif
}

int DefaultMultiMatrixAccessor::getGlobalDimension() const
{
    return globalDim;
}

int DefaultMultiMatrixAccessor::getGlobalOffset(const sofa::core::behavior::BaseMechanicalState* mstate) const
{
    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it = realStateOffsets.find(mstate);
    if (it != realStateOffsets.end())
        return it->second;
    return -1;
}

DefaultMultiMatrixAccessor::MatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const
{
    MatrixRef r;
    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator itRealState = realStateOffsets.find(mstate);

    if (itRealState != realStateOffsets.end()) //case where mechanical state is a non mapped state
    {
        if (globalMatrix)
        {
            r.matrix = globalMatrix;
            r.offset = itRealState->second;
        }
    }
    else //case where mechanical state is a mapped state
    {
        std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix*>::iterator itmapped = mappedMatrices.find(mstate);
        if (itmapped != mappedMatrices.end()) // this mapped state and its matrix has been already added and created
        {
            r.matrix = itmapped->second;
            r.offset = 0;
        }
        else // this mapped state and its matrix hasnt been created we creat it and its matrix by "createMatrix"
        {

            defaulttype::BaseMatrix* m = createMatrix(mstate,mstate);
            r.matrix = m;
            r.offset = 0;
            //when creating an matrix, it dont have to be added before
            assert(diagonalStiffnessBloc.find(mstate) == diagonalStiffnessBloc.end());
            mappedMatrices[mstate]=r.matrix;
        }
    }

    diagonalStiffnessBloc[mstate] = r;

    if( MULTIMATRIX_VERBOSE)
    {
        if (r.matrix != NULL)
        {
            std::cout << "		Giving Stiffness Matrix [" << r.matrix->rowSize() << "." << r.matrix->colSize() << "] for state " << mstate->getName()
                    << " at offset (" << r.offset  <<","<< r.offset <<")"<< std::endl;
        }
        else
            std::cout << "		WARNING: NULL matrix found for state " << mstate->getName() << std::endl;
    }
    return r;
}

DefaultMultiMatrixAccessor::InteractionMatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{
    InteractionMatrixRef r2;
    if (mstate1 == mstate2)// case where state1 == state2, interaction matrix is on the diagonal stiffness bloc
    {
        MatrixRef r = diagonalStiffnessBloc.find(mstate1)->second;
        r2.matrix = r.matrix;
        r2.offRow = r.offset;
        r2.offCol = r.offset;

        if( MULTIMATRIX_VERBOSE)///////////////////////////////////////////
        {
            if (r2.matrix != NULL)
                std::cout << "		Giving Interaction Stiffness Matrix ["
                        << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                        <<"] at offset ("<<r2.offRow <<","<< r2.offCol<<") for self-interaction : "
                        <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"<< std::endl;
            else
                std::cout << "		WARNING : giving NULL matrix for self-interaction "<<
                        mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" << std::endl;
        }
    }
    else// case where state1 # state2
    {
        std::pair<const BaseMechanicalState*,const BaseMechanicalState*> pairMS = std::make_pair(mstate1,mstate2);

        std::map< std::pair<const BaseMechanicalState*,const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.find(pairMS);
        if (it != interactionStiffnessBloc.end())// the interaction is already added
        {
            if(it->second.matrix != NULL)
            {
                r2 = it->second;
            }

            if( MULTIMATRIX_VERBOSE)///////////////////////////////////////////
            {
                if(r2.matrix != NULL)
                {
                    std::cout << "		Giving Interaction Stiffness Matrix ["
                            << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                            <<"] at offset ("<<r2.offRow <<","<< r2.offCol<<")  for interaction : "
                            <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"<< std::endl;
                }
                else
                {
                    std::cout << "		WARNING : giving NULL matrix  for interaction "
                            <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"<< std::endl;
                }
            }
        }
        else// the interaction is not added, we need to creat it and its matrix
        {
            std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it1 = realStateOffsets.find(mstate1);
            std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it2 = realStateOffsets.find(mstate2);

            if(it1 != realStateOffsets.end() && it2 != realStateOffsets.end())// case where all of two ms are real DOF (non-mapped)
            {
                if (globalMatrix)
                {
                    r2.matrix = globalMatrix;
                    r2.offRow = it1->second;
                    r2.offCol = it2->second;

                    if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
                    {
                        if (r2.matrix != NULL)
                            std::cout <<  "		Giving Interaction Stiffness Matrix ["
                                    << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                                    <<"] at offset ("<<r2.offRow <<","<< r2.offCol<<")  for interaction : "
                                    <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"<< std::endl;
                        else
                            std::cout << "		WARNING : giving NULL matrix  for interaction "
                                    << " for interaction : "
                                    << mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                                    << std::endl;
                    }
                }
            }
            else //case where at least one ms is a mapped
            {
                defaulttype::BaseMatrix* m = createMatrix(mstate1,mstate2);
                r2.matrix = m;
                r2.offRow = 0;
                r2.offCol = 0;
                //when creating an matrix, it dont have to be added before
                assert(interactionStiffnessBloc.find(pairMS) == interactionStiffnessBloc.end());

                if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
                {
                    if (r2.matrix != NULL)
                        std::cout <<   "		Giving Interaction Stiffness Matrix ["
                                << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                                << "] at offset ("<<r2.offRow <<","<< r2.offCol<<")  for interaction : "
                                << mstate1->getName()<< "[" <<mstate1->getMatrixSize()<< "] --- " <<mstate2->getName()<<"[" <<mstate2->getMatrixSize()<<"]"<< std::endl;
                    else
                        std::cout << "		WARNING : giving NULL matrix  for interaction "
                                << " for interaction : "
                                << mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                                << std::endl;
                }
            }

            interactionStiffnessBloc[pairMS]=r2;
        }// end of the interaction is not added, we need to creat it and its matrix

    }//end of case where state1 # state2


    if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
    {
        if(r2.matrix == NULL)
            std::cout << "		WARNING : NULL matrix found for interaction " << mstate1->getName()<<" --- "<<mstate2->getName() << std::endl;
    }

    return r2;
}

void DefaultMultiMatrixAccessor::computeGlobalMatrix()
{

    if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
    {
        //				std::cout << "==========================     VERIFICATION REGISTERED IN LOCAL DATA ========================" <<std::endl << std::endl;
        //
        //				for (std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it = realStateOffsets.begin(), itend = realStateOffsets.end(); it != itend; ++it)
        //				{
        //					std::cout << "                Mechanical State (Real) : "<< it->first->getName() <<" registered in list" <<std::endl;
        //				}
        //
        //				for (std::map< const sofa::core::behavior::BaseMechanicalState*,defaulttype::BaseMatrix* >::iterator it = mappedMatrices.begin(), itend = mappedMatrices.end(); it != itend; ++it)
        //				{
        //					std::cout << "                Mechanical State (mapped) : "<< it->first->getName() <<" registered in list" <<std::endl;
        //				}
        //
        //				for (std::map< const BaseMechanicalState*, MatrixRef >::iterator it = diagonalStiffnessBloc.begin(), itend = diagonalStiffnessBloc.end(); it != itend; ++it)
        //				{
        //					std::cout << "                Mechanical State ( all ) : "<< it->first->getName() <<" registered in list" <<std::endl;
        //				}
        //
        //				std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator itBegin = interactionStiffnessBloc.begin();
        //				std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator itEnd = interactionStiffnessBloc.end();
        //
        //				while(itBegin != itEnd)
        //				{
        //					std::cout << "                 Interaction: "
        //		                      << itBegin->first.first->getName() <<" -- "<< itBegin->first.second->getName()<<std::endl;
        //
        //					++itBegin;
        //				}
        //
        //				const int lastMappingId = mappingList.size() - 1;
        //				for(int id=lastMappingId;id>=0;--id)
        //				{
        //					std::cout << "                mapping "<<id<< "-th  :"<<mappingList[id]->getName() <<" registered in list" <<std::endl;
        //				}
        std::cout <<std::endl<< "=======================     CONTRIBUTION CONTRIBUTION CONTRIBUTION     ======================" <<std::endl << std::endl;
    }


    const int lastMappingId = (int)mappingList.size() - 1;
    for(int id=lastMappingId; id>=0; --id)
    {
        sofa::core::BaseMapping* m_mapping = mappingList[id];
        const BaseMechanicalState* instate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechFrom()[0]);
        const BaseMechanicalState* outstate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechTo()[0]);

        const defaulttype::BaseMatrix* matrixJ = m_mapping->getJ();
        const unsigned int nbR_J = matrixJ->rowSize();
        const unsigned int nbC_J = matrixJ->colSize();

        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "[CONTRIBUTION] of " << id << "-th mapping named : " << m_mapping->getName()
                    << " with fromModel : "<< instate->getName()
                    << " and toModel : "   << outstate->getName() <<" -- with matrix J["<<nbR_J <<"."<<nbC_J <<"]" <<std::endl;
        }


        //for toModel -----------------------------------------------------------
        if(mappedMatrices.find(outstate) == mappedMatrices.end())
        {
            if( MULTIMATRIX_VERBOSE)
                std::cout << "	[Propa.Stiff] WARNING toModel : "<< outstate->getName()<< " dont have stiffness matrix"<<std::endl;
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

            if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
            {
                std::cout << "	[Propa.Stiff] propagating stiffness of : "<< outstate->getName()<< " to stifness "<<instate->getName()<<std::endl;
                std::cout <<"	                    **multiplication of "
                        <<" K1["<<sizeK1<<"."<<sizeK1<< "]("<< offset1<<","<<offset1 <<  ")   =  "
                        <<" Jt["<<nbC_J<<"."<<nbR_J<< "] * "
                        <<" K2["<<sizeK2<<"."<<sizeK2<<"]("<< offset2<<","<<offset2 <<  ") * "
                        <<" J["<<nbR_J<<"."<<nbC_J<< "]"
                        <<std::endl;
            }

            // Matrix multiplication  K11 += Jt * K22 * J
            for(unsigned int i1 =0; i1 < sizeK1 ; ++i1)
            {
                for(unsigned int j1 =0 ; j1 < sizeK1 ; ++j1)
                {
                    double Jt_K2_J_i1j1 = 0;

                    for(unsigned int i2 =0 ; i2 < sizeK2 ; ++i2)
                    {
                        for(unsigned int j2 =0 ; j2 < sizeK2 ; ++j2)
                        {
                            const double K2_i2j2 = (double) K2.matrix->element(offset2 + i2, offset2 + j2);
                            for(unsigned int k2=0 ; k2 < sizeK2 ; ++k2)
                            {
                                const double Jt_i1k2 = (double) matrixJ->element( i1 , k2 ) ;
                                const double  J_k2j1 = (double) matrixJ->element( k2 , j1 ) ;

                                Jt_K2_J_i1j1 += Jt_i1k2 * K2_i2j2  * J_k2j1;
                                /*
                                if( MULTIMATRIX_VERBOSE)  // index debug
                                {
                                std::cout<<"K1("<<offset1 + i1<<","<<offset1 + j1<<")  +="
                                <<" Jt("<<i1<<","<<k2<<")  * "
                                <<"K2("<<offset2 + i2<<","<<offset2 + j2<<") * "
                                <<" J("<<k2<<","<<j1<<")"<<std::endl;
                                }
                                */
                            }
                        }
                    }
                    K1.matrix->add(offset1 + i1 , offset1 + j1 , Jt_K2_J_i1j1);
                }
            }
            // Matrix multiplication  K11 += Jt * K22 * J

        }

        std::vector<std::pair<const BaseMechanicalState*, const BaseMechanicalState*> > interactionList;
        for (std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        {
            if(it->first.first == outstate || it->first.second == outstate )
            {
                interactionList.push_back(it->first);
            }
            ++it;
        }


        const size_t nbInteraction = interactionList.size();
        for(size_t i=0; i< nbInteraction; i++)
        {

            if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
            {
                std::cout << "	[Propa.Interac.Stiff] detected interaction between toModel : "<<interactionList[i].first->getName()
                        << " and : " <<interactionList[i].second->getName()<<std::endl;
            }
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


                if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
                {
                    std::cout << "	[Propa.Interac.Stiff] propagating interaction "
                            <<outstate->getName() << "--" <<interactionList[i].second->getName()
                            <<"  to : "
                            <<instate->getName() << "--" <<interactionList[i].second->getName()<<std::endl;

                    std::cout <<"	                    **multiplication of "
                            <<" I12["<<nbR_I_12<<"."<<nbC_I_12<< "]("<< offR_I_12<<","<<offC_I_12 <<  ")  =  "
                            <<" Jt["<<nbC_J<<"."<<nbR_J<< "]  *  "
                            <<" I32["<<nbR_I_32<<"."<<nbC_I_32<<"]("<< offR_I_32<<","<<offC_I_32 <<  ")" <<std::endl;
                }

                // Matrix multiplication   I_12 += Jt * I_32
                for(unsigned int _i = 0; _i < nbR_I_12 ; _i++)
                {
                    for(unsigned int _j = 0; _j < nbC_I_12 ; _j++)
                    {
                        double Jt_I32_ij = 0;
                        for(unsigned int _k = 0; _k < nbR_I_32 ; _k++)
                        {
                            const double Jt_ik    = (double) matrixJ->element( _k, _i ) ;
                            const double  I_32_kj = (double) I_32.matrix->element( offR_I_32 + _k, offC_I_32+_j) ;

                            Jt_I32_ij += Jt_ik  *  I_32_kj;
                            /*
                            if( MULTIMATRIX_VERBOSE)  // index debug
                            {
                            std::cout<<"I12("<<offR_I_12 + _i<<","<<offC_I_12 +  _j<<")  +="
                            <<" Jt("<<_k<<","<<_i<<")  * "
                            <<"I32("<<offR_I_32 + _k<<","<< offC_I_32+_j<<")"<<std::endl;
                            }
                            */
                        }
                        I_12.matrix->add(offR_I_12 + _i , offC_I_12 +  _j , Jt_I32_ij);
                    }
                }// Matrix multiplication   I_12 += Jt * I_32
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

                if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
                {
                    std::cout << "	[Propa.Interac.Stiff] propagating interaction "
                            <<interactionList[i].first->getName()<< "--" <<outstate->getName()
                            <<" to : "<<interactionList[i].first->getName()<< "--" <<instate->getName()<<std::endl;

                    std::cout <<"	                    **multiplication of "
                            <<" I_21["<<nbR_I_21<<"."<<nbC_I_21<<"]("<< offR_I_21<<","<<offC_I_21 <<  ")  =  "
                            <<" I23["<<nbR_I_23<<"."<<nbC_I_23<< "]("<< offR_I_23<<","<<offC_I_23 <<  ")  *  "
                            <<" J["<<nbR_J<<"."<<nbC_J<<"]" <<std::endl;
                }


                // Matrix multiplication  I_21 +=  I_23 * J
                for(unsigned int _i = 0; _i < nbR_I_21 ; _i++)
                {
                    for(unsigned int _j = 0; _j < nbC_I_21 ; _j++)
                    {
                        double I23_J_ij = 0;
                        for(unsigned int _k = 0; _k < nbC_I_23 ; _k++)
                        {
                            const double I_23_ik = (double) I_23.matrix->element( offR_I_23 + _i, offC_I_23+_k) ;
                            const double J_kj    = (double) matrixJ->element( _k, _j ) ;

                            I23_J_ij += I_23_ik  * J_kj ;
                            /*
                            if( MULTIMATRIX_VERBOSE)  // index debug
                            {
                            std::cout<<"I21("<<offR_I_21 + _i<<","<<offC_I_21 + _j<<")  +="
                            <<"I23("<<offR_I_23 + _i<<","<< offC_I_23+_k<<") * "
                            <<" Jt("<<_k<<","<<_k<<")"<<std::endl;
                            }
                            */
                        }
                        I_21.matrix->add(offR_I_21 + _i , offC_I_21 + _j , I23_J_ij);
                    }
                }// Matrix multiplication  I_21 +=  I_23 * J

            }

            //after propagating the interaction, we remove the older interaction
            interactionStiffnessBloc.erase( interactionStiffnessBloc.find(interactionList[i]) );
            if( MULTIMATRIX_VERBOSE)
            {
                std::cout << "	--[Propa.Interac.Stiff] remove interaction of : "<<interactionList[i].first->getName()
                        << " and : " << interactionList[i].second->getName()<<std::endl;
            }

        }//end of interaction loop

    }//end of mapping loop
}

defaulttype::BaseMatrix* DefaultMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2)
{
    //   static std::map<const sofa::core::behavior::BaseMechanicalState*, component::linearsolver::FullMatrix<SReal>*> cochon;

    // The auxiliar interaction matrix is added if and only if at least one of two state is not real state
    //assert(! (realStateOffsets.find(mstate1) != realStateOffsets.end() && realStateOffsets.find(mstate2) != realStateOffsets.end()) );

    //component::linearsolver::FullMatrix<SReal>* m = new component::linearsolver::FullMatrix<SReal>;
    component::linearsolver::CompressedRowSparseMatrix<SReal>* m = new component::linearsolver::CompressedRowSparseMatrix<SReal>;
    if(mstate1 == mstate2)
    {
        m->resize( mstate1->getMatrixSize(),mstate1->getMatrixSize());

        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "			++ Creating matrix["<< m->rowSize() <<"."<< m->colSize() <<"]   for mapped state " << mstate1->getName() << "[" << mstate1->getMatrixSize()<<"]"<< std::endl;
        }
    }
    else
    {
        m->resize( mstate1->getMatrixSize(),mstate2->getMatrixSize() );

        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "			++ Creating interraction matrix["<< m->rowSize() <<"x"<< m->colSize()
                      << "] for interaction " << mstate1->getName() << "[" << mstate1->getMatrixSize()
                      << "] --- "             << mstate2->getName() << "[" << mstate2->getMatrixSize()<<"]" <<std::endl;
        }
    }
    //  }

    return m;
}




#ifdef SOFA_SUPPORT_CRS_MATRIX
//TODO separating in other file
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CRSMultiMatrixAccessor::addMechanicalMapping(sofa::core::BaseMapping* mapping)
{
    const sofa::defaulttype::BaseMatrix* jmatrix = mapping->getJ();

    if ((jmatrix != NULL) && (mapping->isMechanical()) && (mapping->areMatricesMapped()))
    {
        const BaseMechanicalState* mappedState  = const_cast<const BaseMechanicalState*>(mapping->getMechTo()[0]);
        defaulttype::BaseMatrix* mappedstiffness;
        mappedstiffness = mapping->createMappedMatrix(mappedState,mappedState,&CRSMultiMatrixAccessor::createMatrix);
        mappedMatrices[mappedState]=mappedstiffness;

        mappingList.push_back(mapping);

        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "Mapping Visitor : adding validated MechanicalMapping " << mapping->getName()
                    << " with J["<< jmatrix->rowSize()<<"."<<jmatrix->colSize()<<"]" <<std::endl;
        }
    }
    else
    {
        std::cout << "	-- Warning DefaultMultiMatrixAccessor : mapping " << mapping->getName()<<" do not build matrices " << std::endl;
    }
}

defaulttype::BaseMatrix* CRSMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2)
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
        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "			++ Creating matrix Mapped Mechanical State  : "<< mstate1->getName()
                    <<" associated to K["<< mstate1->getMatrixSize() <<"x"<< mstate1->getMatrixSize() << "] in the format _"
                    << nbDOFs1 << "x"<< nbDOFs1 <<"_ of blocs _["
                    << dofSize1 << "x"<< dofSize1 <<"]_"
                    <<std::endl;
        }
        return createBlocSparseMatrix(dofSize1,dofSize1,sizeof(SReal) /*elementsize*/,nbDOFs1,nbDOFs1,MULTIMATRIX_VERBOSE);

    }
    else
    {

        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "			++ Creating matrix Interaction: "
                    << mstate1->getName() <<" -- "<< mstate2->getName()
                    <<" associated to K["<< mstate1->getMatrixSize() <<"x"<< mstate2->getMatrixSize() << "] in the format _"
                    << nbDOFs1 << "x"<< nbDOFs2 <<"_ of blocs _["
                    << dofSize1 << "x"<< dofSize2 <<"]_"
                    <<std::endl;
        }

        return createBlocSparseMatrix(dofSize1,dofSize2,sizeof(SReal) /*elementsize*/,nbDOFs1,nbDOFs2,MULTIMATRIX_VERBOSE);
    }
}

void CRSMultiMatrixAccessor::computeGlobalMatrix()
{
    if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
    {
        std::cout << "==========================     VERIFICATION BLOC MATRIX FORMATS    ========================" <<std::endl << std::endl;

        for (std::map< const BaseMechanicalState*, MatrixRef >::iterator it = diagonalStiffnessBloc.begin(), itend = diagonalStiffnessBloc.end(); it != itend; ++it)
        {
            std::cout << " Mechanical State  : "<< it->first->getName()
                    <<" associated to K["<< it->second.matrix->rowSize() <<"x"<< it->second.matrix->colSize()<< "] in the format _"
                    << it->second.matrix->bRowSize() << "x"<< it->second.matrix->bColSize() <<"_ of blocs _["
                    << it->second.matrix->getBlockRows() << "x"<< it->second.matrix->getBlockCols() <<"]_"
                    <<std::endl;
        }

        std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator itBegin = interactionStiffnessBloc.begin();
        std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator itEnd = interactionStiffnessBloc.end();

        while(itBegin != itEnd)
        {
            std::cout << " Interaction: "
                    << itBegin->first.first->getName() <<" -- "<< itBegin->first.second->getName()
                    <<" associated to K["<< itBegin->second.matrix->rowSize() <<"x"<< itBegin->second.matrix->colSize()<< "] in the format _"
                    << itBegin->second.matrix->bRowSize() << "x"<< itBegin->second.matrix->bColSize() <<"_ of blocs _["
                    << itBegin->second.matrix->getBlockRows() << "x"<< itBegin->second.matrix->getBlockCols() <<"]_"
                    <<std::endl;

            ++itBegin;
        }

        const int lastMappingId = mappingList.size() - 1;
        for(int id=lastMappingId; id>=0; --id)
        {
            sofa::core::BaseMapping* m_mapping = mappingList[id];
            const BaseMechanicalState* instate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechFrom()[0]);
            const BaseMechanicalState* outstate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechTo()[0]);
            const defaulttype::BaseMatrix* matrixJ = m_mapping->getJ();

            std::cout << "  "<<id<< "-th Mapping : " <<m_mapping->getName()<< " associated to matrix J["
                    << matrixJ->rowSize() <<"x"<< matrixJ->colSize()<< "] in the format _"
                    << matrixJ->bRowSize() << "x"<< matrixJ->bColSize() <<"_ of blocs _["
                    << matrixJ->getBlockRows() << "x"<< matrixJ->getBlockCols() <<"]_"
                    <<std::endl;

            std::cout << "			inState  : "<< instate->getName()
                    <<" associated to K11["<< instate->getMatrixSize() <<"x"<< instate->getMatrixSize() << "] in the format _"
                    << instate->getSize() << "x"<< instate->getSize() <<"_ of blocs _["
                    << instate->getDerivDimension() << "x"<< instate->getDerivDimension() <<"]_"
                    <<std::endl;

            std::cout << "			outState  : "<< outstate->getName()
                    <<" associated to K11["<< outstate->getMatrixSize() <<"x"<< outstate->getMatrixSize() << "] in the format _"
                    << outstate->getSize() << "x"<< outstate->getSize() <<"_ of blocs _["
                    << outstate->getDerivDimension() << "x"<< outstate->getDerivDimension() <<"]_"
                    <<std::endl;
        }
        std::cout <<std::endl << "=======================     CONTRIBUTION CONTRIBUTION CONTRIBUTION     ======================" <<std::endl << std::endl;
    }


    const int lastMappingId = mappingList.size() - 1;
    for(int id=lastMappingId; id>=0; --id)
    {
        sofa::core::BaseMapping* m_mapping = mappingList[id];
        const BaseMechanicalState* instate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechFrom()[0]);
        const BaseMechanicalState* outstate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechTo()[0]);

        const defaulttype::BaseMatrix* matrixJ = m_mapping->getJ();
        const unsigned int nbR_J = matrixJ->rowSize();
        const unsigned int nbC_J = matrixJ->colSize();

        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
        {
            std::cout << "[CONTRIBUTION] of " << id << "-th mapping named : " << m_mapping->getName()
                    << " with fromModel : "<< instate->getName()
                    << " and toModel : "   << outstate->getName() <<" -- with matrix J["<<nbR_J <<"."<<nbC_J <<"]" <<std::endl;
        }


        //for toModel -----------------------------------------------------------
        if(mappedMatrices.find(outstate) == mappedMatrices.end())
        {
            if( MULTIMATRIX_VERBOSE)
                std::cout << "	[Propa.Stiff] WARNING toModel : "<< outstate->getName()<< " dont have stiffness matrix"<<std::endl;
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

            defaulttype::BaseMatrix* matrixJJ =const_cast<defaulttype::BaseMatrix*>(matrixJ);

            int JblocRsize     = matrixJJ->getBlockRows();
            int JblocCsize     = matrixJJ->getBlockCols();
            int JnbBlocCol     = matrixJJ->bColSize();

            int K2blocCsize    = K2.matrix->getBlockCols();
            int K2nbBlocCol   = K2.matrix->bColSize();

            int JelementSize   = matrixJJ->getElementSize();
            int MelementSize   = K2.matrix->getElementSize();

            // creating a tempo matrix  tempoMatrix
            defaulttype::BaseMatrix* tempoMatrix = createBlocSparseMatrix(JblocCsize, K2blocCsize, MelementSize, JnbBlocCol, K2nbBlocCol,MULTIMATRIX_VERBOSE);
            // Matrix multiplication  tempoMatrix += Jt * K22
            opAddMulJTM(tempoMatrix,   matrixJJ, K2.matrix,       0,0      , JblocRsize, JblocCsize , K2blocCsize, JelementSize,MelementSize,MULTIMATRIX_VERBOSE);
            // Matrix multiplication  K11         += tempoMatrix * J
            opAddMulMJ( K1.matrix  ,tempoMatrix,  matrixJJ, offset1,offset1, JblocCsize, K2blocCsize, JblocCsize , JelementSize,MelementSize,MULTIMATRIX_VERBOSE);

            delete tempoMatrix;



            if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
            {
                std::cout << "	[Propa.Stiff] propagating stiffness of : "<< outstate->getName()<< " to stifness "<<instate->getName()<<std::endl;
                std::cout <<"	                    **multiplication of "
                        <<" K1["<<sizeK1<<"."<<sizeK1<< "]("<< offset1<<","<<offset1 <<  ")   =  "
                        <<" Jt["<<nbC_J<<"."<<nbR_J<< "] * "
                        <<" K2["<<sizeK2<<"."<<sizeK2<<"]("<< offset2<<","<<offset2 <<  ") * "
                        <<" J["<<nbR_J<<"."<<nbC_J<< "]"
                        <<std::endl;
            }
        }

        std::vector<std::pair<const BaseMechanicalState*, const BaseMechanicalState*> > interactionList;
        for (std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        {
            if(it->first.first == outstate || it->first.second == outstate )
            {
                interactionList.push_back(it->first);
            }
            ++it;
        }


        const unsigned nbInteraction = interactionList.size();
        for(unsigned i=0; i< nbInteraction; i++)
        {

            if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
            {
                std::cout << "	[Propa.Interac.Stiff] detected interaction between toModel : "<<interactionList[i].first->getName()
                        << " and : " <<interactionList[i].second->getName()<<std::endl;
            }
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


                if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
                {
                    std::cout << "	[Propa.Interac.Stiff] propagating interaction "
                            <<outstate->getName() << "--" <<interactionList[i].second->getName()
                            <<"  to : "
                            <<instate->getName() << "--" <<interactionList[i].second->getName()<<std::endl;

                    std::cout <<"	                    **multiplication of "
                            <<" I12["<<nbR_I_12<<"."<<nbC_I_12<< "]("<< offR_I_12<<","<<offC_I_12 <<  ")  =  "
                            <<" Jt["<<nbC_J<<"."<<nbR_J<< "]  *  "
                            <<" I32["<<nbR_I_32<<"."<<nbC_I_32<<"]("<< offR_I_32<<","<<offC_I_32 <<  ")" <<std::endl;
                }

                // Matrix multiplication   I_12 += Jt * I_32

                defaulttype::BaseMatrix* matrixJJ =const_cast<defaulttype::BaseMatrix*>(matrixJ);
                int JblocRsize     = matrixJJ->getBlockRows();
                int JblocCsize     = matrixJJ->getBlockCols();
                int I_32_blocCsize = I_32.matrix->getBlockCols();
                int JelementSize   = matrixJJ->getElementSize();
                int MelementSize   = I_32.matrix->getElementSize();

                opAddMulJTM(I_12.matrix,matrixJJ,I_32.matrix,offR_I_12,offC_I_12, JblocRsize, JblocCsize,I_32_blocCsize,JelementSize,MelementSize,MULTIMATRIX_VERBOSE);
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

                if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
                {
                    std::cout << "	[Propa.Interac.Stiff] propagating interaction "
                            <<interactionList[i].first->getName()<< "--" <<outstate->getName()
                            <<" to : "<<interactionList[i].first->getName()<< "--" <<instate->getName()<<std::endl;

                    std::cout <<"	                    **multiplication of "
                            <<" I_21["<<nbR_I_21<<"."<<nbC_I_21<<"]("<< offR_I_21<<","<<offC_I_21 <<  ")  =  "
                            <<" I23["<<nbR_I_23<<"."<<nbC_I_23<< "]("<< offR_I_23<<","<<offC_I_23 <<  ")  *  "
                            <<" J["<<nbR_J<<"."<<nbC_J<<"]" <<std::endl;
                }


                // Matrix multiplication  I_21 +=  I_23 * J
                defaulttype::BaseMatrix* matrixJJ =const_cast<defaulttype::BaseMatrix*>(matrixJ);
                int I_23_blocRsize = I_23.matrix->getBlockRows();
                int I_23_blocCsize = I_23.matrix->getBlockCols();
                int JblocCsize     = matrixJJ->getBlockCols();
                int JelementSize   = matrixJJ->getElementSize();
                int MelementSize   = I_23.matrix->getElementSize();

                opAddMulMJ(I_21.matrix,I_23.matrix,matrixJJ,offR_I_21,offC_I_21, I_23_blocRsize,I_23_blocCsize,JblocCsize,JelementSize,MelementSize,MULTIMATRIX_VERBOSE);

            }

            //after propagating the interaction, we remove the older interaction
            interactionStiffnessBloc.erase( interactionStiffnessBloc.find(interactionList[i]) );
            if( MULTIMATRIX_VERBOSE)
            {
                std::cout << "	--[Propa.Interac.Stiff] remove interaction of : "<<interactionList[i].first->getName()
                        << " and : " << interactionList[i].second->getName()<<std::endl;
            }

        }//end of interaction loop

    }//end of mapping loop
}

#endif

























//    /// @return the number of rows in each block, or 1 of there are no fixed block size
//    virtual int getBlockRows() const { return NL; }
//    /// @return the number of columns in each block, or 1 of there are no fixed block size
//    virtual int getBlockCols() const { return NC; }
//    /// @return the number of rows of blocks
//    virtual int bRowSize() const { return rowBSize(); }
//    /// @return the number of columns of blocks
//    virtual int bColSize() const { return colBSize(); }


} // namespace linearsolver

} // namespace component

} // namespace sofa
