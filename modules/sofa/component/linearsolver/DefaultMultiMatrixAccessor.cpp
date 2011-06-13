/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/component/linearsolver/DefaultMultiMatrixAccessor.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>


namespace sofa
{

namespace component
{

namespace linearsolver
{

DefaultMultiMatrixAccessor::DefaultMultiMatrixAccessor()
    : globalMatrix(NULL), globalDim(0) ,MULTIMATRIX_VERBOSE(false)
{
}

DefaultMultiMatrixAccessor::~DefaultMultiMatrixAccessor()
{
    clear();
}

void DefaultMultiMatrixAccessor::clear()
{
    globalDim = 0;
    //realStateOffsets.clear();
    for (std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it = realStateOffsets.begin(), itend = realStateOffsets.end(); it != itend; ++it)
        it->second = -1;


    for (std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::iterator it = mappedMatrices.begin(), itend = mappedMatrices.end(); it != itend; ++it)
        if (it->second != NULL) delete it->second;

    mappedMatrices.clear();
    interactionsMappedTree.clear();
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

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "DefaultMultiMatrixAccessor: adding MechanicalState " << mstate->getName() << " in global matrix at offset " << realStateOffsets[mstate] << " with size " << dim << std::endl;
    }
}

void DefaultMultiMatrixAccessor::addMechanicalMapping(sofa::core::BaseMapping* mapping)
{
#ifndef SOFA_SUPPORT_MAPPED_MATRIX
    return;
#endif

    const sofa::defaulttype::BaseMatrix* jmatrix = mapping->getJ();

    if (jmatrix != NULL && mapping->isMechanical())
    {
        mappingList.push_back(mapping);

        if( MULTIMATRIX_VERBOSE)
        {
            std::cout << "DefaultMultiMatrixAccessor: adding validated MechanicalMapping " << mapping->getName() << std::endl;
        }
    }
    else
    {
        std::cout << "	-- Warningg DefaultMultiMatrixAccessor : mapping" << mapping->getName()<<" is not mechanical one or dont have J matrix " << std::endl;
    }
}

void DefaultMultiMatrixAccessor::addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* /*mstate*/)
{
    //do not add the mapped mechanical state here because
    // a mapped mechanical state is added if and only if it has its own stiffness matrix
    // so if and only if there are a forcefield or other component call getMatrix(mappedstate)
    // we add this mappedstate and build its stiffness at the same time
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
        std::cout << "DefaultMultiMatrixAccessor: Global Matrix size = " << globalDim << "x" << globalDim << " with " << realStateOffsets.size() << " models." << std::endl;
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
        //r = diagonalStiffnessBloc.find(mstate)->second;
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
            defaulttype::BaseMatrix* m = this->createMatrix(mstate);
            r.matrix = m;
            r.offset = 0;
            //when creating an matrix, it dont have to be added before
            assert(diagonalStiffnessBloc.find(mstate) == diagonalStiffnessBloc.end());
            //mappedMatrices.insert( std::make_pair(mstate,r.matrix) );
            mappedMatrices[mstate]=r.matrix;
        }
    }

    diagonalStiffnessBloc[mstate] = r;

    if( MULTIMATRIX_VERBOSE)
    {
        if (r.matrix != NULL)
        {
            std::cout << "DefaultMultiMatrixAccessor: giving matrix " << r.matrix->rowSize() << "x" << r.matrix->colSize() << " for real state " << mstate->getName() << " using offset " << r.offset << std::endl;
        }
        else
            std::cout << "DefaultMultiMatrixAccessor: NULL matrix found for state " << mstate->getName() << std::endl;
    }

    return r;
}

DefaultMultiMatrixAccessor::InteractionMatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{
    InteractionMatrixRef r2;
    if (mstate1 == mstate2)// case where state1 == state2, interaction matrix is on the diagonal stiffness bloc
    {
        //MatrixRef r = getMatrix(mstate1);
        MatrixRef r = diagonalStiffnessBloc.find(mstate1)->second;
        r2.matrix = r.matrix;
        r2.offRow = r.offset;
        r2.offCol = r.offset;

        if( MULTIMATRIX_VERBOSE)///////////////////////////////////////////
        {
            if (r2.matrix != NULL)
                std::cout << "DefaultMultiMatrixAccessor: giving bloc matrix "
                        <<" at offset ("<<r2.offRow <<","<< r2.offCol<<") of global matrix"
                        << r2.matrix->rowSize() << "x" << r2.matrix->colSize() << " for self-interaction "
                        <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"<< std::endl;
            else
                std::cout << "DefaultMultiMatrixAccessor: giving NULL matrix for self-interaction "<<
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
                    std::cout << "DefaultMultiMatrixAccessor: giving matrix "
                            <<" at offset ("<<r2.offRow <<","<< r2.offCol<<") "
                            << r2.matrix->rowSize() << "x" << r2.matrix->colSize() << " for interaction "
                            <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"<< std::endl;
                }
                else
                {
                    std::cout << "DefaultMultiMatrixAccessor: giving NULL matrix  for interaction "
                            <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                            <<" at ("<<it->second.offRow <<","<< it->second.offCol<<")"<< std::endl;
                }
            }
        }
        else// the interaction is not added, we need to creat it and its matrix
        {
            std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::const_iterator itms1 = mappedMatrices.find(mstate1);
            std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::const_iterator itms2 = mappedMatrices.find(mstate2);

            if(itms1 == mappedMatrices.end() && itms2 == mappedMatrices.end())// case where all of two ms are real DOF (non-mapped)
            {
                if (globalMatrix)
                {
                    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it1 = realStateOffsets.find(mstate1);
                    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it2 = realStateOffsets.find(mstate2);
                    if (it1 != realStateOffsets.end() && it2 != realStateOffsets.end())
                    {
                        r2.matrix = globalMatrix;
                        r2.offRow = it1->second;
                        r2.offCol = it2->second;

                        if( MULTIMATRIX_VERBOSE)/////////////////////////////////////////////////////////
                        {
                            if (r2.matrix != NULL)
                                std::cout << "DefaultMultiMatrixAccessor: giving bloc matrix "
                                        <<" at offset ("<<r2.offRow <<","<< r2.offCol<<") of global matrix"
                                        << r2.matrix->rowSize() << "x" << r2.matrix->colSize() << " for interaction "
                                        <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"<< std::endl;
                            else
                                std::cout << "DefaultMultiMatrixAccessor: giving NULL matrix  for interaction "
                                        << " for interaction real states"
                                        << mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                                        << std::endl;
                        }
                    }
                }
            }
            else //case where at least one ms is a mapped
            {
                defaulttype::BaseMatrix* m = createInteractionMatrix(mstate1,mstate2);
                r2.matrix = m;
                r2.offRow = 0;
                r2.offCol = 0;
                //when creating an matrix, it dont have to be added before
                assert(interactionStiffnessBloc.find(pairMS) == interactionStiffnessBloc.end());
                interactionsMappedTree.push_back(pairMS);

                if( MULTIMATRIX_VERBOSE)
                {
                    if (r2.matrix != NULL)
                        std::cout << "DefaultMultiMatrixAccessor: giving created matrix "
                                << " with offset ("<<r2.offRow <<","<< r2.offCol<<") size"
                                << r2.matrix->rowSize() << "x" << r2.matrix->colSize() << " for interaction "
                                << mstate1->getName()<< "[" <<mstate1->getMatrixSize()<< "] --- " <<mstate2->getName()<<"[" <<mstate2->getMatrixSize()<<"]"<< std::endl;
                    else
                        std::cout << "DefaultMultiMatrixAccessor: giving NULL matrix  for interaction "
                                << " for interaction real states"
                                << mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                                << std::endl;
                }
            }

            interactionStiffnessBloc[pairMS]=r2;
        }// end of the interaction is not added, we need to creat it and its matrix

    }//end of case where state1 # state2


    if( MULTIMATRIX_VERBOSE)
    {
        if(r2.matrix == NULL)
            std::cout << "DefaultMultiMatrixAccessor: NULL matrix found for interaction " << mstate1->getName()<<" --- "<<mstate2->getName() << std::endl;
    }

    return r2;
}

void DefaultMultiMatrixAccessor::computeGlobalMatrix()
{

}

defaulttype::BaseMatrix* DefaultMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* /*mstate*/) const
{
    /// @TODO support for mapped matrices
    return NULL;
}

defaulttype::BaseMatrix* DefaultMultiMatrixAccessor::createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* /*mstate1*/, const sofa::core::behavior::BaseMechanicalState* /*mstate2*/) const
{
    /// @TODO support for mapped matrices
    return NULL;
}




//TODO separating in other file
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
defaulttype::BaseMatrix* MappedMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const
{
    // A diagonal stiffness matrix is added if and only if it doenst exist
    assert(mappedMatrices.find(mstate) == mappedMatrices.end() );

    component::linearsolver::FullMatrix<SReal>* m = new component::linearsolver::FullMatrix<SReal>;
    m->resize( mstate->getMatrixSize(),mstate->getMatrixSize());

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "		++ creating and adding matrix["<< m->rowSize() <<"x"<< m->colSize() <<"]   for mapped state " << mstate->getName() << "[" << mstate->getMatrixSize()<<"]"<< std::endl;
    }

    return m;
}

defaulttype::BaseMatrix* MappedMultiMatrixAccessor::createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{

    component::linearsolver::FullMatrix<SReal>* m = new component::linearsolver::FullMatrix<SReal>;
    m->resize( mstate1->getMatrixSize(),mstate2->getMatrixSize() );

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "		++ creating interraction matrix["<< m->rowSize() <<"x"<< m->colSize()
                << "] for interaction " << mstate1->getName() << "[" << mstate1->getMatrixSize()
                << "] --- "             << mstate2->getName() << "[" << mstate2->getMatrixSize()<<"]" <<std::endl;
    }

    return m;
}

void MappedMultiMatrixAccessor::computeGlobalMatrix()
{
    if( MULTIMATRIX_VERBOSE)
    {
        const int lastMappingId = mappingList.size() - 1;
        std::cout << "================  MappedMultiMatrixAccessor: "<<lastMappingId +1 <<" validated mappings registered ================" <<std::endl;
        for(int id=lastMappingId; id>=0; --id)
        {
            std::cout << "                mapping "<<id<< "-th  :"<<mappingList[id]->getName() <<" registered in list" <<std::endl;
        }
        std::cout << "=============================================================================================" <<std::endl;
    }

#if 0
    const int lastMappingId = mappingList.size() - 1;
    for(int id=lastMappingId; id>=0; --id)
    {
        sofa::core::BaseMapping* m_mapping = mappingList[id];
        const BaseMechanicalState* instate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechFrom()[0]);
        const BaseMechanicalState* outstate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechTo()[0]);
        const defaulttype::BaseMatrix* matrixJ = m_mapping->getJ();

        if( MULTIMATRIX_VERBOSE)
        {
            std::cout << " contribution of " << id << "-th mapping named : " << m_mapping->getName()
                    << " with fromModel : "<< instate->getName()
                    << " and toModel : "   << outstate->getName() <<" --" <<std::endl;
        }


        //for toModel -----------------------------------------------------------
        if(mappedMatrices.find(outstate) == mappedMatrices.end())
        {
            if( MULTIMATRIX_VERBOSE)
                std::cout << "	WARNING toModel : "<< outstate->getName()<< " not found in local data registed"<<std::endl;
        }



//		if(diagonalStiffnessBloc.find(outstate) != diagonalStiffnessBloc.end())
//		{
//			//    K11 += Jt * K22 * J       //
//			MatrixRef K1 = this->getMatrix(instate);
//			MatrixRef K2 = this->getMatrix(outstate);
//
//			const unsigned int sizeK1 = K1.matrix->rowSize(); //instate->getMatrixSize();
//			const unsigned int sizeK2 = K2.matrix->rowSize(); //outstate->getMatrixSize();
//
//			const unsigned int offset1 = K1.offset;
//			const unsigned int offset2 = K2.offset;
//
//			for(unsigned int i1 =0 ; i1 < sizeK1 ; ++i1)
//			{
//				for(unsigned int j1 =0 ; j1 < sizeK1 ; ++j1)
//				{
//					double Jt_K2_J_i1j1 = 0;
//
//					for(unsigned int i2 =0 ; i2 < sizeK2 ; ++i2)
//					{
//						for(unsigned int j2 =0 ; j2 < sizeK2 ; ++j2)
//						{
//							const double K2_i2j2 = (double) K2.matrix->element(offset2 + i2, offset2 + j2);
//							for(unsigned int k2=0 ; k2 < sizeK2 ; ++k2)
//							{
//								const double Jt_i1k2 = (double) matrixJ->element( i1 , k2 ) ;
//								const double  J_k2j1 = (double) matrixJ->element( k2 , j1 ) ;
//
//								Jt_K2_J_i1j1 += Jt_i1k2 * K2_i2j2  * J_k2j1;
//							}
//						}
//					}
//
//					K1.matrix->add(offset1 + i1 , offset1 + j1 , Jt_K2_J_i1j1);
//				}
//			}
//
//			if( MULTIMATRIX_VERBOSE)
//				std::cout << "	toModel : "<< outstate->getName()<< " has itself stiffness"<<std::endl;
//		}

        if( MULTIMATRIX_VERBOSE)
        {

            std::vector<std::pair<const BaseMechanicalState*, const BaseMechanicalState*> > interactionList;
            std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin();
            std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator itEnd = interactionStiffnessBloc.end();
            while(it !=itEnd)
            {
                if(it->first.first == outstate || it->first.second == outstate )
                {
                    interactionList.push_back(it->first);
                }
                ++it;
            }

            for(unsigned i=0; i< interactionList.size(); i++)
            {
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
                    //              ..........
                    //===========================
                }
                if(interactionList[i].second == outstate)
                {
                    InteractionMatrixRef I23 = this->getMatrix(interactionList[i].first,outstate);
                    InteractionMatrixRef I21 = this->getMatrix(interactionList[i].first,instate);
                    //===========================
                    //          I_21 +=      I_23 * J
                    //              ..........
                    //===========================
                }

                //after propagating the interaction, we remove the older interaction
                interactionStiffnessBloc.erase( interactionStiffnessBloc.find(interactionList[i]) );


                std::cout << "			interaction between toModel : "<<interactionList[i].first->getName()
                        << " and the state : "                       <<interactionList[i].second->getName()<<std::endl;
            }
        }

























//		const int nbInteraction = interactionsMappedTree.size();
//		for(int id=0;id<nbInteraction;id++)
//		{
//			if (interactionsMappedTree[id].first  == outstate)
//			{
//				std::cout << "	toModel : "<< outstate->getName()<< " has interaction stiffness with : "
//						<< interactionsMappedTree[id].second->getName()<<std::endl;
//			}
//			if (interactionsMappedTree[id].second == outstate)
//			{
//				std::cout << "	toModel : "<< outstate->getName()<< " has interaction stiffness with : "
//						<< interactionsMappedTree[id].first->getName()<<std::endl;
//			}
//		}



//		//test for fromModel -----------------------------------------------------------
//		if (realStateOffsets.find(instate) != realStateOffsets.end()) //case where fromModel is a non mapped state
//		{
//			std::cout << "	fromModel : "<< instate->getName()<< " found in real MS registed list"<<std::endl;
//		}
//		else if(mappedMatrices.find(instate) != mappedMatrices.end()) //case where fromModel is a mapped state
//		{
//			std::cout << "	fromModel : "<< instate->getName()<< " found in mapped MS registed list"<<std::endl;
//		}
//		else
//		{
//			std::cout << "	fromModel : "<< instate->getName()<< " not found in local data registed"<<std::endl;
//		}





    }

#endif



}



} // namespace linearsolver

} // namespace component

} // namespace sofa
