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

    mappingBottomUpTree.clear();
    mappingTopDownTree.clear();

    for (std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::iterator it = mappedMatrices.begin(), itend = mappedMatrices.end(); it != itend; ++it)
        if (it->second != NULL) delete it->second;
    mappedMatrices.clear();

    interactionsMappedTree.clear();

    diagonalStiffnessBloc.clear();

    for (std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        if (it->second.matrix != NULL && it->second.matrix != globalMatrix) delete it->second.matrix;
    interactionStiffnessBloc.clear();

    buff12 = NULL;
    buff21 = NULL;
}

void DefaultMultiMatrixAccessor::setGlobalMatrix(defaulttype::BaseMatrix* matrix)
{
    this->globalMatrix = matrix;
}

void DefaultMultiMatrixAccessor::addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
//    std::cout << "DefaultMultiMatrixAccessor: added state " << mstate->getName() <<  std::endl;/////////////////////////

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
//    std::cout << "DefaultMultiMatrixAccessor: added mapping " << mapping->getName() <<  std::endl;///////////////////

    const sofa::defaulttype::BaseMatrix* jmatrix = mapping->getJ();

    if (jmatrix != NULL && mapping->isMechanical())
    {
        const sofa::core::behavior::BaseMechanicalState* instate  = const_cast<const sofa::core::behavior::BaseMechanicalState*>(mapping->getMechFrom()[0]);
        const sofa::core::behavior::BaseMechanicalState* outstate  = const_cast<const sofa::core::behavior::BaseMechanicalState*>(mapping->getMechTo()[0]);

        //if the input of the mapping is a non-mapped state, this mapping will be added for the matrix contribution
        std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator itRootState = realStateOffsets.find(instate);
        if(itRootState != realStateOffsets.end())
        {
            mappingBottomUpTree[outstate] = mapping;
            mappingTopDownTree[instate]   = mapping;
        }
        else
        {
            //if the input of the mapping is a mapped state, this input must be already registered as a output of another mapping in the mappingTree
            std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::const_iterator itmappedState = mappingBottomUpTree.find(instate);

            if( itmappedState != mappingBottomUpTree.end() )
            {
                mappingBottomUpTree[outstate] = mapping;
                mappingTopDownTree[instate]   = mapping;
            }
        }

        if( MULTIMATRIX_VERBOSE)
        {
            std::cout << "DefaultMultiMatrixAccessor: adding MechanicalMapping " << mapping->getName() << std::endl;
        }
    }
}

void DefaultMultiMatrixAccessor::addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
//	mappedMatrices[mstate] = NULL;
//
//    if( MULTIMATRIX_VERBOSE)
//    {
//    	std::cout << "DefaultMultiMatrixAccessor: adding MappedState " << mstate->getName() << " to a NULL_matrix" << std::endl;
//	}

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
        r = diagonalStiffnessBloc.find(mstate)->second;
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
            r.matrix = this->createMatrix(mstate);
            r.offset = 0;
            //when creating an matrix, it dont have to be added before
            assert(diagonalStiffnessBloc.find(mstate) == diagonalStiffnessBloc.end());
            mappedMatrices.insert( std::make_pair(mstate,r.matrix) );
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
                r2.matrix = createInteractionMatrix(mstate1,mstate2);
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
        std::cout << "MappedMultiMatrixAccessor: ++ creating and adding matrix["<< m->rowSize() <<"x"<< m->colSize() <<"]   for mapped state " << mstate->getName() << "[" << mstate->getMatrixSize()<<"]"<< std::endl;
    }

    return m;
}

defaulttype::BaseMatrix* MappedMultiMatrixAccessor::createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{

    component::linearsolver::FullMatrix<SReal>* m = new component::linearsolver::FullMatrix<SReal>;
    m->resize( mstate1->getMatrixSize(),mstate2->getMatrixSize() );

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "MappedMultiMatrixAccessor: ++ creating interraction matrix["<< m->rowSize() <<"x"<< m->colSize()
                << "] for interaction " << mstate1->getName() << "[" << mstate1->getMatrixSize()
                << "] --- "             << mstate2->getName() << "[" << mstate2->getMatrixSize()<<"]" <<std::endl;
    }

    return m;
}

void MappedMultiMatrixAccessor::computeGlobalMatrix()
{
    //test if the two tree has added the same mappings
//	for(std::map< const BaseMechanicalState*, sofa::core::BaseMapping* >::iterator itMapping = mappingBottomUpTree.begin(),itEnd = mappingBottomUpTree.end();itMapping!=itEnd;itMapping++)
//    {
//		std::cout << " ============  mappingBottomUpTree "<<itMapping->second->getName()
//				  << " stateTo:"<<itMapping->first->getName()
//				  <<std::endl;
//    }
//
//    for(std::map< const BaseMechanicalState*, sofa::core::BaseMapping* >::iterator itMapping = mappingTopDownTree.begin(),itEnd = mappingTopDownTree.end();itMapping!=itEnd;itMapping++)
//    {
//		std::cout << " ============  mappingTopDown "<<itMapping->second->getName()
//				  << " stateFrom:"<<itMapping->first->getName()
//				  <<std::endl;
//    }

    // cleaning the mappingBottomUpTree, the case where stiffness matrix of the mapped has never been created,
    // this mechanical state can not propagate down to bottom,
    // we delete its mapping before doing the propagation

    std::map< const BaseMechanicalState*, sofa::core::BaseMapping* >::iterator itMapping = mappingBottomUpTree.begin();
    std::map< const BaseMechanicalState*, sofa::core::BaseMapping* >::iterator itMappingend = mappingBottomUpTree.end();
    std::map< const BaseMechanicalState*, const BaseMechanicalState*> mappingsToDelete;
    while(itMapping != itMappingend)
    {
        //if the toModel its not found in the mapped mechanical state tree
        //we delete it from the mapping tree
        if(mappedMatrices.find(itMapping->first) == mappedMatrices.end() )
        {
            BaseMechanicalState* outstate = const_cast<BaseMechanicalState*>(itMapping->first);
            BaseMechanicalState* instate = itMapping->second->getMechFrom()[0];

            mappingsToDelete.insert( std::make_pair(instate,outstate) );

            if( MULTIMATRIX_VERBOSE)
            {
                std::cout << " -- MappedMultiMatrixAccessor: MAPPING to be removed "<<itMapping->second->getName()
                        << " because the mapped state : "<<itMapping->first->getName()
                        << " dont have stiffness "<<std::endl;
            }


            while (mappingTopDownTree.find(outstate) != mappingTopDownTree.end() )
            {
                sofa::core::BaseMapping* _mappingToDelete = mappingTopDownTree.find(outstate)->second;
                outstate = _mappingToDelete->getMechTo()[0];
                instate  = _mappingToDelete->getMechFrom()[0];
                mappingsToDelete.insert( std::make_pair(instate,outstate) );

                if( MULTIMATRIX_VERBOSE)
                {
                    std::cout << " -- MappedMultiMatrixAccessor: MAPPING to be removed by propagation top down "<<_mappingToDelete->getName()
                            << " stateFrom:"<<instate->getName()<<"   stateTo:"<<outstate->getName()
                            <<std::endl;
                }
            }
        }
        ++itMapping;
    }

    for(std::map< const BaseMechanicalState*, const BaseMechanicalState*>::iterator itdel = mappingsToDelete.begin(),itdelEnd = mappingsToDelete.end(); itdel!=itdelEnd; itdel++)
    {
        if( MULTIMATRIX_VERBOSE)
        {
            std::cout << " -- : MAPPING removed "
                    << " stateFrom:"<<itdel->second->getName()<<"   stateTo:"<<itdel->first->getName()
                    <<"    mapping in UpTree:"<<mappingBottomUpTree.find(itdel->second)->second->getName()
                    <<"    mappingin DownTree:"<<mappingTopDownTree.find(itdel->first)->second->getName()
                    <<std::endl;
        }
        mappingTopDownTree.erase(itdel->first);
        mappingBottomUpTree.erase(itdel->second);
    }



    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //       DIAGONAL STIFFNESS BLOG      DIAGONAL STIFFNESS BLOG      DIAGONAL STIFFNESS BLOG               //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::reverse_iterator ritMapping;
    std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::reverse_iterator _itBegin = mappingBottomUpTree.rbegin();
    std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::reverse_iterator   _itEnd = mappingBottomUpTree.rend();
    for(ritMapping = _itBegin; ritMapping != _itEnd; ++ritMapping)
    {
        const sofa::core::behavior::BaseMechanicalState* mstate1 = ritMapping->second->getMechFrom()[0];
        const sofa::core::behavior::BaseMechanicalState* mstate2 = ritMapping->second->getMechTo()[0];

        std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::const_iterator itmapped = mappedMatrices.find(mstate2);

        // compute contribution only if the mapped matrix K2 is filled
        if(itmapped != mappedMatrices.end() && itmapped->second != NULL )
        {

            MatrixRef K1 = diagonalStiffnessBloc[mstate1];
            MatrixRef K2 = diagonalStiffnessBloc[mstate2];

            const defaulttype::BaseMatrix* matrixJ = ritMapping->second->getJ();

            const unsigned int sizeK1 = mstate1->getMatrixSize();
            const unsigned int sizeK2 = mstate2->getMatrixSize();

            const unsigned int offset1 = K1.offset;
            const unsigned int offset2 = K2.offset;

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

            if( MULTIMATRIX_VERBOSE)
            {
                std::cout << "MappedMultiMatrixAccessor: MAPPING Registered "<<ritMapping->second->getName() ;

                std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix*>::iterator itmapped = mappedMatrices.find(mstate1);
                if (itmapped != mappedMatrices.end())
                {
                    // this state is mapped
                    std::cout <<     " mapped matrix _K1[" << K1.matrix->rowSize() <<"x" << K1.matrix->colSize() <<"]";
                }
                else
                {
                    std::cout <<     " local real DOF matrix K1[" << K1.matrix->rowSize() <<"x" << K1.matrix->colSize()
                            <<     "] in global matrix K["<<globalMatrix->rowSize()<<"x"<<globalMatrix->colSize()<<"] at offset "<<K1.offset;
                }

                std::cout << "    mapped matrix K2[" << K2.matrix->rowSize() <<"x" << K2.matrix->colSize() <<"]"
                        << "     J[" <<   matrixJ->rowSize() <<"x" <<   matrixJ->colSize() <<"]"
                        <<std::endl;
            }

        }
    }






//	///////////////////////////////////////////////////////////////////////////////////////////////////////////
//    //    INTERACTION STIFFNESS BLOG      INTERACTION STIFFNESS BLOG      INTERACTION STIFFNESS BLOG         //
//    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	int lastIter = interactionsMappedTree.size() -1 ;
//	for(int iIter = lastIter ; iIter >-1 ; --iIter)
//	{
//
//		BaseMechanicalState* mstate33 = const_cast<BaseMechanicalState*>(interactionsMappedTree[iIter].first);
//		BaseMechanicalState* mstate44 = const_cast<BaseMechanicalState*>(interactionsMappedTree[iIter].second);
//
//		BaseMechanicalState* mstate11 = mstate33;
//		BaseMechanicalState* mstate22 = mstate44;
//
//
//	    if(realStateOffsets.find(mstate33) != realStateOffsets.end() && mappedMatrices.find(mstate44) != mappedMatrices.end())
//		{
//			//case where interation are between non mapped states and mapped state
//
//
//			if( MULTIMATRIX_VERBOSE)
//			{
//				std::cout <<"MappedMultiMatrixAccessor: INTERACTION Registered between "
//						<< "  nonMAPPED " <<mstate33->getName()
//						<< " --- MAPPED " <<mstate44->getName() <<std::endl;
//			}
//		}
//		else if(mappedMatrices.find(mstate33) != mappedMatrices.end() && realStateOffsets.find(mstate44) != realStateOffsets.end())
//		{
//			//case where interation are between mapped states and non mapped state
//
//			if( MULTIMATRIX_VERBOSE)
//			{
//				std::cout <<"MappedMultiMatrixAccessor: INTERACTION Registered between "
//						<< "  MAPPED " <<mstate33->getName()
//						<< " --- nonMAPPED " <<mstate44->getName() <<std::endl;
//			}
//		}
//		else if(mappedMatrices.find(mstate33) != mappedMatrices.end() && mappedMatrices.find(mstate44) != mappedMatrices.end())
//		{
//			//case where interation are between mapped states
//
//
//			if( MULTIMATRIX_VERBOSE)
//			{
//				std::cout <<"MappedMultiMatrixAccessor: INTERACTION Registered between "
//						<< "  MAPPED " <<mstate33->getName()
//						<< " --- MAPPED " <<mstate44->getName() <<std::endl;
//			}
//		}
//		else
//		{
//			// case where interation are between non mapped states (or non registered states in propagation tree)
//		    // nothing to do for the proparation
//
//
//			if( MULTIMATRIX_VERBOSE)
//			{
//				std::cout <<"MappedMultiMatrixAccessor: INTERACTION Registered between "
//						<<mstate33->getName() <<" --- "
//						<<mstate44->getName() <<" not need propagation "<<std::endl;
//			}
//		}
//
//
//
//
//
//
//
//
//
//
//
//	    //////////////////////////////////////////////////////////////////////////////////////
//	    //////////////////////////////////////////////////////////////////////////////////////
//	    //////////////////////////////////////////////////////////////////////////////////////
//	    //////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//
//		if( MULTIMATRIX_VERBOSE)
//		{
//			std::cout <<std::endl<<"=====================================:        registered interaction beetween "
//					<< mstate33->getName() <<"--" <<mstate44->getName() <<std::endl;
//		}
//
//
//		if(realStateOffsets.find(mstate33) != realStateOffsets.end())
//		{
//			if(mappingBottomUpTree.find(mstate44) == mappingBottomUpTree.end())
//			{
//
//			}
//			else
//			{
//
//			}
//
//		}
//		else
//		{
//			while(mappingBottomUpTree.find(mstate33) != mappingBottomUpTree.end() )
//			{
//				mstate11 = mappingBottomUpTree.find(mstate33)->second->getMechFrom()[0];
//
//				while(mappingBottomUpTree.find(mstate44) != mappingBottomUpTree.end())
//				{
//					mstate22 = mappingBottomUpTree.find(mstate44)->second->getMechFrom()[0];
//
//
//
//				////////////////////////////////////////////////////////////////////////////////////////
//				if( MULTIMATRIX_VERBOSE)
//				{
//					std::cout <<"                                              PROPAGATION, INTERACTION beetween   FROM "
//							<< mstate33->getName() <<"--" <<mstate44->getName() <<"               TO   "
//							<< mstate11->getName() <<"--" <<mstate22->getName() <<std::endl;
//				}
//				////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//					mstate44 = mstate22;
//				}
//
//				mstate33 = mstate11;
//			}
//		}
//
//		if( MULTIMATRIX_VERBOSE)
//		{
//			std::cout <<std::endl<<"=====================================:    interaction after propagation beetween "
//					<< mstate33->getName() <<"--" <<mstate44->getName() <<std::endl;
//		}
//
//	}




}



} // namespace linearsolver

} // namespace component

} // namespace sofa
