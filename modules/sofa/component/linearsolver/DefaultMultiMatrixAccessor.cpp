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

//#define MULTIMATRIX_VERBOSE

namespace sofa
{

namespace component
{

namespace linearsolver
{

DefaultMultiMatrixAccessor::DefaultMultiMatrixAccessor()
    : globalMatrix(NULL), globalDim(0)
{
}

DefaultMultiMatrixAccessor::~DefaultMultiMatrixAccessor()
{
    clear();
}

void DefaultMultiMatrixAccessor::clear()
{
    globalDim = 0;
    //globalOffsets.clear();
    for (std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it = globalOffsets.begin(), itend = globalOffsets.end(); it != itend; ++it)
        it->second = -1;

    for (std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::iterator it = mappedMatrices.begin(), itend = mappedMatrices.end(); it != itend; ++it)
        if (it->second != NULL) delete it->second;
    mappedMatrices.clear();
}

void DefaultMultiMatrixAccessor::setGlobalMatrix(defaulttype::BaseMatrix* matrix)
{
    this->globalMatrix = matrix;
}

void DefaultMultiMatrixAccessor::addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
//    std::cout << "DefaultMultiMatrixAccessor: added state " << mstate->getName() <<  std::endl;/////////////////////////

    unsigned int dim = mstate->getMatrixSize();
    globalOffsets[mstate] = globalDim;
    globalDim += dim;
#ifdef MULTIMATRIX_VERBOSE
    std::cout << "DefaultMultiMatrixAccessor: added state " << mstate->getName() << " at offset " << globalOffsets[mstate] << " size " << dim << std::endl;
#endif
}

void DefaultMultiMatrixAccessor::addMechanicalMapping(sofa::core::BaseMapping* mapping)
{
//    std::cout << "DefaultMultiMatrixAccessor: added mapping " << mapping->getName() <<  std::endl;///////////////////

    const sofa::defaulttype::BaseMatrix* jmatrix = mapping->getJ();
    if (jmatrix == NULL ||!mapping->isMechanical())
    {
        //The case where mapping is not a machanical mapping or do not construct the J matrix
        //This mapping will not contribute to the matrix construction
        mappingsContribution[mapping] = false ;
    }
    else
    {
        const sofa::core::behavior::BaseMechanicalState* instate  = const_cast<const sofa::core::behavior::BaseMechanicalState*>(mapping->getMechFrom()[0]);
        std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator itRootState = globalOffsets.find(instate);

        //if the input of the mapping is a non-mapped state, this mapping will be added for the matrix contribution
        if(itRootState != globalOffsets.end())
        {
            const sofa::core::behavior::BaseMechanicalState* outstate  = const_cast<const sofa::core::behavior::BaseMechanicalState*>(mapping->getMechTo()[0]);
            mappingsContribution[mapping] = true;
            mappingsTree[outstate] = mapping;
            return;
        }

        std::map< const sofa::core::behavior::BaseMechanicalState*,  sofa::core::BaseMapping*>::const_iterator itmappedState = mappingsTree.find(instate);
        if( itmappedState == mappingsTree.end() )
        {
            //if the input state of the mapping not found in one of the root states
            //neither in the one of mapped state in the mappingTree, this mapping will not added for the matrix contribution
            mappingsContribution[mapping] = false;
            return;
        }
        else
        {
            sofa::core::BaseMapping* parentMapping = itmappedState->second;
            bool contributiveMapping = mappingsContribution[parentMapping];
            if(contributiveMapping )
            {
                const sofa::core::behavior::BaseMechanicalState* outstate  = const_cast<const sofa::core::behavior::BaseMechanicalState*>(mapping->getMechTo()[0]);
                mappingsContribution[mapping] = true;
                mappingsTree[outstate] = mapping;
                return;
            }
            else
            {
                mappingsContribution[mapping] = false;
                return;
            }
        }

    }
}

void DefaultMultiMatrixAccessor::addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
//    std::cout << "DefaultMultiMatrixAccessor: added MAPPED state " << mstate->getName() <<  std::endl;/////////////////////////
    /// @TODO support for mapped matrices
    std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::const_iterator it = mappedMatrices.find(mstate);
    if(it == mappedMatrices.end() )
    {
        mappedMatrices[mstate] = NULL;
#ifdef MULTIMATRIX_VERBOSE
        std::cout <<  "DefaultMultiMatrixAccessor: added mapped state "  << mstate->getName() << " at a mapped matrix[NULL_matrix]" << std::endl;
#endif
    }
    else
    {
#ifdef MULTIMATRIX_VERBOSE
        std::cout <<  "DefaultMultiMatrixAccessor: mapped state "  << mstate->getName() << " already added at a mapped matrix[NULL_matrix]" << std::endl;
#endif
        return;
    }
}

void DefaultMultiMatrixAccessor::setupMatrices()
{
    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it = globalOffsets.begin(), itend = globalOffsets.end();
    while (it != itend)
    {
        if (it->second < 0)
        {
            std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it2 = it;
            ++it;
            globalOffsets.erase(it2);
        }
        else
        {
            if (globalMatrix)
            {
                MatrixRef& r = localMatrixMap[it->first];
                r.matrix = globalMatrix;
                r.offset = it->second;
            }
            ++it;
        }
    }
#ifdef MULTIMATRIX_VERBOSE
    std::cout << "DefaultMultiMatrixAccessor: Global Matrix size = " << globalDim << "x" << globalDim << " with " << globalOffsets.size() << " models." << std::endl;
#endif
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
    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it = globalOffsets.find(mstate);
    if (it != globalOffsets.end())
        return it->second;
    return -1;
}

DefaultMultiMatrixAccessor::MatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const
{
    std::map< const sofa::core::behavior::BaseMechanicalState*, MatrixRef >::const_iterator itlocal = localMatrixMap.find(mstate);
    if (itlocal != localMatrixMap.end())
    {
        MatrixRef r = itlocal->second;
#ifdef MULTIMATRIX_VERBOSE
        if (r.matrix != NULL)
        {
            std::cout << "DefaultMultiMatrixAccessor: valid " << r.matrix->rowSize() << "x" << r.matrix->colSize() << " matrix found for " << mstate->getName() << " using offset " << r.offset << std::endl;
        }
        else
            std::cout << "DefaultMultiMatrixAccessor: NULL matrix found for " << mstate->getName() << std::endl;
#endif
        return r;
    }

    std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix*>::iterator itmapped = mappedMatrices.find(mstate);
    if (itmapped != mappedMatrices.end()) // this state is mapped
    {
        if (itmapped->second == NULL) // we need to create the matrix
        {
            itmapped->second = createMatrix(mstate);
#ifdef MULTIMATRIX_VERBOSE
            if (itmapped->second != NULL)
                std::cout << "DefaultMultiMatrixAccessor: Mapped state " << mstate->getName() << " " << itmapped->second->rowSize() << "x" << itmapped->second->colSize() << " matrix already created." << std::endl;
            else
                std::cout << "DefaultMultiMatrixAccessor: Mapped state " << mstate->getName() << " ignored. It will not contribute to the final mechanical matrix." << std::endl;
#endif
        }
        MatrixRef ref;
        if (itmapped->second != NULL)
        {
            ref.matrix = itmapped->second;
            ref.offset = 0;
        }
        return ref;
    }


#ifdef MULTIMATRIX_VERBOSE
    std::cout << "DefaultMultiMatrixAccessor: NO matrix found for " << mstate->getName() << std::endl;
#endif

    return MatrixRef();
}

DefaultMultiMatrixAccessor::InteractionMatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{
    if (mstate1 == mstate2)
    {
        MatrixRef r = getMatrix(mstate1);
        InteractionMatrixRef r2;
        r2.matrix = r.matrix;
        r2.offRow = r.offset;
        r2.offCol = r.offset;
#ifdef MULTIMATRIX_VERBOSE
        if (r.matrix != NULL)
            std::cout << "DefaultMultiMatrixAccessor: valid " << r.matrix->rowSize() << "x" << r.matrix->colSize() << " matrix found for self-interaction "
                    <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                    <<" at ("<<r2.offRow <<","<< r2.offCol<<")"<< std::endl;
        else
            std::cout << "DefaultMultiMatrixAccessor: NULL matrix found for self-interaction "<<
                    mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" << std::endl;
#endif
        return r2;
    }

    std::map< std::pair<const sofa::core::behavior::BaseMechanicalState*,const sofa::core::behavior::BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionMatrixMap.find(std::make_pair(mstate1,mstate2));
    if (it != interactionMatrixMap.end())
    {
        if(it->second.matrix != NULL)
        {
#ifdef MULTIMATRIX_VERBOSE
            std::cout << "DefaultMultiMatrixAccessor: valid " << it->second.matrix->rowSize() << "x" << it->second.matrix->colSize() << " matrix found for interaction "
                    <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                    <<" at ("<<it->second.offRow <<","<< it->second.offCol<<")"<< std::endl;
#endif
            return it->second;
        }
        else
        {
            it->second.matrix = createInteractionMatrix(mstate1,mstate2);
            it->second.offRow = 0;
            it->second.offCol = 0;
            return it->second;
        }
    }
    else
    {
        std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::const_iterator itms1 = mappedMatrices.find(mstate1);
        std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::const_iterator itms2 = mappedMatrices.find(mstate2);


        if(itms1 == mappedMatrices.end() && itms2 == mappedMatrices.end())// case where all of two ms are real DOF (non-mapped)
        {
            if (globalMatrix)
            {
                std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it1 = globalOffsets.find(mstate1);
                std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator it2 = globalOffsets.find(mstate2);
                if (it1 != globalOffsets.end() && it2 != globalOffsets.end())
                {
                    InteractionMatrixRef r;
                    r.matrix = globalMatrix;
                    r.offRow = it1->second;
                    r.offCol = it2->second;
                    it->second = r;
#ifdef MULTIMATRIX_VERBOSE
                    if (r.matrix != NULL)
                        std::cout << "DefaultMultiMatrixAccessor: set " << r.matrix->rowSize() << "x" << r.matrix->colSize() << " matrix for interaction "
                                <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]"
                                <<" at ("<<r.offRow <<","<< r.offCol<<")"<< std::endl;
#endif
                    return r;
                }
            }
        }
        else //case where at least one ms is a mapped
        {
            defaulttype::BaseMatrix * m = createInteractionMatrix(mstate1,mstate2);
            InteractionMatrixRef r;
            r.matrix = m;
            r.offRow = 0;
            r.offCol = 0;
            it->second = r;
            return r;
        }
    }

#ifdef MULTIMATRIX_VERBOSE
    std::cout << "DefaultMultiMatrixAccessor: NO matrix found for interaction " << mstate1->getName()<<" --- "<<mstate2->getName() << std::endl;
#endif
    return InteractionMatrixRef();
}

void DefaultMultiMatrixAccessor::computeGlobalMatrix()
{
    /// @TODO support for mapped matrices
    //if (globalMatrix)
    //    std::cout << "DefaultMultiMatrixAccessor: final matrix: " << *globalMatrix << std::endl;
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
    /// @TODO support for mapped matrices
    component::linearsolver::FullMatrix<SReal>* m = new component::linearsolver::FullMatrix<SReal>;
    m->resize( mstate->getMatrixSize(),mstate->getMatrixSize());

#ifdef MULTIMATRIX_VERBOSE
    std::cout << "MappedMultiMatrixAccessor: ++ creating matrix["<< m->rowSize() <<"x"<< m->colSize() <<"]   for state " << mstate->getName() << "[" << mstate->getMatrixSize()<<"]"<< std::endl;
#endif

    return m;
}

defaulttype::BaseMatrix* MappedMultiMatrixAccessor::createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{
    /// @TODO support for mapped matrices
    component::linearsolver::FullMatrix<SReal>* m = new component::linearsolver::FullMatrix<SReal>;
    m->resize( mstate1->getMatrixSize(),mstate2->getMatrixSize() );

#ifdef MULTIMATRIX_VERBOSE
    std::cout << "MappedMultiMatrixAccessor: ++ creating interraction matrix["<< m->rowSize() <<"x"<< m->colSize()
            <<"] for interaction " << mstate1->getName() << "[" << mstate1->getMatrixSize()
            <<"] --- "             << mstate2->getName() << "[" << mstate2->getMatrixSize()<<"]" <<std::endl;
#endif

    return m;
}

void MappedMultiMatrixAccessor::computeGlobalMatrix()
{
#ifdef MULTIMATRIX_VERBOSE
#endif
    for(std::map<sofa::core::BaseMapping*, bool>::const_reverse_iterator rit = mappingsContribution.rbegin(); rit != mappingsContribution.rend(); ++rit)
    {
//			std::cout << "MappedMultiMatrixAccessor: ----- registered mechanical mapping : "<< rit->first->getName();
//		if(rit->second)
//			std::cout << " TRUE ";
//		else
//			std::cout << " FALSE ";
//		std::cout << " inputState "<< rit->first->getMechFrom()[0]->getName()
//				  << " outputState "<< rit->first->getMechTo()[0]->getName()
//				  <<std::endl;

    }
}



} // namespace linearsolver

} // namespace component

} // namespace sofa
