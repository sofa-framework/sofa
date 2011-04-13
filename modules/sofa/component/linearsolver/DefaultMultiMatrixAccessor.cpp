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

    interactionMappedTree.clear();

    diagonalStiffnessBloc.clear();

    for (std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        if (it->second.matrix != NULL && it->second.matrix != globalMatrix) delete it->second.matrix;
    interactionStiffnessBloc.clear();
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
            mappingsContributionTree[outstate] = mapping;
        }
        else
        {
            //if the input of the mapping is a mapped state, this input must be already registered as a output of another mapping in the mappingTree
            std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::const_iterator itmappedState = mappingsContributionTree.find(instate);

            if( itmappedState != mappingsContributionTree.end() )
            {
                mappingsContributionTree[outstate] = mapping;
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
    mappedMatrices[mstate] = NULL;

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "DefaultMultiMatrixAccessor: adding MappedState " << mstate->getName() << " in a NULL_matrix" << std::endl;
    }

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

    //case where mechanical state is a non mapped state
    std::map< const sofa::core::behavior::BaseMechanicalState*, int >::const_iterator itRealState = realStateOffsets.find(mstate);
    if (itRealState != realStateOffsets.end())
    {
        r = diagonalStiffnessBloc.find(mstate)->second;

        if( MULTIMATRIX_VERBOSE)
        {
            if (r.matrix != NULL)
            {
                std::cout << "DefaultMultiMatrixAccessor: giving matrix " << r.matrix->rowSize() << "x" << r.matrix->colSize() << " for real state " << mstate->getName() << " using offset " << r.offset << std::endl;
            }
            else
                std::cout << "DefaultMultiMatrixAccessor: NULL matrix found for state " << mstate->getName() << std::endl;
        }
    }
    else
    {
        //case where mechanical state is a mapped state

        std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix*>::iterator itmapped = mappedMatrices.find(mstate);
        if (itmapped != mappedMatrices.end()) // this state is mapped
        {
            if (itmapped->second == NULL) // we need to create the matrix
            {
                itmapped->second = createMatrix(mstate);
            }
            if (itmapped->second != NULL)
            {
                r.matrix = itmapped->second;
                r.offset = 0;
            }

            if( MULTIMATRIX_VERBOSE)
            {
                if (itmapped->second != NULL)
                    std::cout << "DefaultMultiMatrixAccessor: giving matrix " << r.matrix->rowSize() << "x" << r.matrix->colSize() << " for mapped state " << mstate->getName() << std::endl;
                else
                    std::cout << "DefaultMultiMatrixAccessor: giving matrix NULL for mapped state " << mstate->getName() << std::endl;
            }
        }
    }

    diagonalStiffnessBloc[mstate] = r;

    if( MULTIMATRIX_VERBOSE)
    {
        if(r.matrix == NULL)
            std::cout << "DefaultMultiMatrixAccessor: NO matrix found for state" << mstate->getName() << std::endl;
    }

    return r;
}

DefaultMultiMatrixAccessor::InteractionMatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{
    InteractionMatrixRef r2;
    if (mstate1 == mstate2)// case where state1 == state2, interaction matrix is on the diagonal stiffness bloc
    {
        MatrixRef r = getMatrix(mstate1);
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
    else
    {
        std::pair<const sofa::core::behavior::BaseMechanicalState*,const sofa::core::behavior::BaseMechanicalState*> pairMS = std::make_pair(mstate1,mstate2);

        std::map< std::pair<const sofa::core::behavior::BaseMechanicalState*,const sofa::core::behavior::BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.find(pairMS);
        if (it != interactionStiffnessBloc.end())
        {
            if(it->second.matrix != NULL)
            {
                r2 = it->second;
            }
            else
            {
                it->second.matrix = createInteractionMatrix(mstate1,mstate2);
                it->second.offRow = 0;
                it->second.offCol = 0;
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
        else
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
                        if( MULTIMATRIX_VERBOSE)
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
                //r2.matrix = m;
                r2.offRow = 0;
                r2.offCol = 0;

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

            //interactionStiffnessBloc.insert(std::make_pair(pairMS,r2) );
            //interactionMappedTree.insert(pairMS);
            interactionStiffnessBloc[pairMS]=r2;

        }

    }


    if( MULTIMATRIX_VERBOSE)
    {
        if(r2.matrix == NULL)
            std::cout << "DefaultMultiMatrixAccessor: NO matrix found for interaction " << mstate1->getName()<<" --- "<<mstate2->getName() << std::endl;
    }

    return r2;
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

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "MappedMultiMatrixAccessor: ++ creating matrix["<< m->rowSize() <<"x"<< m->colSize() <<"]   for state " << mstate->getName() << "[" << mstate->getMatrixSize()<<"]"<< std::endl;
    }

    return m;
}

defaulttype::BaseMatrix* MappedMultiMatrixAccessor::createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{
    /// @TODO support for mapped matrices
    component::linearsolver::FullMatrix<SReal>* m = new component::linearsolver::FullMatrix<SReal>;
    m->resize( mstate1->getMatrixSize(),mstate2->getMatrixSize() );

    if( MULTIMATRIX_VERBOSE)
    {
        std::cout << "MappedMultiMatrixAccessor: ++ creating interraction matrix["<< m->rowSize() <<"x"<< m->colSize()
                << "] for interaction " << mstate1->getName() << "[" << mstate1->getMatrixSize()
                << "] --- "             << mstate2->getName() << "[" << mstate2->getMatrixSize()<<"]" <<std::endl;
    }

    interactionMappedTree.push_back( std::make_pair(mstate1,mstate2) );
    return m;
}

void MappedMultiMatrixAccessor::computeGlobalMatrix()
{

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::const_reverse_iterator _rit;
    const std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::const_reverse_iterator _itBegin = mappingsContributionTree.rbegin();
    const std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* >::const_reverse_iterator   _itEnd = mappingsContributionTree.rend();
    for(_rit = _itBegin; _rit != _itEnd; ++_rit)
    {
        const sofa::core::behavior::BaseMechanicalState* mstate1 = _rit->second->getMechFrom()[0];
        const sofa::core::behavior::BaseMechanicalState* mstate2 = _rit->second->getMechTo()[0];

        std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* >::const_iterator itmapped = mappedMatrices.find(mstate2);

        // compute contribution only if the mapped matrix K2 is filled
        if(itmapped != mappedMatrices.end() && itmapped->second != NULL )
        {

            MatrixRef K1 = diagonalStiffnessBloc[mstate1];
            MatrixRef K2 = diagonalStiffnessBloc[mstate2];

            const defaulttype::BaseMatrix* matrixJ = _rit->second->getJ();

            const unsigned int sizeK1 = mstate1->getMatrixSize();
            const unsigned int sizeK2 = mstate2->getMatrixSize();

            const unsigned int offset1 = K1.offset;
            const unsigned int offset2 = K2.offset;

            for(unsigned int i1 =0 ; i1 < sizeK1 ; ++i1)
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
                            }
                        }
                    }

                    K1.matrix->add(offset1 + i1 , offset1 + j1 , Jt_K2_J_i1j1);
                }
            }

            if( MULTIMATRIX_VERBOSE)
            {
                std::cout << "MappedMultiMatrixAccessor: MAPPING "<<_rit->second->getName() <<"  MATRIX CONTRIBUTION : ";

                std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix*>::iterator itmapped = mappedMatrices.find(mstate1);
                if (itmapped != mappedMatrices.end())
                {
                    // this state is mapped
                    std::cout <<     "mapped matrix K1[" << K1.matrix->rowSize() <<"x" << K1.matrix->colSize() <<"]";
                }
                else
                {
                    std::cout <<     "local DOF matrix _K1[" << mstate1->getMatrixSize() <<"x" << mstate1->getMatrixSize()
                            <<     "] in global matrix K["<<globalMatrix->rowSize()<<"x"<<globalMatrix->colSize()<<"] at offset "<<K1.offset;
                }

                std::cout << "    mapped matrix K2[" << K2.matrix->rowSize() <<"x" << K2.matrix->colSize() <<"]"
                        << "     J[" <<   matrixJ->rowSize() <<"x" <<   matrixJ->colSize() <<"]"
                        <<std::endl;
            }

        }
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    int lastIter = interactionMappedTree.size() -1 ;
    for(int iIter = lastIter ; iIter >-1 ; --iIter)
    {
        if( MULTIMATRIX_VERBOSE)
        {
            std::cout <<"MappedMultiMatrixAccessor INTERACTION Registered "
                    << interactionMappedTree[iIter].first->getName() <<" --- "
                    << interactionMappedTree[iIter].second->getName() <<std::endl;
        }
    }



}



} // namespace linearsolver

} // namespace component

} // namespace sofa
