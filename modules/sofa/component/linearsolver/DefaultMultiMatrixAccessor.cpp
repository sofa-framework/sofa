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
    for (std::set< defaulttype::BaseMatrix* >::const_iterator it = tempMatrices.begin(), itend = tempMatrices.end(); it != itend; ++it)
        delete *it;
}

void DefaultMultiMatrixAccessor::clear()
{
    globalDim = 0;
    //globalOffsets.clear();
    for (std::map< const sofa::core::behavior::BaseMechanicalState*, int >::iterator it = globalOffsets.begin(), itend = globalOffsets.end(); it != itend; ++it)
        it->second = -1;
}

void DefaultMultiMatrixAccessor::setGlobalMatrix(defaulttype::BaseMatrix* matrix)
{
    this->globalMatrix = matrix;
}

void DefaultMultiMatrixAccessor::addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
    unsigned int dim = mstate->getMatrixSize();
    globalOffsets[mstate] = globalDim;
    globalDim += dim;
#ifdef MULTIMATRIX_VERBOSE
    std::cout << "DefaultMultiMatrixAccessor: DOFs from " << mstate->getName() << " at offset " << globalOffsets[mstate] << " size " << dim << std::endl;
#endif
}

void DefaultMultiMatrixAccessor::addMechanicalMapping(const sofa::core::BaseMapping* /*mapping*/)
{
    /// @TODO support for mapped matrices
}

#ifdef MULTIMATRIX_VERBOSE
void DefaultMultiMatrixAccessor::addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
    /// @TODO support for mapped matrices
    std::cout << "DefaultMultiMatrixAccessor: Mapped state " << mstate->getName() << " ignored. It will not contribute to the final mechanical matrix." << std::endl;
#else
void DefaultMultiMatrixAccessor::addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* )
{
#endif
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
    std::map< const sofa::core::behavior::BaseMechanicalState*, MatrixRef >::const_iterator it = localMatrixMap.find(mstate);
    if (it != localMatrixMap.end())
    {
        MatrixRef r = it->second;
#ifdef MULTIMATRIX_VERBOSE
        if (r)
            std::cout << "DefaultMultiMatrixAccessor: valid " << r.matrix->rowSize() << "x" << r.matrix->colSize() << " matrix found for " << mstate->getName() << " using offset " << r.offset << std::endl;
        else
            std::cout << "DefaultMultiMatrixAccessor: NULL matrix found for " << mstate->getName() << std::endl;
#endif
        return r;
    }
#ifdef MULTIMATRIX_VERBOSE
    else
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
        return r2;
    }

    std::map< std::pair<const sofa::core::behavior::BaseMechanicalState*,const sofa::core::behavior::BaseMechanicalState*>, InteractionMatrixRef >::const_iterator it = interactionMatrixMap.find(std::make_pair(mstate1,mstate2));
    if (it != interactionMatrixMap.end())
        return it->second;
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
            return r;
        }
    }
    return InteractionMatrixRef();
}

void DefaultMultiMatrixAccessor::computeGlobalMatrix()
{
    /// @TODO support for mapped matrices
    //if (globalMatrix)
    //    std::cout << "DefaultMultiMatrixAccessor: final matrix: " << *globalMatrix << std::endl;
}

defaulttype::BaseMatrix* DefaultMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* /*mstate*/)
{
    /// @TODO support for mapped matrices
    return NULL;
}

defaulttype::BaseMatrix* DefaultMultiMatrixAccessor::createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* /*mstate1*/, const sofa::core::behavior::BaseMechanicalState* /*mstate2*/)
{
    /// @TODO support for mapped matrices
    return NULL;
}

} // namespace linearsolver

} // namespace component

} // namespace sofa
