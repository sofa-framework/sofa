/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBoundaryCondition/PartialFixedConstraint.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <SofaBaseTopology/TopologySubsetData.inl>

#include <sofa/core/visual/VisualParams.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

template <class DataTypes>
PartialFixedConstraint<DataTypes>::PartialFixedConstraint()
    : d_fixedDirections( initData(&d_fixedDirections,"fixedDirections","for each direction, 1 if fixed, 0 if free") )
{
    // default to indice 0
    this->d_indices.beginEdit()->push_back(0);
    this->d_indices.endEdit();

    this->pointHandler = new typename Inherited::FCPointHandler(this, &this->d_indices);

    VecBool blockedDirection;
    for( unsigned i=0; i<NumDimensions; i++)
        blockedDirection[i] = true;
    d_fixedDirections.setValue(blockedDirection);
}


template <class DataTypes>
PartialFixedConstraint<DataTypes>::~PartialFixedConstraint()
{
    //Parent class FixedConstraint already destruct : pointHandler and data
}


template <class DataTypes>
void PartialFixedConstraint<DataTypes>::init()
{
    this->Inherited::init();
}


template <class DataTypes>
void PartialFixedConstraint<DataTypes>::reinit()
{
    this->Inherited::reinit();
}


template <class DataTypes>
template <class DataDeriv>
void PartialFixedConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/, DataDeriv& res)
{
    const VecBool& blockedDirection = d_fixedDirections.getValue();

    if (this->d_fixAll.getValue() == true)
    {
        // fix everyting
        for( unsigned i=0; i<res.size(); i++ )
        {
            for (unsigned j = 0; j < NumDimensions; j++)
            {
                if (blockedDirection[j])
                {
                    res[i][j] = (Real) 0.0;
                }
            }
        }
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        unsigned i=0;
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end() && i<res.size(); ++it, ++i)
        {
            for (unsigned j = 0; j < NumDimensions; j++)
            {
                if (blockedDirection[j])
                {
                    res[*it][j] = (Real) 0.0;
                }
            }
        }
    }
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT(mparams, res.wref());
}


template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        projectResponseT<MatrixDerivRowType>(mparams, rowIt.row());
        ++rowIt;
    }
}

// Matrix Integration interface
template <class DataTypes>
void PartialFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset)
{

    const unsigned int N = Deriv::size();
    const VecBool& blockedDirection = d_fixedDirections.getValue();

    if( this->d_fixAll.getValue() )
    {
        unsigned size = this->mstate->getSize();
        for(unsigned int i=0; i<size; i++)
        {
            // Reset Fixed Row and Col
            for (unsigned int c=0; c<N; ++c)
            {
                if( blockedDirection[c] ) mat->clearRowCol(offset + N * i + c);
            }
            // Set Fixed Vertex
            for (unsigned int c=0; c<N; ++c)
            {
                if( blockedDirection[c] ) mat->set(offset + N * i + c, offset + N * i + c, 1.0);
            }
        }
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            // Reset Fixed Row and Col
            for (unsigned int c=0; c<N; ++c)
            {
                if( blockedDirection[c] ) mat->clearRowCol(offset + N * (*it) + c);
            }
            // Set Fixed Vertex
            for (unsigned int c=0; c<N; ++c)
            {
                if( blockedDirection[c] ) mat->set(offset + N * (*it) + c, offset + N * (*it) + c, 1.0);
            }
        }
    }
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset)
{


    const unsigned int N = Deriv::size();

    const VecBool& blockedDirection = d_fixedDirections.getValue();

    if( this->d_fixAll.getValue() )
    {
        for( int i=0; i<vect->size(); i++ )
        {
            for (unsigned int c = 0; c < N; ++c)
            {
                if (blockedDirection[c])
                {
                    vect->clear(offset + N * i + c);
                }
            }
        }
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for (unsigned int c = 0; c < N; ++c)
            {
                if (blockedDirection[c])
                {
                    vect->clear(offset + N * (*it) + c);
                }
            }
        }
    }
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate.get(mparams));
    if(r)
    {
        const unsigned int N = Deriv::size();
        const VecBool& blockedDirection = d_fixedDirections.getValue();
        const SetIndexArray & indices = this->d_indices.getValue();

        if( this->d_fixAll.getValue() )
        {
            unsigned size = this->mstate->getSize();
            for(unsigned int i=0; i<size; i++)
            {
                // Reset Fixed Row and Col
                for (unsigned int c=0; c<N; ++c)
                {
                    if (blockedDirection[c])
                    {
                        r.matrix->clearRowCol(r.offset + N * i + c);
                    }
                }
                // Set Fixed Vertex
                for (unsigned int c=0; c<N; ++c)
                {
                    if (blockedDirection[c])
                    {
                        r.matrix->set(r.offset + N * i + c, r.offset + N * i + c, 1.0);
                    }
                }
            }
        }
        else
        {
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                // Reset Fixed Row and Col
                for (unsigned int c=0; c<N; ++c)
                {
                    if (blockedDirection[c])
                    {
                        r.matrix->clearRowCol(r.offset + N * (*it) + c);
                    }
                }
                // Set Fixed Vertex
                for (unsigned int c=0; c<N; ++c)
                {
                    if (blockedDirection[c])
                    {
                        r.matrix->set(r.offset + N * (*it) + c, r.offset + N * (*it) + c, 1.0);
                    }
                }
            }
        }
    }
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
{
    static const unsigned blockSize = DataTypes::deriv_total_size;

    const VecBool& blockedDirection = d_fixedDirections.getValue();

    if( this->d_fixAll.getValue() )
    {
        unsigned size = this->mstate->getSize();
        for( unsigned i=0; i<size; i++ )
        {
            for (unsigned int c = 0; c < blockSize; ++c)
            {
                if (blockedDirection[c])
                {
                    M->clearRowCol( offset + i * blockSize + c );
                }
            }
        }
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for (unsigned int c = 0; c < blockSize; ++c)
            {
                if (blockedDirection[c])
                {
                    M->clearRowCol( offset + (*it) * blockSize + c);
                }
            }
        }
    }
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_INL


