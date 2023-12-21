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

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/component/constraint/projective/PartialFixedProjectiveConstraint.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <sofa/core/visual/VisualParams.h>


namespace sofa::component::constraint::projective
{

template <class DataTypes>
PartialFixedProjectiveConstraint<DataTypes>::PartialFixedProjectiveConstraint()
    : d_fixedDirections( initData(&d_fixedDirections,"fixedDirections","for each direction, 1 if fixed, 0 if free") )
    , d_projectVelocity(initData(&d_projectVelocity, false, "projectVelocity", "project velocity to ensure no drift of the fixed point"))
{
    VecBool blockedDirection;
    for( unsigned i=0; i<NumDimensions; i++)
        blockedDirection[i] = true;
    d_fixedDirections.setValue(blockedDirection);
}


template <class DataTypes>
PartialFixedProjectiveConstraint<DataTypes>::~PartialFixedProjectiveConstraint()
{
    //Parent class FixedConstraint already destruct : pointHandler and data
}

template <class DataTypes>
void PartialFixedProjectiveConstraint<DataTypes>::reinit()
{
    this->Inherited::reinit();
}


template <class DataTypes>
template <class DataDeriv>
void PartialFixedProjectiveConstraint<DataTypes>::projectResponseT(DataDeriv& res,
    const std::function<void(DataDeriv&, const unsigned int, const VecBool&)>& clear)
{
    const VecBool& blockedDirection = d_fixedDirections.getValue();

    if (this->d_fixAll.getValue() == true)
    {
        // fix everything
        for( std::size_t i=0; i<res.size(); i++ )
        {
            clear(res, i, blockedDirection);
        }
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            clear(res, *it, blockedDirection);
        }
    }
}

template <class DataTypes>
void PartialFixedProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(res.wref(), 
        [](VecDeriv& dx, const unsigned int index, const VecBool& b)
        { 
            for (std::size_t j = 0; j < b.size(); j++) if (b[j]) dx[index][j] = 0.0; 
        }
    );
}

// projectVelocity applies the same changes on velocity vector as projectResponse on position vector :
// Each fixed point received a null velocity vector.
// When a new fixed point is added while its velocity vector is already null, projectVelocity is not usefull.
// But when a new fixed point is added while its velocity vector is not null, it's necessary to fix it to null or 
// to set the projectVelocity option to True. If not, the fixed point is going to drift.
template <class DataTypes>
void PartialFixedProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    SOFA_UNUSED(mparams);

    if(!d_projectVelocity.getValue()) return;

    const VecBool& blockedDirection = d_fixedDirections.getValue();
    helper::WriteAccessor<DataVecDeriv> res = vData;

    if ( this->d_fixAll.getValue() )
    {
        // fix everyting
        for (Size i = 0; i < res.size(); i++)
        {
            for (unsigned int c = 0; c < NumDimensions; ++c)
            {
                if (blockedDirection[c]) res[i][c] = 0;
            }
        }
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        for(Index ind : indices)
        {
            for (unsigned int c = 0; c < NumDimensions; ++c)
            {
                if (blockedDirection[c])
                    res[ind][c] = 0;
            }
        }
    }
}


template <class DataTypes>
void PartialFixedProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    projectResponseT<MatrixDeriv>(c.wref(),
        [](MatrixDeriv& res, const unsigned int index, const VecBool& btype)
        {
            auto itRow = res.begin();
            auto itRowEnd = res.end();

            while (itRow != itRowEnd)
            {
                for (auto colIt = itRow.begin(); colIt != itRow.end(); colIt++)
                {
                    if (index == (unsigned int)colIt.index())
                    {
                        Deriv b = colIt.val();
                        for (unsigned int j = 0; j < btype.size(); j++)
                            if (btype[j]) b[j] = 0.0;
                        res.writeLine(itRow.index()).setCol(colIt.index(), b);
                    }
                }
                ++itRow;
            }
        });
}

template <class DataTypes>
void PartialFixedProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    const int o = matrix->getGlobalOffset(this->mstate.get());
    if (o >= 0)
    {
        const unsigned int offset = (unsigned int)o;
        const unsigned int N = Deriv::size();

        const VecBool& blockedDirection = d_fixedDirections.getValue();

        if( this->d_fixAll.getValue() )
        {
            for(sofa::Index i=0; i<(sofa::Size) vector->size(); i++ )
            {
                for (unsigned int c = 0; c < N; ++c)
                {
                    if (blockedDirection[c])
                    {
                        vector->clear(offset + N * i + c);
                    }
                }
            }
        }
        else
        {
            const SetIndexArray & indices = this->d_indices.getValue();
            for (const unsigned int index : indices)
            {
                for (unsigned int c = 0; c < N; ++c)
                {
                    if (blockedDirection[c])
                    {
                        vector->clear(offset + N * index + c);
                    }
                }
            }
        }
    }
}

template <class DataTypes>
void PartialFixedProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    if(const core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate.get()))
    {
        const unsigned int N = Deriv::size();
        const VecBool& blockedDirection = d_fixedDirections.getValue();
        const SetIndexArray & indices = this->d_indices.getValue();

        if( this->d_fixAll.getValue() )
        {
            const unsigned size = this->mstate->getSize();
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
void PartialFixedProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset )
{
    static const unsigned blockSize = DataTypes::deriv_total_size;

    const VecBool& blockedDirection = d_fixedDirections.getValue();

    if( this->d_fixAll.getValue() )
    {
        const unsigned size = this->mstate->getSize();
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

template <class DataTypes>
void PartialFixedProjectiveConstraint<DataTypes>::applyConstraint(
    sofa::core::behavior::ZeroDirichletCondition* matrix)
{
    static constexpr unsigned int N = Deriv::size();
    const VecBool& blockedDirection = d_fixedDirections.getValue();

    if( this->d_fixAll.getValue() )
    {
        const sofa::Size size = this->mstate->getSize();

        for(sofa::Index i = 0; i < size; ++i)
        {
            for (unsigned int c=0; c<N; ++c)
            {
                if (blockedDirection[c])
                {
                    matrix->discardRowCol(N * i + c, N * i + c);
                }
            }
        }
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();

        for (const auto index : indices)
        {
            for (unsigned int c = 0; c < N; ++c)
            {
                if (blockedDirection[c])
                {
                    matrix->discardRowCol(N * index + c, N * index + c);
                }
            }
        }
    }
}
} // namespace sofa::component::constraint::projective
