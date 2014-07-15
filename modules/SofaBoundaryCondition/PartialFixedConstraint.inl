/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <SofaBoundaryCondition/PartialFixedConstraint.h>
#include <sofa/simulation/common/Simulation.h>
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

// Define TestNewPointFunction
template< class DataTypes>
bool PartialFixedConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    return fc != 0;
}

// Define RemovalFunction
template< class DataTypes>
void PartialFixedConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, value_type &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
}


template <class DataTypes>
PartialFixedConstraint<DataTypes>::PartialFixedConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , f_indices( initData(&f_indices,"indices","Indices of the fixed points") )
    , f_fixAll( initData(&f_fixAll,false,"fixAll","filter all the DOF to implement a fixed object") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    , fixedDirections( initData(&fixedDirections,"fixedDirections","for each direction, 1 if fixed, 0 if free") )
{
    // default to indice 0
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
    VecBool blockedDirection;
    for( unsigned i=0; i<NumDimensions; i++)
        blockedDirection[i] = true;
    fixedDirections.setValue(blockedDirection);

    pointHandler = new FCPointHandler(this, &f_indices);
}


template <class DataTypes>
PartialFixedConstraint<DataTypes>::~PartialFixedConstraint()
{
    if (pointHandler)
        delete pointHandler;
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::clearConstraints()
{
    f_indices.beginEdit()->clear();
    f_indices.endEdit();
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::addConstraint(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void PartialFixedConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    f_indices.createTopologicalEngine(topology, pointHandler);
    f_indices.registerTopologicalData();

    const SetIndexArray & indices = f_indices.getValue();

    unsigned int maxIndex=this->mstate->getSize();
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int index=indices[i];
        if (index >= maxIndex)
        {
            serr << "Index " << index << " not valid!" << sendl;
            removeConstraint(index);
        }
    }
}

template <class DataTypes>
template <class DataDeriv>
void PartialFixedConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataDeriv& res)
{
    const VecBool& blockedDirection = fixedDirections.getValue();
    //serr<<"PartialFixedConstraint<DataTypes>::projectResponse, res.size()="<<res.size()<<sendl;
    if (f_fixAll.getValue() == true)
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
        const SetIndexArray & indices = f_indices.getValue();
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
//    cerr<<"PartialFixedConstraint<DataTypes>::projectResponse is called  res = "<<endl<<res<<endl;
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT(mparams /* PARAMS FIRST */, res.wref());
}

// projectVelocity applies the same changes on velocity vector as projectResponse on position vector :
// Each fixed point received a null velocity vector.
// When a new fixed point is added while its velocity vector is already null, projectVelocity is not usefull.
// But when a new fixed point is added while its velocity vector is not null, it's necessary to fix it to null. If not, the fixed point is going to drift.
template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& /*vData*/)
{
#if 0 /// @TODO ADD A FLAG FOR THIS
    helper::WriteAccessor<DataVecDeriv> res = vData;
    //serr<<"PartialFixedConstraint<DataTypes>::projectVelocity, res.size()="<<res.size()<<sendl;
    if( f_fixAll.getValue()==true )
    {
        // fix everyting
        for( unsigned i=0; i<res.size(); i++ )
        {
            res[i] = Deriv();
        }
    }
    else
    {
        const SetIndexArray & indices = f_indices.getValue();
        unsigned i=0;
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end() && i<res.size(); ++it, ++i)
        {
            res[*it] = Deriv();
        }
    }
#endif
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& /*xData*/)
{

}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataMatrixDeriv& cData)
{
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        projectResponseT<MatrixDerivRowType>(mparams /* PARAMS FIRST */, rowIt.row());
        ++rowIt;
    }
    //cerr<<"PartialFixedConstraint<DataTypes>::projectJacobianMatrix : helper::WriteAccessor<DataMatrixDeriv> c =  "<<endl<< c<<endl;
}

// Matrix Integration interface
template <class DataTypes>
void PartialFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset)
{
    //sout << "applyConstraint in Matrix with offset = " << offset << sendl;
    //cerr<<" PartialFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset) is called "<<endl;

    //TODO take f_fixAll into account

    const unsigned int N = Deriv::size();
    const SetIndexArray & indices = f_indices.getValue();

    const VecBool& blockedDirection = fixedDirections.getValue();
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

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset)
{
    //sout << "applyConstraint in Vector with offset = " << offset << sendl;
    //cerr<<"PartialFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset) is called "<<endl;

    //TODO take f_fixAll into account


    const unsigned int N = Deriv::size();

    const VecBool& blockedDirection = fixedDirections.getValue();
    const SetIndexArray & indices = f_indices.getValue();
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


template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
{
    unsigned blockSize = DataTypes::deriv_total_size;

    const VecBool& blockedDirection = fixedDirections.getValue();

    if( f_fixAll.getValue()==true )
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
        const SetIndexArray & indices = f_indices.getValue();
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
void PartialFixedConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    if (!this->isActive())
        return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    //serr<<"PartialFixedConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<sendl;


    const SetIndexArray & indices = f_indices.getValue();

    if (_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector<sofa::defaulttype::Vector3> points;
        sofa::defaulttype::Vector3 point;
        //serr<<"PartialFixedConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
        if (f_fixAll.getValue() == true)
        {
            for (unsigned i = 0; i < x.size(); i++)
            {
                point = DataTypes::getCPos(x[i]);
                points.push_back(point);
            }
        }
        else
        {
            unsigned i=0;
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end() && i<x.size(); ++it, ++i)
            {
                point = DataTypes::getCPos(x[*it]);
                points.push_back(point);
            }
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4, float> (1, 0.5, 0.5, 1));
    }
    else // new drawing by spheres
    {
        std::vector<sofa::defaulttype::Vector3> points;
        sofa::defaulttype::Vector3 point;
        if (f_fixAll.getValue() == true)
        {
            for (unsigned i = 0; i < x.size(); i++)
            {
                point = DataTypes::getCPos(x[i]);
                points.push_back(point);
            }
        }
        else
        {
            unsigned i=0;
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end() && i<x.size(); ++it, ++i)
            {
                point = DataTypes::getCPos(x[*it]);
                points.push_back(point);
            }
        }
        vparams->drawTool()->drawSpheres(points, (float) _drawSize.getValue(), sofa::defaulttype::Vec<4, float> (1.0f, 0.35f, 0.35f, 1.0f));
    }
}





} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_INL


