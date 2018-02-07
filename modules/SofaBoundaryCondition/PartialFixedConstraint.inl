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
    , d_indices( initData(&d_indices,"indices","Indices of the fixed points") )
    , d_fixAll( initData(&d_fixAll,false,"fixAll","filter all the DOF to implement a fixed object") )
    , d_drawSize( initData(&d_drawSize,(SReal)0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    , fixedDirections( initData(&fixedDirections,"fixedDirections","for each direction, 1 if fixed, 0 if free") )
    , d_projectVelocity( initData(&d_projectVelocity,false,"activate_projectVelocity","activate project velocity to set velocity") )
{
    // default to indice 0
    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();
    VecBool blockedDirection;
    for( unsigned i=0; i<NumDimensions; i++)
        blockedDirection[i] = true;
    fixedDirections.setValue(blockedDirection);

    pointHandler = new FCPointHandler(this, &d_indices);
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
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::addConstraint(unsigned int index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*d_indices.beginEdit(),index);
    d_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void PartialFixedConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    d_indices.createTopologicalEngine(topology, pointHandler);
    d_indices.registerTopologicalData();

    const SetIndexArray & indices = d_indices.getValue();

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
void PartialFixedConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/, DataDeriv& res)
{
    const VecBool& blockedDirection = fixedDirections.getValue();
    //serr<<"PartialFixedConstraint<DataTypes>::projectResponse, res.size()="<<res.size()<<sendl;
    if (d_fixAll.getValue() == true)
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
        const SetIndexArray & indices = d_indices.getValue();
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
void PartialFixedConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT(mparams, res.wref());
}

// projectVelocity applies the same changes on velocity vector as projectResponse on position vector :
// Each fixed point received a null velocity vector.
// When a new fixed point is added while its velocity vector is already null, projectVelocity is not usefull.
// But when a new fixed point is added while its velocity vector is not null, it's necessary to fix it to null. If not, the fixed point is going to drift.
template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    if(!d_projectVelocity.getValue()) return;
    helper::WriteAccessor<DataVecDeriv> res ( mparams, vData );

    //serr<<"PartialFixedConstraint<DataTypes>::projectVelocity, res.size()="<<res.size()<<sendl;
    if( d_fixAll.getValue()==true )
    {
        // fix everyting
        for( unsigned i=0; i<res.size(); i++ )
        {
            res[i] = Deriv();
        }
    }
    else
    {
        const SetIndexArray & indices = d_indices.getValue();
        unsigned i=0;
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end() && i<res.size(); ++it, ++i)
        {
            res[*it] = Deriv();
        }
    }
}

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& /*xData*/)
{

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
    const SetIndexArray & indices = d_indices.getValue();

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
    const SetIndexArray & indices = d_indices.getValue();
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
void PartialFixedConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate.get(mparams));
    if(r)
    {
        //sout << "applyConstraint in Matrix with offset = " << offset << sendl;
        //cerr<<"FixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset) is called "<<endl;
        const unsigned int N = Deriv::size();
        const VecBool& blockedDirection = fixedDirections.getValue();
        const SetIndexArray & indices = d_indices.getValue();

        //TODO take f_fixAll into account


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

template <class DataTypes>
void PartialFixedConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
{
    static const unsigned blockSize = DataTypes::deriv_total_size;

    const VecBool& blockedDirection = fixedDirections.getValue();

    if( d_fixAll.getValue()==true )
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
        const SetIndexArray & indices = d_indices.getValue();
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


    const SetIndexArray & indices = d_indices.getValue();

    if (d_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector<sofa::defaulttype::Vector3> points;
        sofa::defaulttype::Vector3 point;
        //serr<<"PartialFixedConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
        if (d_fixAll.getValue() == true)
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
        if (d_fixAll.getValue() == true)
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
        vparams->drawTool()->drawSpheres(points, (float) d_drawSize.getValue(), sofa::defaulttype::Vec<4, float> (1.0f, 0.35f, 0.35f, 1.0f));
    }
}





} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_INL


