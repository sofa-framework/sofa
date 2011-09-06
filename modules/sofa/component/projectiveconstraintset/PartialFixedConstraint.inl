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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/component/projectiveconstraintset/PartialFixedConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace core::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::behavior;


// Define TestNewPointFunction
template< class DataTypes>
bool PartialFixedConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    PartialFixedConstraint<DataTypes> *fc= (PartialFixedConstraint<DataTypes> *)param;
    return fc != 0;
}

// Define RemovalFunction
template< class DataTypes>
void PartialFixedConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    PartialFixedConstraint<DataTypes> *fc= (PartialFixedConstraint<DataTypes> *)param;
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
}


// Handle topological changes
template <class DataTypes> void PartialFixedConstraint<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->endChange();

    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
}

template <class DataTypes>
PartialFixedConstraint<DataTypes>::~PartialFixedConstraint()
{
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
    topology::PointSubset my_subset = f_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );


    unsigned int maxIndex=this->mstate->getSize();
    for (topology::PointSubset::iterator it = my_subset.begin();  it != my_subset.end(); )
    {
        topology::PointSubset::iterator currentIterator=it;
        const unsigned int index=*it;
        it++;
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
    const SetIndexArray & indices = f_indices.getValue().getArray();
    VecBool blockedDirection = fixedDirections.getValue();
    //serr<<"PartialFixedConstraint<DataTypes>::projectResponse, res.size()="<<res.size()<<sendl;
    if (f_fixAll.getValue() == true)
    {
        // fix everyting
        for (int i = 0; i < topology->getNbPoints(); i++)
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
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
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
    const SetIndexArray & indices = f_indices.getValue().getArray();
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
        for (SetIndexArray::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
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

    const unsigned int N = Deriv::size();
    const SetIndexArray & indices = f_indices.getValue().getArray();

    VecBool blockedDirection = fixedDirections.getValue();
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
    const unsigned int N = Deriv::size();

    VecBool blockedDirection = fixedDirections.getValue();
    const SetIndexArray & indices = f_indices.getValue().getArray();
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
void PartialFixedConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    if (!this->isActive())
        return;
    const VecCoord& x = *this->mstate->getX();
    //serr<<"PartialFixedConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<sendl;


    const SetIndexArray & indices = f_indices.getValue().getArray();

    if (_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector<Vector3> points;
        Vector3 point;
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
            for (SetIndexArray::const_iterator it = indices.begin(); it
                    != indices.end(); ++it)
            {
                point = DataTypes::getCPos(x[*it]);
                points.push_back(point);
            }
        }
        vparams->drawTool()->drawPoints(points, 10, Vec<4, float> (1, 0.5, 0.5, 1));
    }
    else // new drawing by spheres
    {
        std::vector<Vector3> points;
        Vector3 point;
        glColor4f(1.0f, 0.35f, 0.35f, 1.0f);
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
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                point = DataTypes::getCPos(x[*it]);
                points.push_back(point);
            }
        }
        vparams->drawTool()->drawSpheres(points, (float) _drawSize.getValue(), Vec<4, float> (1.0f, 0.35f, 0.35f, 1.0f));
    }
}

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void PartialFixedConstraint<Rigid3dTypes >::draw(const core::visual::VisualParams* vparams);
template <>
void PartialFixedConstraint<Rigid2dTypes >::draw(const core::visual::VisualParams* vparams);
#endif
#ifndef SOFA_DOUBLE
template <>
void PartialFixedConstraint<Rigid3fTypes >::draw(const core::visual::VisualParams* vparams);
template <>
void PartialFixedConstraint<Rigid2fTypes >::draw(const core::visual::VisualParams* vparams);
#endif



} // namespace constraint

} // namespace component

} // namespace sofa

#endif


