/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FrameFixedConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FrameFixedConstraint_INL

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include "FrameFixedConstraint.h"
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>

#include <sofa/helper/gl/BasicShapes.h>

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
using helper::ReadAccessor;
using helper::WriteAccessor;


//// Define TestNewPointFunction
//template< class DataTypes>
//bool FrameFixedConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
//{
//        FrameFixedConstraint<DataTypes> *fc= (FrameFixedConstraint<DataTypes> *)param;
//	if (fc) {
//		return true;
//	}else{
//		return false;
//	}
//}
//
//// Define RemovalFunction
//template< class DataTypes>
//void FrameFixedConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
//{
//        FrameFixedConstraint<DataTypes> *fc= (FrameFixedConstraint<DataTypes> *)param;
//	if (fc) {
//		fc->removeConstraint((unsigned int) pointIndex);
//	}
//	return;
//}

template <class DataTypes>
FrameFixedConstraint<DataTypes>::FrameFixedConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , f_index( initData(&f_index,"indices","Indices of the constrained frames") )
    , f_allowed( initData(&f_allowed,"allowed","Displacements allowed. One value per DOF, 1 means allowed, 0 means forbidden") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
{
    // default to index 0
    WriteAccessor<Data<vector<unsigned> > > index( f_index);
    WriteAccessor<Data<vector<VecAllowed > > > allowed( f_allowed);
    index.push_back(0);
    allowed.push_back( VecAllowed() );
}


//// Handle topological changes
//template <class DataTypes> void FrameFixedConstraint<DataTypes>::handleTopologyChange()
//{
//	std::list<const TopologyChange *>::const_iterator itBegin=topology->beginChange();
//	std::list<const TopologyChange *>::const_iterator itEnd=topology->endChange();
//
//	f_index.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
//	f_index.endEdit();
//
//}

template <class DataTypes>
FrameFixedConstraint<DataTypes>::~FrameFixedConstraint()
{
}


template <class DataTypes>
void FrameFixedConstraint<DataTypes>::init()
{
    Inherit1::init();


    //  if (!topology)
    //    serr << "Can not find the topology." << sendl;

    // Initialize functions and parameters
    //	topology::PointSubset my_subset = f_indices.getValue();
    //
    //	my_subset.setTestFunction(FCTestNewPointFunction);
    //	my_subset.setRemovalFunction(FCRemovalFunction);
    //
    //	my_subset.setTestParameter( (void *) this );
    //	my_subset.setRemovalParameter( (void *) this );
    //
    //
    //  unsigned int maxIndex=this->mstate->getSize();
    //  for (topology::PointSubset::iterator it = my_subset.begin();  it != my_subset.end(); )
    //  {
    //    topology::PointSubset::iterator currentIterator=it;
    //    const unsigned int index=*it;
    //    it++;
    //    if (index >= maxIndex)
    //    {
    //      serr << "Index " << index << " not valid!" << sendl;
    //      removeConstraint(index);
    //    }
    //  }

}


template <class DataTypes>
void FrameFixedConstraint<DataTypes>::projectResponse(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    const vector<unsigned> & indices = f_index.getValue();
    const vector<VecAllowed> & allowed = f_allowed.getValue();
    //cerr<<"FrameFixedConstraint<DataTypes>::projectResponse, indices = "<< indices << endl;
    //cerr<<"FrameFixedConstraint<DataTypes>::projectResponse, motion changes allowed = "<< allowed << endl;
    for(unsigned i=0; i<indices.size(); i++)
    {
        for(unsigned j=0; j<dimensions; j++ )
        {
            if( !allowed[i][j] )
                res[indices[i]][j] = 0;
        }
    }

}

//template <class DataTypes>
//void FrameFixedConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataMatrixDeriv& cData)
//{
//    helper::WriteAccessor<DataMatrixDeriv> c = cData;
//
//    MatrixDerivRowIterator rowIt = c->begin();
//    MatrixDerivRowIterator rowItEnd = c->end();
//
//    while (rowIt != rowItEnd)
//    {
//        projectResponseT<MatrixDerivRowType>(mparams /* PARAMS FIRST */, rowIt.row());
//        ++rowIt;
//    }
//}

// projectVelocity applies the same changes on velocity vector as projectResponse on position vector :
// Each fixed point received a null velocity vector.
// When a new fixed point is added while its velocity vector is already null, projectVelocity is not usefull.
// But when a new fixed point is added while its velocity vector is not null, it's necessary to fix it to null. If not, the fixed point is going to drift.
template <class DataTypes>
void FrameFixedConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& /*vData*/)
{
#if 0 /// @TODO ADD A FLAG FOR THIS
    const SetIndexArray & indices = f_indices.getValue().getArray();
    //serr<<"FrameFixedConstraint<DataTypes>::projectVelocity, res.size()="<<res.size()<<sendl;
    if( f_fixAll.getValue()==true )    // fix everyting
    {
        for( unsigned i=0; i<res.size(); i++ )
            res[i] = Deriv();
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
void FrameFixedConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& /*xData*/)
{

}

//// Matrix Integration interface
//template <class DataTypes>
//void FrameFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset)
//{
//    //sout << "applyConstraint in Matrix with offset = " << offset << sendl;
//    const unsigned int N = Deriv::size();
//    const SetIndexArray & indices = f_indices.getValue().getArray();
//
//    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
//    {
//        // Reset Fixed Row and Col
//        for (unsigned int c=0;c<N;++c)
//            mat->clearRowCol(offset + N * (*it) + c);
//        // Set Fixed Vertex
//        for (unsigned int c=0;c<N;++c)
//            mat->set(offset + N * (*it) + c, offset + N * (*it) + c, 1.0);
//    }
//}
//
//template <class DataTypes>
//void FrameFixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset)
//{
//    //sout << "applyConstraint in Vector with offset = " << offset << sendl;
//    const unsigned int N = Deriv::size();
//
//    const SetIndexArray & indices = f_indices.getValue().getArray();
//    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
//    {
//        for (unsigned int c=0;c<N;++c)
//            vect->clear(offset + N * (*it) + c);
//    }
//}


template <class DataTypes>
void FrameFixedConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->isActive()) return;
    const VecCoord& x = *this->mstate->getX();
    //serr<<"FrameFixedConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<sendl;




    const vector<unsigned> & indices = f_index.getValue();

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;
        Vector3 point;
        //serr<<"FrameFixedConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
        for (vector<unsigned>::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            point = DataTypes::getCPos(x[*it]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else // new drawing by spheres
    {
        std::vector< Vector3 > points;
        Vector3 point;
        glColor4f (1.0f,0.35f,0.35f,1.0f);
        for (vector<unsigned>::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            point = DataTypes::getCPos(x[*it]);
            points.push_back(point);
        }
        vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
    }
}

//            // Specialization for rigids
//#ifndef SOFA_FLOAT
//            template <>
//                    void FrameFixedConstraint<Rigid3dTypes >::draw();
//            template <>
//                    void FrameFixedConstraint<Rigid2dTypes >::draw();
//#endif
//#ifndef SOFA_DOUBLE
//            template <>
//                    void FrameFixedConstraint<Rigid3fTypes >::draw();
//            template <>
//                    void FrameFixedConstraint<Rigid2fTypes >::draw();
//#endif



} // namespace constraint

} // namespace component

} // namespace sofa

#endif


