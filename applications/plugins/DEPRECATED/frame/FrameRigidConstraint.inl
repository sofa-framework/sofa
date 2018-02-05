/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FrameRigidConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FrameRigidConstraint_INL

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include "FrameRigidConstraint.h"
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



template <class DataTypes>
FrameRigidConstraint<DataTypes>::FrameRigidConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , f_index( initData(&f_index,"indices","Indices of the constrained frames") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
{
    // default to index 0
    WriteAccessor<Data<vector<unsigned> > > index( f_index);
    index.push_back(0);
}


template <class DataTypes>
FrameRigidConstraint<DataTypes>::~FrameRigidConstraint()
{
}


template <class DataTypes>
void FrameRigidConstraint<DataTypes>::init()
{
    Inherit1::init();
}


template <class DataTypes>
void FrameRigidConstraint<DataTypes>::projectResponse(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    const vector<unsigned> & indices = f_index.getValue();
    //                cerr<<"FrameRigidConstraint<DataTypes>::projectResponse, indices = "<< indices << endl;
    //                cerr<<"FrameRigidConstraint<DataTypes>::projectResponse, motion changes allowed = "<< allowed << endl;
    for(unsigned i=0; i<indices.size(); i++)
    {
//                    cerr<<"FrameRigidConstraint<DataTypes>::projectResponse, res to project = "<<    res[indices[i]] << endl;
//                    res[indices[i]].setRigid();
//                    cerr<<"FrameRigidConstraint<DataTypes>::projectResponse, res projected = "<<    res[indices[i]] << endl;
    }

}


// projectVelocity applies the same changes on velocity vector as projectResponse on position vector :
// Each fixed point received a null velocity vector.
// When a new fixed point is added while its velocity vector is already null, projectVelocity is not usefull.
// But when a new fixed point is added while its velocity vector is not null, it's necessary to fix it to null. If not, the fixed point is going to drift.
template <class DataTypes>
void FrameRigidConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> res = vData;
    const vector<unsigned> & indices = f_index.getValue();
    for(unsigned i=0; i<indices.size(); i++)
    {
//                    cerr<<"FrameRigidConstraint<DataTypes>::projectVelocity, velocity to project = "<<    res[indices[i]] << endl;
        res[indices[i]].setRigid();
//                    cerr<<"FrameRigidConsraint<DataTypes>::projectVelocity, velocity projected = "<<    res[indices[i]] << endl;
    }
}

template <class DataTypes>
void FrameRigidConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& xData)
{

    helper::WriteAccessor<DataVecCoord> res = xData;
    const vector<unsigned> & indices = f_index.getValue();
    for(unsigned i=0; i<indices.size(); i++)
    {
//                    cerr<<"FrameRigidConstraint<DataTypes>::projectPosition, position to project = "<<    res[indices[i]] << endl;
        res[indices[i]].setRigid();
//                    cerr<<"FrameRigidConstraint<DataTypes>::projectPosition, position projected = "<<    res[indices[i]] << endl;
    }
}



template <class DataTypes>
void FrameRigidConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->isActive()) return;
    const VecCoord& x = *this->mstate->getX();
    //serr<<"FrameRigidConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<sendl;




    const vector<unsigned> & indices = f_index.getValue();

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;
        Vector3 point;
        //serr<<"FrameRigidConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
        for (vector<unsigned>::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            point = DataTypes::getCPos(x[*it]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.0,0.5,1));
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
        vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.0f,0.35f,1.0f));
    }
}



} // namespace constraint

} // namespace component

} // namespace sofa

#endif


