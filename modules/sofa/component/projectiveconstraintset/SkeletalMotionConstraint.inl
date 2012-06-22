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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_SKELETALMOTIONCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_SKELETALMOTIONCONSTRAINT_INL

#include <sofa/component/projectiveconstraintset/SkeletalMotionConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/topology/TopologySubsetData.inl>

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

template <class DataTypes>
SkeletalMotionConstraint<DataTypes>::SkeletalMotionConstraint() : ProjectiveConstraintSet<DataTypes>()
{

}

template <class DataTypes>
SkeletalMotionConstraint<DataTypes>::~SkeletalMotionConstraint()
{

}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::init()
{
    ProjectiveConstraintSet<DataTypes>::init();
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::reset()
{
    ProjectiveConstraintSet<DataTypes>::reset();
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::findKeyTimes()
{
    Real cT = (Real) this->getContext()->getTime();
    finished = false;

    for(unsigned int i = 0; i < skeletonJoints.size(); ++i)
    {
        if(skeletonJoints[i].mChannels.empty() || skeletonJoints[i].mChannels.size() != skeletonJoints[i].mTimes.size())
            continue;

        for(unsigned int j = 0; j < skeletonJoints[i].mTimes.size(); ++j)
        {
            Real keyTime = skeletonJoints[i].mTimes[j];
            if(keyTime <= cT)
            {
                prevT = keyTime;

                const RigidCoord<3, Real>& motion = skeletonJoints[i].mChannels[j];
                skeletonJoints[i].mPreviousMotion = motion;
            }
            else
            {
                nextT = keyTime;

                const RigidCoord<3, Real>& motion = skeletonJoints[i].mChannels[j];
                skeletonJoints[i].mNextMotion = motion;

                finished = true;
                break;
            }
        }
    }
}

template <class DataTypes>
template <class DataDeriv>
void SkeletalMotionConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataDeriv& dx)
{
    Real cT = (Real) this->getContext()->getTime();

    if(0.0 != cT)
    {
        findKeyTimes();

        if(finished && nextT != prevT)
        {
            //set the motion to the Dofs
            for(int i = 0; i < dx.size(); ++i)
                dx[i] = Deriv();
        }
    }
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(mparams /* PARAMS FIRST */, res.wref());
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real cT = (Real) this->getContext()->getTime();

    if(0.0 != cT)
    {
        findKeyTimes();

        if(finished && nextT != prevT)
        {
            //set the motion to the Dofs
            for(unsigned int i = 0; i < dx.size(); ++i)
            {
                SkeletonJoint<DataTypes>& skeletonJoint = skeletonJoints[skeletonBones[i].mSkeletonJointIndex];

                if(skeletonJoint.mChannels.empty())
                {
                    dx[i] = Deriv();
                }
                else
                {
                    dx[i].getVCenter() = (skeletonJoint.mNextMotion.getCenter() - skeletonJoint.mPreviousMotion.getCenter()) * (1.0 / (nextT - prevT));

                    Quat previousOrientation = skeletonJoint.mPreviousMotion.getOrientation();
                    Quat nextOrientation = skeletonJoint.mNextMotion.getOrientation();
                    Vec<3, Real> diff = nextOrientation.angularDisplacement(previousOrientation, nextOrientation);
                    dx[i].getVOrientation() = diff * (1.0 / (nextT - prevT));
                }
            }
        }
        else
        {
            for(unsigned int i = 0; i < dx.size(); ++i)
                dx[i] = Deriv();
        }
    }
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    Real cT = (Real) this->getContext()->getTime();

    if(0.0 != cT)
    {
        findKeyTimes();

        // if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
        interpolatePosition<Coord>(cT, x.wref());
    }
}

template <class DataTypes>
template <class MyCoord>
void SkeletalMotionConstraint<DataTypes>::interpolatePosition(Real cT, typename boost::enable_if<boost::is_same<MyCoord, RigidCoord<3, Real> >, VecCoord>::type& x)
{
    // set the motion to the SkeletonJoint corresponding rigid
    if(finished && nextT != prevT)
    {
        Real dt = (cT - prevT) / (nextT - prevT);

        for(int i = 0; i < skeletonJoints.size(); ++i)
        {
            if(skeletonJoints[i].mChannels.empty())
                continue;

            skeletonJoints[i].mLocalRigid.getCenter() = skeletonJoints[i].mPreviousMotion.getCenter() + (skeletonJoints[i].mNextMotion.getCenter() - skeletonJoints[i].mPreviousMotion.getCenter()) * dt;
            skeletonJoints[i].mLocalRigid.getOrientation().slerp(skeletonJoints[i].mPreviousMotion.getOrientation(), skeletonJoints[i].mNextMotion.getOrientation(), dt, true);
        }
    }
    else
    {
        for(int i = 0; i < skeletonJoints.size(); ++i)
        {
            if(skeletonJoints[i].mChannels.empty())
                continue;

            skeletonJoints[i].mLocalRigid.getCenter() = skeletonJoints[i].mNextMotion.getCenter();
            skeletonJoints[i].mLocalRigid.getOrientation() = skeletonJoints[i].mNextMotion.getOrientation();
        }
    }

    // apply the final transformation from skeletonBones to dofs here
    localToGlobal<Coord>(x);
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataMatrixDeriv& cData)
{
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while(rowIt != rowItEnd)
    {
        projectResponseT<MatrixDerivRowType>(mparams /* PARAMS FIRST */, rowIt.row());
        ++rowIt;
    }
}

template <class DataTypes>
template <class MyCoord>
void SkeletalMotionConstraint<DataTypes>::localToGlobal(typename boost::enable_if<boost::is_same<MyCoord, RigidCoord<3, Real> >, VecCoord>::type& x)
{
    for(int i = 0; i < skeletonJoints.size(); ++i)
    {
        RigidCoord< 3, Real> worldRigid = skeletonJoints[i].mLocalRigid;

        // break if the current SkeletonJoint is the root
        for(int parentIndex = skeletonJoints[i].mParentIndex; -1 != parentIndex; parentIndex = skeletonJoints[parentIndex].mParentIndex)
        {
            RigidCoord< 3, Real> parentLocalRigid = skeletonJoints[parentIndex].mLocalRigid;
            worldRigid = parentLocalRigid.mult(worldRigid);
        }

        skeletonJoints[i].mWorldRigid = worldRigid;
    }

    for(int i = 0; i < skeletonBones.size(); ++i)
        x[i] = skeletonJoints[skeletonBones[i].mSkeletonJointIndex].mWorldRigid;
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::setSkeletalMotion(const std::vector<SkeletonJoint<DataTypes> >& skeletonJoints, const std::vector<SkeletonBone>& skeletonBones)
{
    this->skeletonJoints = skeletonJoints;
    this->skeletonBones = skeletonBones;
}

// Matrix Integration interface
template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix * /*mat*/, unsigned int /*offset*/)
{
    //sout << "applyConstraint in Matrix with offset = " << offset << sendl;
    /*const unsigned int N = Deriv::size();
    const SetIndexArray & indices = m_indices.getValue();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
    	// Reset Fixed Row and Col
    	for (unsigned int c=0;c<N;++c)
    		mat->clearRowCol(offset + N * (*it) + c);
    	// Set Fixed Vertex
    	for (unsigned int c=0;c<N;++c)
    		mat->set(offset + N * (*it) + c, offset + N * (*it) + c, 1.0);
    }*/
}

template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector * /*vect*/, unsigned int /*offset*/)
{
    //sout << "applyConstraint in Vector with offset = " << offset << sendl;
    /*const unsigned int N = Deriv::size();

    const SetIndexArray & indices = m_indices.getValue();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
    	for (unsigned int c=0;c<N;++c)
    		vect->clear(offset + N * (*it) + c);
    }*/
}

// display the paths the constrained dofs will go through
template <class DataTypes>
void SkeletalMotionConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    //const VecCoord& x = *this->mstate->getX();

    sofa::helper::vector<Vector3> points;
    sofa::helper::vector<Vector3> linesX;
    sofa::helper::vector<Vector3> linesY;
    sofa::helper::vector<Vector3> linesZ;
    sofa::helper::vector<Vector3> colorFalloff;

    Vector3 point;
    Vector3 line;

    // draw joints (not bones we draw them differently later)
    {
        for(unsigned int i = 0; i < skeletonJoints.size(); ++i)
        {
            RigidCoord< 3, Real> jointWorldRigid = skeletonJoints[i].mWorldRigid;

            unsigned int j;
            for(j = 0; j < skeletonBones.size(); ++j)
                if((int)i == skeletonBones[j].mSkeletonJointIndex)
                    break;

            if(skeletonBones.size() != j)
                continue;

            point = DataTypes::getCPos(jointWorldRigid);
            points.push_back(point);

            linesX.push_back(point);
            line = point + DataTypes::getCRot(jointWorldRigid).rotate(Vec3f(0.1, 0.0, 0.0));
            linesX.push_back(line);

            linesY.push_back(point);
            line = point + DataTypes::getCRot(jointWorldRigid).rotate(Vec3f(0.0, 0.1, 0.0));
            linesY.push_back(line);

            linesZ.push_back(point);
            line = point + DataTypes::getCRot(jointWorldRigid).rotate(Vec3f(0.0, 0.0, 0.1));
            linesZ.push_back(line);
        }
        vparams->drawTool()->drawPoints(points, 10, Vec<4, float> (0.75, 0.75, 1.0, 1));
        vparams->drawTool()->drawLines(linesX, 2, Vec<4, float> (0.75, 0.0, 0.0, 1));
        vparams->drawTool()->drawLines(linesY, 2, Vec<4, float> (0.0, 0.75, 0.0, 1));
        vparams->drawTool()->drawLines(linesZ, 2, Vec<4, float> (0.0, 0.0, 0.75, 1));
    }

    points.clear();
    linesX.clear();
    linesY.clear();
    linesZ.clear();

    // draw bones now
    {
        for(unsigned int i = 0; i < skeletonBones.size(); ++i)
        {
            RigidCoord< 3, Real> boneWorldRigid = skeletonJoints[skeletonBones[i].mSkeletonJointIndex].mWorldRigid;

            point = DataTypes::getCPos(boneWorldRigid);
            points.push_back(point);

            linesX.push_back(point);
            line = point + DataTypes::getCRot(boneWorldRigid).rotate(Vec3f(0.1, 0.0, 0.0));
            linesX.push_back(line);

            linesY.push_back(point);
            line = point + DataTypes::getCRot(boneWorldRigid).rotate(Vec3f(0.0, 0.1, 0.0));
            linesY.push_back(line);

            linesZ.push_back(point);
            line = point + DataTypes::getCRot(boneWorldRigid).rotate(Vec3f(0.0, 0.0, 0.1));
            linesZ.push_back(line);
        }
        vparams->drawTool()->drawPoints(points, 10, Vec<4, float> (1.0, 0.5, 0.5, 1));
        vparams->drawTool()->drawLines(linesX, 2, Vec<4, float> (1.0, 0.0, 0.0, 1));
        vparams->drawTool()->drawLines(linesY, 2, Vec<4, float> (0.0, 1.0, 0.0, 1));
        vparams->drawTool()->drawLines(linesZ, 2, Vec<4, float> (0.0, 0.0, 1.0, 1));
    }
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
