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

#include <sofa/component/constraint/projective/SkeletalMotionProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <iostream>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

namespace sofa::component::constraint::projective
{

template <class DataTypes>
SkeletalMotionProjectiveConstraint<DataTypes>::SkeletalMotionProjectiveConstraint() :
    sofa::core::behavior::ProjectiveConstraintSet<DataTypes>()
    , d_skeletonJoints(initData(&d_skeletonJoints, "joints", "skeleton joints"))
    , d_skeletonBones(initData(&d_skeletonBones, "bones", "skeleton bones"))
	, d_animationSpeed(initData(&d_animationSpeed, 1.0f, "animationSpeed", "animation speed"))
    , d_active(initData(&d_active, true, "active", "is the constraint active?"))
    , finished(false)
{
}

template <class DataTypes>
SkeletalMotionProjectiveConstraint<DataTypes>::~SkeletalMotionProjectiveConstraint()
{
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::init()
{
    nextPositions.resize(d_skeletonBones.getValue().size());
    sofa::core::behavior::ProjectiveConstraintSet<DataTypes>::init();
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::reset()
{
    sofa::core::behavior::ProjectiveConstraintSet<DataTypes>::reset();
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::findKeyTimes(Real ct)
{
    //Note: works only if the times are sorted
    
    finished = false;

    for(unsigned int i = 0; i < d_skeletonJoints.getValue().size(); ++i)
    {
        SkeletonJoint<DataTypes>& skeletonJoint = (*d_skeletonJoints.beginEdit())[i];
        

        if(skeletonJoint.mChannels.empty() || skeletonJoint.mChannels.size() != skeletonJoint.mTimes.size())
            continue;

        for(unsigned int j = 0; j < skeletonJoint.mTimes.size(); ++j)
        {
            Real keyTime = (Real) skeletonJoint.mTimes[j];
            if(keyTime <= ct)
            {
                {
                    skeletonJoint.mPreviousMotionTime = keyTime;
                    const defaulttype::RigidCoord<3, Real>& motion = skeletonJoint.mChannels[j];
                    skeletonJoint.mPreviousMotion = motion;
                }
                if(prevT < keyTime)
                    prevT = keyTime;
            }
            else
            {
                {
                    skeletonJoint.mNextMotionTime = keyTime;
                    const defaulttype::RigidCoord<3, Real>& motion = skeletonJoint.mChannels[j];
                    skeletonJoint.mNextMotion = motion;
                }
                if(nextT > keyTime)
                    nextT = keyTime;

                finished = true;
                break;
            }
        }
		d_skeletonJoints.endEdit();
    }
}

template <class TDataTypes> template <class DataDeriv>
void SkeletalMotionProjectiveConstraint<TDataTypes>::projectResponseT(DataDeriv& res,
    const std::function<void(DataDeriv&, const unsigned int)>& clear)
{
    if( !d_active.getValue() ) return;

    for (unsigned int i = 0; i < res.size(); ++i)
        clear(res, i);
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    if( !d_active.getValue() ) return;

    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(res.wref(), [](VecDeriv& res, const unsigned int index) { res[index].clear(); });
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    if( !d_active.getValue() ) return;

    helper::WriteAccessor<DataVecDeriv> dx = vData;
    helper::ReadAccessor<DataVecCoord> x = ((MechanicalState*)this->getContext()->getMechanicalState())->readPositions();
    Real cT = (Real) this->getContext()->getTime() * d_animationSpeed.getValue();
    Real dt = (Real) this->getContext()->getDt();

    if(0.0 != cT)
    {
        findKeyTimes(cT+dt);
        if(finished)
        {
            // compute the position of the bones at cT + dt
            this->interpolatePosition<Coord>(cT+dt, nextPositions);
            // compute the velocity using finite difference
            for (unsigned i=0; i<nextPositions.size(); i++)
                dx[i] = DataTypes::coordDifference(nextPositions[i], x[i])/dt;

        }
        else
        {
            for(unsigned int i = 0; i < dx.size(); ++i)
                dx[i] = Deriv();
        }
    }
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    if( !d_active.getValue() ) return;

    helper::WriteAccessor<DataVecCoord> x = xData;
    Real cT = (Real) this->getContext()->getTime() * d_animationSpeed.getValue();

    if(0.0 != cT)
    {
        findKeyTimes(cT);

        // if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
        interpolatePosition<Coord>(cT, x.wref());
    }
}

template <class DataTypes>
template <class MyCoord>
void SkeletalMotionProjectiveConstraint<DataTypes>::interpolatePosition(Real cT, typename std::enable_if<std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x)
{
    // set the motion to the SkeletonJoint corresponding rigid
    
    if(finished)    
        for(unsigned int i = 0; i < d_skeletonJoints.getValue().size(); ++i)
        {
            
            SkeletonJoint<DataTypes>& skeletonJoint = (*d_skeletonJoints.beginEdit())[i];
            if(  skeletonJoint.mPreviousMotionTime !=  skeletonJoint.mNextMotionTime)
            {
                Real dt = (Real)((cT - skeletonJoint.mPreviousMotionTime) / (skeletonJoint.mNextMotionTime - skeletonJoint.mPreviousMotionTime));

                const type::vector<defaulttype::RigidCoord<3, Real> >& channels = skeletonJoint.mChannels;

                if(channels.empty())
                    continue;

                skeletonJoint.mLocalRigid.getCenter() = skeletonJoint.mPreviousMotion.getCenter() + (skeletonJoint.mNextMotion.getCenter() - skeletonJoint.mPreviousMotion.getCenter()) * dt;
                skeletonJoint.mLocalRigid.getOrientation().slerp(skeletonJoint.mPreviousMotion.getOrientation(), skeletonJoint.mNextMotion.getOrientation(), (float) dt, true);

                d_skeletonJoints.endEdit();
            }
            else
            {
                const type::vector<defaulttype::RigidCoord<3, Real> >& channels = skeletonJoint.mChannels;

                if(channels.empty())
                    continue;

                skeletonJoint.mLocalRigid.getCenter() = skeletonJoint.mNextMotion.getCenter();
                skeletonJoint.mLocalRigid.getOrientation() = skeletonJoint.mNextMotion.getOrientation();

                d_skeletonJoints.endEdit();
            }
        }
    else
    {
        for(unsigned int i = 0; i < d_skeletonJoints.getValue().size(); ++i)
        {
            SkeletonJoint<DataTypes>& skeletonJoint = (*d_skeletonJoints.beginEdit())[i];

            const type::vector<defaulttype::RigidCoord<3, Real> >& channels = skeletonJoint.mChannels;

            if(channels.empty())
                continue;

            skeletonJoint.mLocalRigid.getCenter() = skeletonJoint.mNextMotion.getCenter();
            skeletonJoint.mLocalRigid.getOrientation() = skeletonJoint.mNextMotion.getOrientation();

			d_skeletonJoints.endEdit();
        }
    }

    // apply the final transformation from d_skeletonBones to dofs here
    localToGlobal<Coord>(x);
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    if( !d_active.getValue() ) return;

    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    projectResponseT<MatrixDeriv>(c.wref(), [](MatrixDeriv& res, const unsigned int index) { res.clearColBlock(index); });
}

template <class DataTypes>
template <class MyCoord>
void SkeletalMotionProjectiveConstraint<DataTypes>::localToGlobal(typename std::enable_if<std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x)
{
    for(unsigned int i = 0; i < d_skeletonJoints.getValue().size(); ++i)
    {
        SkeletonJoint<DataTypes>& skeletonJoint = (*d_skeletonJoints.beginEdit())[i];

        defaulttype::RigidCoord< 3, Real> worldRigid = skeletonJoint.mLocalRigid;

        // break if the parent joint is the root
        for(int parentIndex = skeletonJoint.mParentIndex; -1 != parentIndex; parentIndex = d_skeletonJoints.getValue()[parentIndex].mParentIndex)
        {
            defaulttype::RigidCoord< 3, Real> parentLocalRigid = d_skeletonJoints.getValue()[parentIndex].mLocalRigid;
            worldRigid = parentLocalRigid.mult(worldRigid);
        }

        skeletonJoint.mWorldRigid = worldRigid;

		d_skeletonJoints.endEdit();
    }

    for(unsigned int i = 0; i < d_skeletonBones.getValue().size(); ++i)
        x[i] = d_skeletonJoints.getValue()[d_skeletonBones.getValue()[i]].mWorldRigid;
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::setSkeletalMotion(const type::vector<SkeletonJoint<DataTypes> >& skeletonJoints, const type::vector<SkeletonBone>& skeletonBones)
{
    this->d_skeletonJoints.setValue(skeletonJoints);
    this->d_skeletonBones.setValue(skeletonBones);
    this->init();
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::addChannel(unsigned int jointIndex , Coord channel, double time)
{
	(*d_skeletonJoints.beginEdit())[jointIndex].addChannel(channel, time);
	d_skeletonJoints.endEdit();
}

// Matrix Integration interface
template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/)
{
    if( !d_active.getValue() ) return;

    /*const unsigned int N = Deriv::size();
    const SetIndexArray & indices = d_indices.getValue();

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
void SkeletalMotionProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* /*mparams*/, linearalgebra::BaseVector* /*vector*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/)
{
    if( !d_active.getValue() ) return;

    /*const unsigned int N = Deriv::size();

    const SetIndexArray & indices = d_indices.getValue();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
    	for (unsigned int c=0;c<N;++c)
    		vect->clear(offset + N * (*it) + c);
    }*/
}

template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset )
{
    const unsigned blockSize = DataTypes::deriv_total_size;
    const unsigned size = this->mstate->getSize();
    for( unsigned i=0; i<size; i++ )
    {
        M->clearRowsCols( offset + i * blockSize, offset + (i+1) * (blockSize) );
    }
}

// display the paths the constrained dofs will go through
template <class DataTypes>
void SkeletalMotionProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if( !d_active.getValue() ) return;

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    sofa::type::vector<type::Vec3> points;
    sofa::type::vector<type::Vec3> linesX;
    sofa::type::vector<type::Vec3> linesY;
    sofa::type::vector<type::Vec3> linesZ;
    sofa::type::vector<type::Vec3> colorFalloff;

    type::Vec3 point;
    type::Vec3 line;

    // draw joints (not bones we draw them differently later)
    {
        for(unsigned int i = 0; i < d_skeletonJoints.getValue().size(); ++i)
        {
            defaulttype::RigidCoord< 3, Real> jointWorldRigid = d_skeletonJoints.getValue()[i].mWorldRigid;

            unsigned int j;
            for(j = 0; j < d_skeletonBones.getValue().size(); ++j)
                if((int)i == d_skeletonBones.getValue()[j])
                    break;

            if(d_skeletonBones.getValue().size() != j)
                continue;

            point = DataTypes::getCPos(jointWorldRigid);
            points.push_back(point);

            linesX.push_back(point);
            line = point + DataTypes::getCRot(jointWorldRigid).rotate(type::Vec3(0.1_sreal, 0.0_sreal, 0.0_sreal));
            linesX.push_back(line);

            linesY.push_back(point);
            line = point + DataTypes::getCRot(jointWorldRigid).rotate(type::Vec3(0.0_sreal, 0.1_sreal, 0.0_sreal));
            linesY.push_back(line);

            linesZ.push_back(point);
            line = point + DataTypes::getCRot(jointWorldRigid).rotate(type::Vec3(0.0_sreal, 0.0_sreal, 0.1_sreal));
            linesZ.push_back(line);
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::type::RGBAColor (1.0f , 0.5f , 0.5f , 1.0f));
        vparams->drawTool()->drawLines (linesX,  2, sofa::type::RGBAColor (0.75f, 0.0f , 0.0f , 1.0f));
        vparams->drawTool()->drawLines (linesY,  2, sofa::type::RGBAColor (0.0f , 0.75f, 0.0f , 1.0f));
        vparams->drawTool()->drawLines (linesZ,  2, sofa::type::RGBAColor (0.0f , 0.0f , 0.75f, 1.0f));
    }

    points.clear();
    linesX.clear();
    linesY.clear();
    linesZ.clear();

    // draw bones now
    {
        for(unsigned int i = 0; i < d_skeletonBones.getValue().size(); ++i)
        {
            const defaulttype::RigidCoord< 3, Real> boneWorldRigid = d_skeletonJoints.getValue()[d_skeletonBones.getValue()[i]].mWorldRigid;

            point = DataTypes::getCPos(boneWorldRigid);
            points.push_back(point);

            linesX.push_back(point);
            line = point + DataTypes::getCRot(boneWorldRigid).rotate(type::Vec3(0.1f, 0.0f, 0.0f));
            linesX.push_back(line);

            linesY.push_back(point);
            line = point + DataTypes::getCRot(boneWorldRigid).rotate(type::Vec3(0.0f, 0.1f, 0.0f));
            linesY.push_back(line);

            linesZ.push_back(point);
            line = point + DataTypes::getCRot(boneWorldRigid).rotate(type::Vec3(0.0f, 0.0f, 0.1f));
            linesZ.push_back(line);
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::type::RGBAColor (1.0f, 0.5f, 0.5f, 1.0f));
        vparams->drawTool()->drawLines (linesX, 2 , sofa::type::RGBAColor (1.0f, 0.0f, 0.0f, 1.0f));
        vparams->drawTool()->drawLines (linesY, 2 , sofa::type::RGBAColor (0.0f, 1.0f, 0.0f, 1.0f));
        vparams->drawTool()->drawLines (linesZ, 2 , sofa::type::RGBAColor (0.0f, 0.0f, 1.0f, 1.0f));
    }
}

} // namespace sofa::component::constraint::projective
