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
#include <sofa/component/constraint/projective/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/type/SVector.h>
#include <type_traits>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::constraint::projective
{

// a joint of the skeletal hierarchy, it participates in the skeletal animation chain and may be animated
template <class DataTypes>
struct SkeletonJoint;

// joints index to export in the MechanicalObject (in order to use them for skinning for instance)
typedef int SkeletonBone;

// impose a specific motion (translation and rotation) for each DOFs of a MechanicalObject
template <class TDataTypes>
class SkeletalMotionProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SkeletalMotionProjectiveConstraint,TDataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));
    typedef TDataTypes DataTypes;
    typedef sofa::core::behavior::ProjectiveConstraintSet<TDataTypes> TProjectiveConstraintSet;
    typedef sofa::core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;

protected:
    SkeletalMotionProjectiveConstraint();

    virtual ~SkeletalMotionProjectiveConstraint();

public:

    void init() override;
    void reset() override;

	float getAnimationSpeed() const			{return d_animationSpeed.getValue();}
	void setAnimationSpeed(float speed)		{d_animationSpeed.setValue(speed);}

    void findKeyTimes(Real ct);

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    using core::behavior::ProjectiveConstraintSet<TDataTypes>::applyConstraint;
    void applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    void projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset ) override;

    void draw(const core::visual::VisualParams* vparams) override;

    template<class MyCoord>
    void localToGlobal(typename std::enable_if<std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x);

    void setSkeletalMotion(const type::vector<SkeletonJoint<DataTypes> >& skeletonJoints, const type::vector<SkeletonBone>& skeletonBones);

	void addChannel(unsigned int jointIndex , Coord channel, double time);

protected:
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx,
        const std::function<void(DataDeriv&, const unsigned int)>& clear);

    template<class MyCoord>
    void interpolatePosition(Real cT, typename std::enable_if<std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x);

protected:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<type::SVector<SkeletonJoint<TDataTypes> > > skeletonJoints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<type::SVector<SkeletonBone> > skeletonBones;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<float> animationSpeed;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> active;


    // every nodes needed in the animation chain
    Data<type::SVector<SkeletonJoint<TDataTypes> > >	d_skeletonJoints; ///< skeleton joints
    // mesh skeleton bones which need to be updated according to the animated nodes, we use them to fill the mechanical object
    Data<type::SVector<SkeletonBone> >				d_skeletonBones; ///< skeleton bones

	// control how fast the animation is played since animation time is not simulation time
	Data<float>											d_animationSpeed; ///< animation speed

    /// is the projective constraint activated?
    Data<bool>                                          d_active;

private:
    /// the key times surrounding the current simulation time (for interpolation)
    Real												prevT, nextT;

    /// the global position of the bones at time t+dt used for the velocities computation
    VecCoord nextPositions;

    /// to know if we found the key times
    bool												finished;

};

template <class DataTypes>
struct SkeletonJoint
{
    friend class SkeletalMotionProjectiveConstraint<DataTypes>;

    typedef typename DataTypes::Coord Coord;

    SkeletonJoint()
        : mParentIndex(-1)
        , mChannels()
        , mTimes()
        , mPreviousMotionTime(0)
        , mNextMotionTime(0)
    {}

    virtual ~SkeletonJoint(){}

	void addChannel(Coord channel, double time)
	{
		mChannels.push_back(channel);
		mTimes.push_back(time);
	}

    inline friend std::ostream& operator << (std::ostream& out, const SkeletonJoint& skeletonJoint)
    {
        out << "Parent"			    << " " << skeletonJoint.mParentIndex		<< " ";
        out << "Channels"		    << " " << skeletonJoint.mChannels.size()	<< " " << skeletonJoint.mChannels	<< " ";
        out << "Times"			    << " " << skeletonJoint.mTimes.size()		<< " " << skeletonJoint.mTimes		<< " ";
        out << "PreviousMotion"	    << " " << skeletonJoint.mPreviousMotion		<< " ";
        out << "PreviousMotionTime" << " " << skeletonJoint.mPreviousMotionTime << " ";
        out << "NextMotion"		    << " " << skeletonJoint.mNextMotion			<< " ";
        out << "NextMotionTime"     << " " << skeletonJoint.mNextMotionTime     << " ";
        out << "LocalRigid"		    << " " << skeletonJoint.mLocalRigid;

        return out;
    }

    inline friend std::istream& operator >> (std::istream& in, SkeletonJoint& skeletonJoint)
    {
        std::string tmp;

        in >> tmp >> skeletonJoint.mParentIndex;

        size_t numChannel;
        in >> tmp >> numChannel;
        skeletonJoint.mChannels.resize(numChannel);
        Coord channel;
        for(Size i = 0; i < numChannel; ++i)
        {
            in >> channel;
            skeletonJoint.mChannels[i] = channel;
        }

        size_t numTime;
        in >> tmp >> numTime;
        skeletonJoint.mTimes.resize(numTime);
        double time;
        for(Size i = 0; i < numTime; ++i)
        {
            in >> time;
            skeletonJoint.mTimes[i] = time;
        }

        in >> tmp >> skeletonJoint.mPreviousMotion;
        in >> tmp >> skeletonJoint.mNextMotion;
        in >> tmp >> skeletonJoint.mLocalRigid;

        return in;
    }

    // parent joint, set to -1 if root, you must set this value
    int									mParentIndex;

    // set the joint rest position, you must set this value
    void setRestPosition(const Coord& restPosition)
    {
        mPreviousMotion = restPosition;
        mNextMotion = restPosition;
        mLocalRigid = restPosition;
    }

    // following data are useful for animation only, you must fill those vectors if this joint is animated

    // each channel represents a local transformation at a given time in the animation
    type::vector<Coord>				mChannels;

    // times corresponding to each animation channel, the channel mChannels[i] must be played at the time contained in mTimes[i]
    type::vector<double>				mTimes;

private:

    // following data are used internally to compute the final joint transformation at a specific time using interpolation

    // previous joint motion
    Coord								mPreviousMotion;

    // next joint motion
    Coord								mNextMotion;

    // time position for previous joint motion 
    double                              mPreviousMotionTime;

    // time position for next joint motion 
    double                              mNextMotionTime;

    // this rigid represent the animated node at a specific time relatively to its parent, it may be an interpolation between two channels
    // we need to store the current rigid in order to compute the final world position of its rigid children
    Coord								mLocalRigid;

    // mCurrentRigid in the world coordinate
    Coord								mWorldRigid;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_SKELETALMOTIONPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API SkeletalMotionProjectiveConstraint<defaulttype::Rigid3Types>;

#endif

} // namespace sofa::component::constraint::projective

