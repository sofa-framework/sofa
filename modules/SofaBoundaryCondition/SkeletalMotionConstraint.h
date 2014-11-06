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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_SKELETALMOTIONCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_SKELETALMOTIONCONSTRAINT_H

#include <sofa/SofaGeneral.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/helper/SVector.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

// a joint of the skeletal hierarchy, it participates in the skeletal animation chain and may be animated
template <class DataTypes>
struct SkeletonJoint;

// joints index to export in the MechanicalObject (in order to use them for skinning for instance)
typedef int SkeletonBone;

// impose a specific motion (translation and rotation) for each DOFs of a MechanicalObject
template <class TDataTypes>
class SkeletalMotionConstraint : public sofa::core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SkeletalMotionConstraint,TDataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));
    typedef TDataTypes DataTypes;
    typedef sofa::core::behavior::ProjectiveConstraintSet<TDataTypes> TProjectiveConstraintSet;
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
    SkeletalMotionConstraint();

    virtual ~SkeletalMotionConstraint();

public:

    void init();
    void reset();

	float getAnimationSpeed() const			{return animationSpeed.getValue();}
	void setAnimationSpeed(float speed)		{animationSpeed.setValue(speed);}

    void findKeyTimes();

    void projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData);
    void projectVelocity(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& vData);
    void projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& xData);
    void projectJacobianMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataMatrixDeriv& cData);

    void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset);
    void applyConstraint(defaulttype::BaseVector *vect, unsigned int offset);

	void projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
	{
		unsigned blockSize = DataTypes::deriv_total_size;	
		unsigned size = this->mstate->getSize();
		for( unsigned i=0; i<size; i++ )
		{
			M->clearRowsCols( offset + i * blockSize, offset + (i+1) * (blockSize) );
		}
	}

    virtual void draw(const core::visual::VisualParams* vparams);

    template<class MyCoord>
    void localToGlobal(typename boost::enable_if<boost::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >, VecCoord>::type& x);

    void setSkeletalMotion(const helper::vector<SkeletonJoint<DataTypes> >& skeletonJoints, const helper::vector<SkeletonBone>& skeletonBones);

	void addChannel(unsigned int index , Coord channel, double time);

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataDeriv& dx);

    template<class MyCoord>
    void interpolatePosition(Real cT, typename boost::enable_if<boost::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >, VecCoord>::type& x);

protected:
	// every nodes needed in the animation chain
    Data<helper::SVector<SkeletonJoint<TDataTypes> > >	skeletonJoints;
    // mesh skeleton bones which need to be updated according to the animated nodes, we use them to fill the mechanical object
    Data<helper::SVector<SkeletonBone> >				skeletonBones;

	// control how fast the animation is played since animation time is not simulation time
	Data<float>											animationSpeed;

    /// is the projective constraint activated?
    Data<bool>                                          active;

private:
    /// the key times surrounding the current simulation time (for interpolation)
    Real												prevT, nextT;

    /// to know if we found the key times
    bool												finished;

};

template <class DataTypes>
struct SkeletonJoint
{
    friend class SkeletalMotionConstraint<DataTypes>;

    typedef typename DataTypes::Coord Coord;

    SkeletonJoint()
        : mParentIndex(-1)
        , mChannels()
        , mTimes()
    {

    }

    virtual ~SkeletonJoint()
    {

    }

	void addChannel(Coord channel, double time)
	{
		mChannels.push_back(channel);
		mTimes.push_back(time);
	}

    inline friend std::ostream& operator << (std::ostream& out, const SkeletonJoint& skeletonJoint)
    {
        out << "Parent"			<< " " << skeletonJoint.mParentIndex		<< " ";
        out << "Channels"		<< " " << skeletonJoint.mChannels.size()	<< " " << skeletonJoint.mChannels	<< " ";
        out << "Times"			<< " " << skeletonJoint.mTimes.size()		<< " " << skeletonJoint.mTimes		<< " ";
        out << "PreviousMotion"	<< " " << skeletonJoint.mPreviousMotion		<< " ";
        out << "NextMotion"		<< " " << skeletonJoint.mNextMotion			<< " ";
        out << "LocalRigid"		<< " " << skeletonJoint.mLocalRigid;

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
        for(size_t i = 0; i < numChannel; ++i)
        {
            in >> channel;
            skeletonJoint.mChannels[i] = channel;
        }

        size_t numTime;
        in >> tmp >> numTime;
        skeletonJoint.mTimes.resize(numTime);
        double time;
        for(size_t i = 0; i < numTime; ++i)
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
    helper::vector<Coord>				mChannels;

    // times corresponding to each animation channel, the channel mChannels[i] must be played at the time contained in mTimes[i]
    helper::vector<double>				mTimes;

private:

    // following data are used internally to compute the final joint transformation at a specific time using interpolation

    // previous joint motion
    Coord								mPreviousMotion;

    // next joint motion
    Coord								mNextMotion;

    // this rigid represent the animated node at a specific time relatively to its parent, it may be an interpolation between two channels
    // we need to store the current rigid in order to compute the final world position of its rigid children
    Coord								mLocalRigid;

    // mCurrentRigid in the world coordinate
    Coord								mWorldRigid;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_SKELETALMOTIONCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API SkeletalMotionConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API SkeletalMotionConstraint<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif

