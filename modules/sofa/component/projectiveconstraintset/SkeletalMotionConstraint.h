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

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using core::objectmodel::Data;
using namespace sofa::core::objectmodel;
using namespace sofa::core::behavior;
using namespace sofa::defaulttype;

// a skeleton bone, here used only for animation, use a SkinningMapping if you want to skin a mesh
// and if you want to do so, use the same index in the mechanical object mapped with the SkinningMapping
// than in the skeletonBones array of the SkeletalMotionConstraint defined below
struct SkeletonBone
{
    SkeletonBone()
        : mSkeletonJointIndex(-1)
    {

    }

    // the corresponding skeleton joint
    int									mSkeletonJointIndex;
};

// a skeleton joint which may be animated and which participates in the skeletal animation chain (it does not always correspond to a bone but it is always useful in order to compute the skeleton bones world transformation)
template <class DataTypes>
struct SkeletonJoint;

template <class DataTypes>
struct SkeletonJoint
{
    typedef typename DataTypes::Real Real;

    SkeletonJoint()
        : mParentIndex(-1)
    {

    }

    // parent animated node
    int									mParentIndex;

    // each channel represents a transformation matrix which is the node local transformation at a given frame in the animation
    std::vector<RigidCoord<3, Real> >	mChannels;

    // times corresponding to each animation channel, the channel mChannels[i] must be played at the time contained in mTimes[i]
    std::vector<double>					mTimes;

    // previous node motion
    RigidCoord<3, Real>					mPreviousMotion;

    // next node motion
    RigidCoord<3, Real>					mNextMotion;

    // this rigid represent the animated node at a specific time relatively to its parent, it may be an interpolation between two channels
    // we need to store the current rigid in order to compute the final world position of its rigid children
    RigidCoord<3, Real>					mLocalRigid;

    // mCurrentRigid in the world coordinate
    RigidCoord<3, Real>					mWorldRigid;
};

// impose a specific motion (translation and rotation) for each DOFs of a MechanicalObject
template <class TDataTypes>
class SkeletalMotionConstraint : public ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SkeletalMotionConstraint,TDataTypes),SOFA_TEMPLATE(ProjectiveConstraintSet, TDataTypes));
    typedef TDataTypes DataTypes;
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

    void findKeyTimes();

    void projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData);
    void projectVelocity(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& vData);
    void projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& xData);
    void projectJacobianMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataMatrixDeriv& cData);

    void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset);
    void applyConstraint(defaulttype::BaseVector *vect, unsigned int offset);

    virtual void draw(const core::visual::VisualParams* vparams);

    template<class MyCoord>
    void localToGlobal(typename boost::enable_if<boost::is_same<MyCoord, RigidCoord<3, Real> >, VecCoord>::type& x);

    void setSkeletalMotion(const std::vector<SkeletonJoint<TDataTypes> >& skeletonJoints, const std::vector<SkeletonBone>& skeletonBones);

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataDeriv& dx);

    template<class MyCoord>
    void interpolatePosition(Real cT, typename boost::enable_if<boost::is_same<MyCoord, RigidCoord<3, Real> >, VecCoord>::type& x);

private:
    // every nodes needed in the animation chain
    std::vector<SkeletonJoint<TDataTypes> >		skeletonJoints;

    // mesh skeletonBones that will need to be updated according to the animated nodes, we use them to fill the mechanical object
    std::vector<SkeletonBone>					skeletonBones;

    /// the key times surrounding the current simulation time (for interpolation)
    Real										prevT, nextT;

    /// to know if we found the key times
    bool										finished;

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

