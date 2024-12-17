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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/type/vector.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::constraint::projective
{


/** impose a motion to given DOFs (translation and rotation)
	The motion between 2 key times is linearly interpolated
*/
template <class TDataTypes>
class LinearVelocityProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LinearVelocityProjectiveConstraint,TDataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,TDataTypes));

    using Index = sofa::Index;
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef type::vector<Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

public :
    /// indices of the DOFs the constraint is applied to
    SetIndex d_indices;
    /// the key frames when the motion is defined by the user
    Data<type::vector<Real> > d_keyTimes;
    /// the motions corresponding to the key frames
    Data<VecDeriv > d_keyVelocities;
    /// the coordinates on which to apply velocities
    SetIndex d_coordinates;

    /// If set to true then the last velocity will still be applied after all the key events
    Data<bool> d_continueAfterEnd;

    /// the key times surrounding the current simulation time (for interpolation)
    Real m_prevT, m_nextT;
    ///the velocities corresponding to the surrounding key times
    Deriv m_prevV, m_nextV;
    ///position at the initial step for constrained DOFs position
    VecCoord m_x0;
    ///position at the previous step for constrained DOFs position
    VecCoord m_xP;

    /// Link to be set to the topology container in the component graph.
    SingleLink<LinearVelocityProjectiveConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    LinearVelocityProjectiveConstraint();

    virtual ~LinearVelocityProjectiveConstraint();
public:
    ///methods to add/remove some indices, keyTimes, keyVelocity
    void clearIndices();
    void addIndex(Index index);
    void removeIndex(Index index);
    void clearKeyVelocities();
    /**add a new key movement
    @param time : the simulation time you want to set a movement (in sec)
    @param movement : the corresponding motion
    for instance, addKeyMovement(1.0, Deriv(5,0,0) ) will set a translation of 5 in x direction a time 1.0s
    **/
    void addKeyVelocity(Real time, Deriv movement);


    /// -- Constraint interface
    void init() override;
    void reset() override;
    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    void applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    void projectMatrix( sofa::linearalgebra::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;

    void applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix) override;

    void draw(const core::visual::VisualParams* vparams) override;

private:
    /// Check if the constraint is still active regarding the current time and the continueAfterEnd data
    bool isConstraintActive() const;

    /// to know if we found the key times
    bool m_finished;

    /// find previous and next time keys
    bool findKeyTimes();
};

#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARVELOCITYPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearVelocityProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearVelocityProjectiveConstraint<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearVelocityProjectiveConstraint<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearVelocityProjectiveConstraint<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearVelocityProjectiveConstraint<defaulttype::Rigid3Types>;

#endif

} // namespace sofa::component::constraint::projective
