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

#include <sofa/core/behavior/PairInteractionProjectiveConstraintSet.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

namespace sofa::component::constraint::projective
{

/** Attach given pair of particles, projecting the positions of the second particles to the first ones.
*/
template <class DataTypes>
class AttachProjectiveConstraint : public core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AttachProjectiveConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::PairInteractionProjectiveConstraintSet,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef type::vector<unsigned int> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

public:
    SetIndex f_indices1; ///< Indices of the source points on the first model
    SetIndex f_indices2; ///< Indices of the fixed points on the second model
    Data<bool> f_twoWay; ///< true if forces should be projected back from model2 to model1
    Data<bool> f_freeRotations; ///< true to keep rotations free (only used for Rigid DOFs)
    Data<bool> f_lastFreeRotation; ///< true to keep rotation of the last attached point free (only used for Rigid DOFs)
    Data<bool> f_restRotations; ///< true to use rest rotations local offsets (only used for Rigid DOFs)
    Data<type::Vec3> f_lastPos; ///< position at which the attach constraint should become inactive
    Data<type::Vec3> f_lastDir; ///< direction from lastPos at which the attach coustraint should become inactive
    Data<bool> f_clamp; ///< true to clamp particles at lastPos instead of freeing them.
    Data<Real> f_minDistance; ///< the constraint become inactive if the distance between the points attached is bigger than minDistance.
    Data< Real > d_positionFactor;      ///< IN: Factor applied to projection of position
    Data< Real > d_velocityFactor;      ///< IN: Factor applied to projection of velocity
    Data< Real > d_responseFactor;      ///< IN: Factor applied to projection of force/acceleration
    Data< type::vector<Real> > d_constraintFactor; ///< Constraint factor per pair of points constrained. 0 -> the constraint is released. 1 -> the constraint is fully constrained

    type::vector<bool> activeFlags;
    type::vector<bool> constraintReleased;
    type::vector<Real> lastDist;
    type::vector<type::Quat<SReal>> restRotations;

protected:
    AttachProjectiveConstraint();
    AttachProjectiveConstraint(core::behavior::MechanicalState<DataTypes> *mm1, core::behavior::MechanicalState<DataTypes> *mm2);
    ~AttachProjectiveConstraint() override;

public:

    /// Inherited from Base
    void init() override;
    void reinit() override;
    void draw(const core::visual::VisualParams* vparams) override;

    /// Inherited from Constraint
    void projectJacobianMatrix(const core::MechanicalParams* mparams, core::MultiMatrixDerivId cId) override;
    void projectResponse(const core::MechanicalParams *mparams, DataVecDeriv& dx1, DataVecDeriv& dx2) override;
    void projectVelocity(const core::MechanicalParams *mparams, DataVecDeriv& v1, DataVecDeriv& v2) override;
    void projectPosition(const core::MechanicalParams *mparams, DataVecCoord& x1, DataVecCoord& x2) override;

    /// Project the global Mechanical Matrix to constrained space using offset parameter
    void applyConstraint(const core::MechanicalParams *mparams,
                         const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    /// Project the global Mechanical Vector to constrained space using offset parameter
    void applyConstraint(const core::MechanicalParams *mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix) override;

    virtual void reinitIfChanged();

    template<class T>
    static std::string templateName(const T* ptr= nullptr) {
        return core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>::templateName(ptr);
    }

protected :
    const Real getConstraintFactor(const int index);
    void doProjectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index, Real positionFactor);
    void doProjectVelocity(Deriv& x1, Deriv& x2, bool freeRotations, unsigned index, Real velocityFactor);
    void doProjectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index, Real responseFactor);

    void calcRestRotations();
    static unsigned int DerivConstrainedSize(bool freeRotations);

};


#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AttachProjectiveConstraint<defaulttype::Rigid2Types>;
#endif

} // namespace sofa::component::constraint::projective
