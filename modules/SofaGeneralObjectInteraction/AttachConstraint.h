/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/PairInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <set>
#include <sofa/core/DataEngine.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/** Attach given pair of particles, projecting the positions of the second particles to the first ones.
*/
template <class DataTypes>
class AttachConstraint : public core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AttachConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::PairInteractionProjectiveConstraintSet,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;

public:
    SetIndex f_indices1; ///< Indices of the source points on the first model
    SetIndex f_indices2; ///< Indices of the fixed points on the second model
    Data<bool> f_twoWay; ///< true if forces should be projected back from model2 to model1
    Data<bool> f_freeRotations; ///< true to keep rotations free (only used for Rigid DOFs)
    Data<bool> f_lastFreeRotation; ///< true to keep rotation of the last attached point free (only used for Rigid DOFs)
    Data<bool> f_restRotations; ///< true to use rest rotations local offsets (only used for Rigid DOFs)
    Data<defaulttype::Vector3> f_lastPos; ///< position at which the attach constraint should become inactive
    Data<defaulttype::Vector3> f_lastDir; ///< direction from lastPos at which the attach coustraint should become inactive
    Data<bool> f_clamp; ///< true to clamp particles at lastPos instead of freeing them.
    Data<Real> f_minDistance; ///< the constraint become inactive if the distance between the points attached is bigger than minDistance.
    Data< Real > d_positionFactor;      ///< IN: Factor applied to projection of position
    Data< Real > d_velocityFactor;      ///< IN: Factor applied to projection of velocity
    Data< Real > d_responseFactor;      ///< IN: Factor applied to projection of force/acceleration
    Data< helper::vector<Real> > d_constraintFactor; ///< Constraint factor per pair of points constrained. 0 -> the constraint is released. 1 -> the constraint is fully constrained

    helper::vector<bool> activeFlags;
    helper::vector<bool> constraintReleased;
    helper::vector<Real> lastDist;
    helper::vector<defaulttype::Quat> restRotations;

protected:
    bool dynamicConstraintFactor;
    AttachConstraint();
    AttachConstraint(core::behavior::MechanicalState<DataTypes> *mm1, core::behavior::MechanicalState<DataTypes> *mm2);
    virtual ~AttachConstraint();
public:
    void init() override;
    void reinit() override;
    void projectJacobianMatrix(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, core::MultiMatrixDerivId /*cId*/) override;
    void projectResponse(const core::MechanicalParams *mparams, DataVecDeriv& dx1, DataVecDeriv& dx2) override;
    void projectVelocity(const core::MechanicalParams *mparams, DataVecDeriv& v1, DataVecDeriv& v2) override;
    void projectPosition(const core::MechanicalParams *mparams, DataVecCoord& x1, DataVecCoord& x2) override;

    /// Project the global Mechanical Matrix to constrained space using offset parameter
    void applyConstraint(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    /// Project the global Mechanical Vector to constrained space using offset parameter
    void applyConstraint(const core::MechanicalParams *mparams, defaulttype::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    virtual void reinitIfChanged();

    template<class T>
    static std::string templateName(const T* ptr= nullptr) {
        return core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>::templateName(ptr);
    }

    virtual void draw(const core::visual::VisualParams* vparams) override;

protected :

    using core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>::projectPosition;
    using core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>::projectVelocity;
    using core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>::projectResponse;

    void projectPosition(Coord& x1, Coord& x2, bool /*freeRotations*/, unsigned index)
    {
        reinitIfChanged();
        // do nothing if distance between x2 & x1 is bigger than f_minDistance
        if (f_minDistance.getValue() != -1 &&
            (x2 - x1).norm() > f_minDistance.getValue())
        {
            constraintReleased[index] = true;
            return;
        }
        constraintReleased[index] = false;

        sofa::helper::ReadAccessor< Data< helper::vector<Real> > > constraintFactor = d_constraintFactor;

        Deriv corr = (x2-x1)*(0.5*d_positionFactor.getValue()*constraintFactor[index]);

        x1 += corr;
        x2 -= corr;
    }

    void projectVelocity(Deriv& x1, Deriv& x2, bool /*freeRotations*/, unsigned index)
    {
        reinitIfChanged();
        // do nothing if distance between x2 & x1 is bigger than f_minDistance
        if (constraintReleased[index]) return;

        sofa::helper::ReadAccessor< Data< helper::vector<Real> > > constraintFactor = d_constraintFactor;
        Deriv corr = (x2-x1)*(0.5*d_velocityFactor.getValue()*constraintFactor[index]);

        x1 += corr;
        x2 -= corr;
    }

    void projectResponse(Deriv& dx1, Deriv& dx2, bool /*freeRotations*/, bool twoway, unsigned index)
    {
        reinitIfChanged();
        // do nothing if distance between x2 & x1 is bigger than f_minDistance
        if (constraintReleased[index]) return;

        if (!twoway)
        {
            dx2 = Deriv();
        }
        else
        {
            Deriv in1 = dx1;
            Deriv in2 = dx2;
            sofa::helper::ReadAccessor< Data< helper::vector<Real> > > constraintFactor = d_constraintFactor;

            dx1 += in2*(d_responseFactor.getValue()*constraintFactor[index]);
            dx2 += in1*(d_responseFactor.getValue()*constraintFactor[index]);
        }
    }

    static unsigned int DerivConstrainedSize(bool freeRotations);

    void calcRestRotations();
};


#if  !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHCONSTRAINT_CPP)
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<defaulttype::Vec3Types>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<defaulttype::Vec2Types>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<defaulttype::Vec1Types>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<defaulttype::Rigid3Types>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API AttachConstraint<defaulttype::Rigid2Types>;

#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
