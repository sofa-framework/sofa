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
#include <sofa/component/constraint/projective/AttachProjectiveConstraint.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/simulation/Node.h>

namespace sofa::component::constraint::projective
{

using sofa::simulation::Node ;

template<>
inline void AttachProjectiveConstraint<defaulttype::Rigid3Types>::doProjectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index, Real positionFactor)
{
    SOFA_UNUSED(positionFactor);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (f_minDistance.getValue() != -1.0 &&
            (x2.getCenter() - x1.getCenter()).norm() > f_minDistance.getValue())
    {
        constraintReleased[index] = true;
        return;
    }
    constraintReleased[index] = false;

    x2.getCenter() = x1.getCenter();
    if (!freeRotations)
    {
        if (!restRotations.empty())
        {
            if (index+1 >= lastDist.size() || activeFlags[index+1])
                x2.getOrientation() = x1.getOrientation()*restRotations[index];
            else
            {
                // gradually set the velocity along the direction axis
                const Real fact = -lastDist[index] / (lastDist[index+1]-lastDist[index]);
                const sofa::type::Vec3 axis(restRotations[index][0], restRotations[index][1], restRotations[index][2]);
                const Real angle = acos(restRotations[index][3])*2;
                x2.getOrientation() = x1.getOrientation()*sofa::type::Quat<SReal>(axis,angle*fact);
            }
        }
        else
            x2.getOrientation() = x1.getOrientation();
    }
}


template<>
inline void AttachProjectiveConstraint<defaulttype::Rigid2Types>::doProjectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index, Real positionFactor)
{
    SOFA_UNUSED(positionFactor);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (f_minDistance.getValue() != -1 &&
            (x2.getCenter() - x1.getCenter()).norm() > f_minDistance.getValue())
    {
        constraintReleased[index] = true;
        return;
    }
    constraintReleased[index] = false;

    x2.getCenter() = x1.getCenter();
    if (!freeRotations)
        x2.getOrientation() = x1.getOrientation();
}


template<>
inline void AttachProjectiveConstraint<defaulttype::Rigid3Types>::doProjectVelocity(Deriv& x1, Deriv& x2, bool freeRotations, unsigned index, Real velocityFactor)
{
    SOFA_UNUSED(velocityFactor);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index])
        return;

    getVCenter( x2) = getVCenter(x1);
    if (!freeRotations)
        getVOrientation(x2) = getVOrientation(x1);
}


template<>
inline void AttachProjectiveConstraint<defaulttype::Rigid2Types>::doProjectVelocity(Deriv& x1, Deriv& x2, bool freeRotations, unsigned index, Real velocityFactor)
{
    SOFA_UNUSED(velocityFactor);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    getVCenter(x2) = getVCenter(x1);
    if (!freeRotations)
        getVOrientation(x2) = getVOrientation(x1);
}

template<>
inline void AttachProjectiveConstraint<defaulttype::Rigid3Types>::doProjectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index, Real responseFactor)
{
    SOFA_UNUSED(responseFactor);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    if (!twoway)
    {
        if (!freeRotations)
            dx2 = Deriv();
        else
            getVCenter(dx2).clear();
    }
    else
    {
        if (!freeRotations)
        {
            dx1 += dx2;
            dx2 = dx1;
        }
        else
        {
            getVCenter(dx1) += getVCenter(dx2);
            getVCenter(dx2) = getVCenter(dx1);
        }
    }
}


template<>
inline void AttachProjectiveConstraint<defaulttype::Rigid2Types>::doProjectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index, Real responseFactor)
{
    SOFA_UNUSED(responseFactor);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    if (!twoway)
    {
        if (!freeRotations)
            dx2 = Deriv();
        else
            getVCenter(dx2).clear();
    }
    else
    {
        if (!freeRotations)
        {
            dx1 += dx2;
            dx2 = dx1;
        }
        else
        {
            getVCenter(dx1) += getVCenter(dx2);
            getVCenter(dx2) = getVCenter(dx1);
        }
    }
}

template<class DataTypes>
inline unsigned int AttachProjectiveConstraint<DataTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (std::is_same<DataTypes, defaulttype::Rigid2Types>::value || std::is_same<DataTypes, defaulttype::Rigid3Types>::value) {
        if (freeRotations)
            return Deriv::spatial_dimensions;
        else
            return Deriv::total_size;
    }
    else {
        SOFA_UNUSED(freeRotations);
        return Deriv::size();
    }
}

// Could be simplified with default values for mm1 and mm2, but that way we are assured that either both or neither of mm1/mm2 are set.
template <class DataTypes>
AttachProjectiveConstraint<DataTypes>::AttachProjectiveConstraint()
    : AttachProjectiveConstraint<DataTypes>::AttachProjectiveConstraint(nullptr, nullptr)
{
}

template <class DataTypes>
AttachProjectiveConstraint<DataTypes>::AttachProjectiveConstraint(core::behavior::MechanicalState<DataTypes> *mm1, core::behavior::MechanicalState<DataTypes> *mm2)
    : core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>(mm1,mm2)
    , f_indices1( initData(&f_indices1,"indices1","Indices of the source points on the first model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the fixed points on the second model") )
    , f_twoWay( initData(&f_twoWay,false,"twoWay", "true if forces should be projected back from model2 to model1") )
    , f_freeRotations( initData(&f_freeRotations,false,"freeRotations", "true to keep rotations free (only used for Rigid DOFs)") )
    , f_lastFreeRotation( initData(&f_lastFreeRotation,false,"lastFreeRotation", "true to keep rotation of the last attached point free (only used for Rigid DOFs)") )
    , f_restRotations( initData(&f_restRotations,false,"restRotations", "true to use rest rotations local offsets (only used for Rigid DOFs)") )
    , f_lastPos( initData(&f_lastPos,"lastPos", "position at which the attach constraint should become inactive") )
    , f_lastDir( initData(&f_lastDir,"lastDir", "direction from lastPos at which the attach coustraint should become inactive") )
    , f_clamp( initData(&f_clamp, false,"clamp", "true to clamp particles at lastPos instead of freeing them.") )
    , f_minDistance( initData(&f_minDistance, static_cast<Real>(-1),"minDistance", "the constraint become inactive if the distance between the points attached is bigger than minDistance.") )
    , d_positionFactor(initData(&d_positionFactor, static_cast<Real>(1.0), "positionFactor", "IN: Factor applied to projection of position"))
    , d_velocityFactor(initData(&d_velocityFactor, static_cast<Real>(1.0), "velocityFactor", "IN: Factor applied to projection of velocity"))
    , d_responseFactor(initData(&d_responseFactor, static_cast<Real>(1.0), "responseFactor", "IN: Factor applied to projection of force/acceleration"))
    , d_constraintFactor( initData(&d_constraintFactor,"constraintFactor","Constraint factor per pair of points constrained. 0 -> the constraint is released. 1 -> the constraint is fully constrained") )
{

}

template <class DataTypes>
AttachProjectiveConstraint<DataTypes>::~AttachProjectiveConstraint()
{
}

template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::init()
{
    this->core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>::init();
    reinit();
}

template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::reinit()
{
    // Check coherency of size between indices vectors 1 and 2
    if(f_indices1.getValue().size() != f_indices2.getValue().size())
    {
        msg_warning() << "Size mismatch between indices1 and indices2 ("
                      << f_indices1.getValue().size() << " != " << f_indices2.getValue().size() << ").";
    }

    // Set to the correct length if dynamic, else check coherency.
    if(d_constraintFactor.getValue().size())
    {
        helper::ReadAccessor<Data<type::vector<Real>>> constraintFactor = d_constraintFactor;
        if(constraintFactor.size() != f_indices2.getValue().size())
        {
            msg_warning() << "Size of vector constraintFactor, do not fit number of indices attached (" << constraintFactor.size() << " != " << f_indices2.getValue().size() << ").";
        }
        else
        {
            for (unsigned int j=0; j<constraintFactor.size(); ++j)
            {
                if((constraintFactor[j] > 1.0) || (constraintFactor[j] < 0.0))
                {
                    msg_warning() << "Value of vector constraintFactor at indice "<<j<<" is out of bounds [0.0 - 1.0]";
                }
            }
        }
    }

    constraintReleased.resize(f_indices2.getValue().size());
    activeFlags.resize(f_indices2.getValue().size());
    std::fill(activeFlags.begin(), activeFlags.end(), true);
    if (f_lastDir.isSet() && f_lastDir.getValue().norm() > 1.0e-10) {
        lastDist.resize(f_indices2.getValue().size());
    }

    if (f_restRotations.getValue())
        calcRestRotations();
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class DataTypes>
void AttachProjectiveConstraint<DataTypes>::reinitIfChanged() {
    if((f_indices1.getParent() || f_indices2.getParent()) && constraintReleased.size() != f_indices2.getValue().size())
    {
        reinit();
    }
}

template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::calcRestRotations()
{
}

template <>
void AttachProjectiveConstraint<sofa::defaulttype::Rigid3Types>::calcRestRotations();

template<class DataTypes>
void AttachProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, core::MultiMatrixDerivId cId)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(cId);
}

template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams * mparams, DataVecCoord& res1_d, DataVecCoord& res2_d)
{
    SOFA_UNUSED(mparams);
    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const bool freeRotations = f_freeRotations.getValue();
    const bool lastFreeRotation = f_lastFreeRotation.getValue();
    const bool last = (f_lastDir.isSet() && f_lastDir.getValue().norm() > 1.0e-10);
    const bool clamp = f_clamp.getValue();

    VecCoord &res1 = *res1_d.beginEdit();
    VecCoord &res2 = *res2_d.beginEdit();

    // update active flags
    reinitIfChanged();

    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;
        if (last)
        {
            Coord p = res1[indices1[i]];
            sofa::type::Vec<3,Real> p3d;
            DataTypes::get(p3d[0],p3d[1],p3d[2],p);
            lastDist[i] = (Real)( (p3d-f_lastPos.getValue())*f_lastDir.getValue());
            if (lastDist[i] > 0.0)
            {
                if (clamp)
                {
                    msg_info_when(activeFlags[i]) << "AttachProjectiveConstraint: point "
                                                  <<indices1[i]<<" stopped." ;
                }
                else
                {
                    msg_info_when(activeFlags[i]) << "AttachProjectiveConstraint: point "
                                                  <<indices1[i]<<" is free.";
                }
                active = false;
            }
        }
        activeFlags[i] = active;
    }
    helper::ReadAccessor<Data<Real>> positionFactor = d_positionFactor;
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        Coord p = res1[indices1[i]];
        if (activeFlags[i])
        {
            msg_info() << "AttachProjectiveConstraint: x2["<<indices2[i]<<"] = x1["<<indices1[i]<<"]";

            doProjectPosition(p, res2[indices2[i]], freeRotations || (lastFreeRotation && (i>=activeFlags.size() || !activeFlags[i+1])), i, positionFactor);
        }
        else if (clamp)
        {
            DataTypes::set(p,f_lastPos.getValue()[0],f_lastPos.getValue()[1],f_lastPos.getValue()[2]);

            msg_info() << "AttachProjectiveConstraint: x2["<<indices2[i]<<"] = lastPos";

            doProjectPosition(p, res2[indices2[i]], freeRotations, i, positionFactor);
        }
    }

    res1_d.endEdit();
    res2_d.endEdit();
}

template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams * mparams, DataVecDeriv& res1_d, DataVecDeriv& res2_d)
{
    SOFA_UNUSED(mparams);
    VecDeriv &res1 = *res1_d.beginEdit();
    VecDeriv &res2 = *res2_d.beginEdit();

    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const bool freeRotations = f_freeRotations.getValue();
    const bool lastFreeRotation = f_lastFreeRotation.getValue();
    const bool clamp = f_clamp.getValue();

    reinitIfChanged();
    helper::ReadAccessor<Data<Real>> velocityFactor = d_velocityFactor;
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;

        if (i < activeFlags.size())
            active = activeFlags[i];

        if (active)
        {
            msg_info() << "AttachProjectiveConstraint: v2["<<indices2[i]<<"] = v1["<<indices1[i]<<"]" ;

            doProjectVelocity(res1[indices1[i]], res2[indices2[i]], freeRotations || (lastFreeRotation && (i>=activeFlags.size() || !activeFlags[i+1])), i, velocityFactor);
        }
        else if (clamp)
        {
            msg_info() << "AttachProjectiveConstraint: v2["<<indices2[i]<<"] = 0" ;

            Deriv v = Deriv();
            doProjectVelocity(v, res2[indices2[i]], freeRotations, i, velocityFactor);
        }
    }

    res1_d.endEdit();
    res2_d.endEdit();
}

template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams * mparams, DataVecDeriv& res1_d, DataVecDeriv& res2_d)
{
    SOFA_UNUSED(mparams);
    VecDeriv &res1 = *res1_d.beginEdit();
    VecDeriv &res2 = *res2_d.beginEdit();

    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const bool twoway = f_twoWay.getValue();
    const bool freeRotations = f_freeRotations.getValue();
    const bool lastFreeRotation = f_lastFreeRotation.getValue();
    const bool clamp = f_clamp.getValue();

    reinitIfChanged();
    helper::ReadAccessor<Data<Real>> responseFactor = d_responseFactor;
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;

        if (i < activeFlags.size())
            active = activeFlags[i];

        if (active)
        {
            if (twoway){
                msg_info() << " r2["<<indices2[i]<<"] = r1["<<indices2[i]<<"] = (r2["<<indices2[i]<<"] + r2["<<indices2[i]<<"])";
            }else{
                msg_info() << " r2["<<indices2[i]<<"] = 0";
            }

            doProjectResponse(res1[indices1[i]], res2[indices2[i]], freeRotations || (lastFreeRotation && (i>=activeFlags.size() || !activeFlags[i+1])), twoway, i, responseFactor);

            msg_info() << " final r2["<<indices2[i]<<"] = "<<res2[indices2[i]]<<"";
        }
        else if (clamp)
        {
            msg_info() << " r2["<<indices2[i]<<"] = 0";

            Deriv v = Deriv();
            doProjectResponse(v, res2[indices2[i]], freeRotations, false, i, responseFactor);

            msg_info() << " final r2["<<indices2[i]<<"] = "<<res2[indices2[i]]<<"";
        }
    }

    res1_d.endEdit();
    res2_d.endEdit();
}


// Matrix Integration interface
template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams * mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    if (f_twoWay.getValue())
        return;

    const sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate2);
    if (!r)
        return;

    sofa::linearalgebra::BaseMatrix *mat = r.matrix;
    const unsigned int offset = r.offset;

    const SetIndexArray & indices = f_indices2.getValue();
    const unsigned int N = Deriv::size();
    const unsigned int NC = DerivConstrainedSize(f_freeRotations.getValue());
    const unsigned int NCLast = DerivConstrainedSize(f_lastFreeRotation.getValue());
    unsigned int i=0;
    const bool clamp = f_clamp.getValue();

    reinitIfChanged();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it, ++i)
    {
        if (!clamp && i < activeFlags.size() && !activeFlags[i])
            continue;

        msg_info() << "AttachProjectiveConstraint: apply in matrix column/row "<<(*it);

        if (NCLast != NC && (i>=activeFlags.size() || !activeFlags[i+1]))
        {
            // Reset Fixed Row and Col
            for (unsigned int c=0; c<NCLast; ++c)
                mat->clearRowCol(offset + N * (*it) + c);
            // Set Fixed Vertex
            for (unsigned int c=0; c<NCLast; ++c)
                mat->set(offset + N * (*it) + c, offset + N * (*it) + c, 1.0);
        }
        else
        {
            // Reset Fixed Row and Col
            for (unsigned int c=0; c<NC; ++c)
                mat->clearRowCol(offset + N * (*it) + c);
            // Set Fixed Vertex
            for (unsigned int c=0; c<NC; ++c)
                mat->set(offset + N * (*it) + c, offset + N * (*it) + c, 1.0);
        }
    }
}


template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams * mparams, linearalgebra::BaseVector* vect, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    if (f_twoWay.getValue())
        return;

    const int o = matrix->getGlobalOffset(this->mstate2);
    if (o < 0)
        return;

    unsigned int offset = (unsigned int)o;

    msg_info() << "applyConstraint in Vector with offset = " << offset ;

    const SetIndexArray & indices = f_indices2.getValue();
    const unsigned int N = Deriv::size();
    const unsigned int NC = DerivConstrainedSize(f_freeRotations.getValue());
    const unsigned int NCLast = DerivConstrainedSize(f_lastFreeRotation.getValue());
    unsigned int i = 0;
    const bool clamp = f_clamp.getValue();

    reinitIfChanged();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it, ++i)
    {
        if (!clamp && i < activeFlags.size() && !activeFlags[i])
            continue;

        if (NCLast != NC && (i>=activeFlags.size() || !activeFlags[i+1]))
        {
            for (unsigned int c=0; c<NCLast; ++c)
                vect->clear(offset + N * (*it) + c);
        }
        else
        {
            for (unsigned int c=0; c<NC; ++c)
                vect->clear(offset + N * (*it) + c);
        }
    }
}

template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix)
{
    if (f_twoWay.getValue())
        return;

    reinitIfChanged();

    static constexpr unsigned int N = Deriv::size();
    const SetIndexArray& indices = f_indices2.getValue();
    const unsigned int NC = DerivConstrainedSize(f_freeRotations.getValue());
    const unsigned int NCLast = DerivConstrainedSize(f_lastFreeRotation.getValue());
    unsigned int i = 0;
    const bool clamp = f_clamp.getValue();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it, ++i)
    {
        if (!clamp && i < activeFlags.size() && !activeFlags[i])
            continue;

        auto index = (*it);

        if (NCLast != NC && (i >= activeFlags.size() || !activeFlags[i + 1]))
        {
            // Reset Fixed Row and Col
            for (unsigned int c = 0; c < NCLast; ++c)
            {
                matrix->discardRowCol(N * index + c, N * index + c);
            }
        }
        else
        {
            // Reset Fixed Row and Col
            for (unsigned int c = 0; c < NC; ++c)
            {
                matrix->discardRowCol(N * index + c, N * index + c);
            }
        }

        ++i;
    }
}

template<class DataTypes>
const typename DataTypes::Real AttachProjectiveConstraint<DataTypes>::getConstraintFactor(const int index) {
    return d_constraintFactor.getValue().size() ? d_constraintFactor.getValue()[index] : 1;
}

template<class DataTypes>
void AttachProjectiveConstraint<DataTypes>::doProjectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index, Real positionFactor)
{
    SOFA_UNUSED(freeRotations);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (f_minDistance.getValue() != -1 &&
            (x2 - x1).norm() > f_minDistance.getValue())
    {
        constraintReleased[index] = true;
        return;
    }
    constraintReleased[index] = false;

    Deriv corr = (x2-x1)*(0.5*positionFactor*getConstraintFactor(index));

    x1 += corr;
    x2 -= corr;
}

template<class DataTypes>
void AttachProjectiveConstraint<DataTypes>::doProjectVelocity(Deriv &x1, Deriv &x2, bool freeRotations, unsigned index, Real velocityFactor)
{
    SOFA_UNUSED(freeRotations);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    Deriv corr = (x2-x1)*(0.5*velocityFactor*getConstraintFactor(index));

    x1 += corr;
    x2 -= corr;
}

template<class DataTypes>
void AttachProjectiveConstraint<DataTypes>::doProjectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index, Real responseFactor)
{
    SOFA_UNUSED(freeRotations);
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    if (!twoway)
    {
        dx2 = Deriv();
    }
    else
    {
        Deriv corr = (dx2-dx1)*0.5*responseFactor*getConstraintFactor(index);
        dx1 += corr;
        dx2 -= corr;
    }
}


template <class DataTypes>
void AttachProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const VecCoord& x1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    constexpr sofa::type::RGBAColor color1(1,0.5,0.5,1);
    std::vector<sofa::type::Vec3> vertices;

    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        if (activeFlags.size() > i && !activeFlags[i])
            continue;
        vertices.push_back(sofa::type::Vec3(x2[indices2[i]][0],x2[indices2[i]][1],x2[indices2[i]][2]));
    }
    vparams->drawTool()->drawPoints(vertices,10,color1);
    vertices.clear();

    constexpr sofa::type::RGBAColor color2(1,0.5,0.5,1);
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        if (activeFlags.size() > i && !activeFlags[i])
            continue;
        vertices.push_back(sofa::type::Vec3(x1[indices1[i]][0],x1[indices1[i]][1],x1[indices1[i]][2]));
        vertices.push_back(sofa::type::Vec3(x2[indices2[i]][0],x2[indices2[i]][1],x2[indices2[i]][2]));
    }
    vparams->drawTool()->drawLines(vertices,1,color2);

}

} // namespace sofa::component::constraint::projective
