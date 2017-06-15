/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ATTACHCONSTRAINT_INL

#include <SofaGeneralObjectInteraction/AttachConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <SofaBaseTopology/TopologySubsetData.inl>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

#ifndef SOFA_FLOAT
template<>
inline void AttachConstraint<defaulttype::Rigid3dTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index)
{
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
    {
        if (!restRotations.empty())
        {
            if (index+1 >= lastDist.size() || activeFlags[index+1])
                x2.getOrientation() = x1.getOrientation()*restRotations[index];
            else
            {
                // gradually set the velocity along the direction axis
                Real fact = -lastDist[index] / (lastDist[index+1]-lastDist[index]);
                sofa::defaulttype::Vector3 axis(restRotations[index][0], restRotations[index][1], restRotations[index][2]);
                Real angle = acos(restRotations[index][3])*2;
                //restRotations[index].toAngleAxis(angle,axis);
                x2.getOrientation() = x1.getOrientation()*sofa::defaulttype::Quat(axis,angle*fact);
            }
        }
        else
            x2.getOrientation() = x1.getOrientation();
    }
}
#endif

#ifndef SOFA_DOUBLE
template<>
inline void AttachConstraint<defaulttype::Rigid3fTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index)
{
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
#endif

#ifndef SOFA_FLOAT
template<>
inline void AttachConstraint<defaulttype::Rigid2dTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index)
{
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
#endif

#ifndef SOFA_DOUBLE
template<>
inline void AttachConstraint<defaulttype::Rigid2fTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations, unsigned index)
{
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
#endif

#ifndef SOFA_FLOAT
template<>
inline void AttachConstraint<defaulttype::Rigid3dTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations, unsigned index)
{
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    getVCenter( x2) = getVCenter(x1);
    if (!freeRotations)
        getVOrientation(x2) = getVOrientation(x1);
    //x2 = Deriv();
}
#endif

#ifndef SOFA_DOUBLE
template<>
inline void AttachConstraint<defaulttype::Rigid3fTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations, unsigned index)
{
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    getVCenter(x2) = getVCenter(x1);
    if (!freeRotations)
        getVOrientation(x2) = getVOrientation(x1);
}
#endif

#ifndef SOFA_FLOAT
template<>
inline void AttachConstraint<defaulttype::Rigid2dTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations, unsigned index)
{
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    getVCenter(x2) = getVCenter(x1);
    if (!freeRotations)
        getVOrientation(x2) = getVOrientation(x1);
}
#endif

#ifndef SOFA_DOUBLE
template<>
inline void AttachConstraint<defaulttype::Rigid2fTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations, unsigned index)
{
    // do nothing if distance between x2 & x1 is bigger than f_minDistance
    if (constraintReleased[index]) return;

    getVCenter(x2) = getVCenter(x1);
    if (!freeRotations)
        getVOrientation(x2) = getVOrientation(x1);
}
#endif

#ifndef SOFA_FLOAT
template<>
inline void AttachConstraint<defaulttype::Rigid3dTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index)
{
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
#endif

#ifndef SOFA_DOUBLE
template<>
inline void AttachConstraint<defaulttype::Rigid3fTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index)
{
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
#endif

#ifndef SOFA_FLOAT
template<>
inline void AttachConstraint<defaulttype::Rigid2dTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index)
{
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
#endif

#ifndef SOFA_DOUBLE
template<>
inline void AttachConstraint<defaulttype::Rigid2fTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool twoway, unsigned index)
{
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
#endif

#ifndef SOFA_FLOAT
template<>
inline unsigned int AttachConstraint<defaulttype::Rigid3dTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::spatial_dimensions;
    else
        return Deriv::total_size;
}
#endif

#ifndef SOFA_DOUBLE
template<>
inline unsigned int AttachConstraint<defaulttype::Rigid3fTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::spatial_dimensions;
    else
        return Deriv::total_size;
}
#endif

#ifndef SOFA_FLOAT
template<>
inline unsigned int AttachConstraint<defaulttype::Rigid2dTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::spatial_dimensions;
    else
        return Deriv::total_size;
}
#endif

#ifndef SOFA_DOUBLE
template<>
inline unsigned int AttachConstraint<defaulttype::Rigid2fTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::spatial_dimensions;
    else
        return Deriv::total_size;
}
#endif

#if 0
// Define TestNewPointFunction
template< class DataTypes>
bool AttachConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    AttachConstraint<DataTypes> *fc= (AttachConstraint<DataTypes> *)param;
    if (fc)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Define RemovalFunction
template< class DataTypes>
void AttachConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    AttachConstraint<DataTypes> *fc= (AttachConstraint<DataTypes> *)param;
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
    return;
}
#endif

template <class DataTypes>
AttachConstraint<DataTypes>::AttachConstraint()
    : f_indices1( initData(&f_indices1,"indices1","Indices of the source points on the first model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the fixed points on the second model") )
    , f_radius( initData(&f_radius,(Real)-1,"radius", "Radius to search corresponding fixed point if no indices are given") )
    , f_twoWay( initData(&f_twoWay,false,"twoWay", "true if forces should be projected back from model2 to model1") )
    , f_freeRotations( initData(&f_freeRotations,false,"freeRotations", "true to keep rotations free (only used for Rigid DOFs)") )
    , f_lastFreeRotation( initData(&f_lastFreeRotation,false,"lastFreeRotation", "true to keep rotation of the last attached point free (only used for Rigid DOFs)") )
    , f_restRotations( initData(&f_restRotations,false,"restRotations", "true to use rest rotations local offsets (only used for Rigid DOFs)") )
    , f_lastPos( initData(&f_lastPos,"lastPos", "position at which the attach constraint should become inactive") )
    , f_lastDir( initData(&f_lastDir,"lastDir", "direction from lastPos at which the attach coustraint should become inactive") )
    , f_clamp( initData(&f_clamp, false,"clamp", "true to clamp particles at lastPos instead of freeing them.") )
    , f_minDistance( initData(&f_minDistance, (Real)-1,"minDistance", "the constraint become inactive if the distance between the points attached is bigger than minDistance.") )
{
    // default to indice 0
//     f_indices1.beginEdit()->push_back(0);
//     f_indices1.endEdit();
//     f_indices2.beginEdit()->push_back(0);
//     f_indices2.endEdit();
}

template <class DataTypes>
AttachConstraint<DataTypes>::AttachConstraint(core::behavior::MechanicalState<DataTypes> *mm1, core::behavior::MechanicalState<DataTypes> *mm2)
    : core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>(mm1,mm2)
    ,f_indices1( initData(&f_indices1,"indices1","Indices of the source points on the first model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the fixed points on the second model") )
    , f_radius( initData(&f_radius,(Real)-1,"radius", "Radius to search corresponding fixed point if no indices are given") )
    , f_twoWay( initData(&f_twoWay,false,"twoWay", "true if forces should be projected back from model2 to model1") )
    , f_freeRotations( initData(&f_freeRotations,false,"freeRotations", "true to keep rotations free (only used for Rigid DOFs)") )
    , f_lastFreeRotation( initData(&f_lastFreeRotation,false,"lastFreeRotation", "true to keep rotation of the last attached point free (only used for Rigid DOFs)") )
    , f_restRotations( initData(&f_restRotations,false,"restRotations", "true to use rest rotations local offsets (only used for Rigid DOFs)") )
    , f_lastPos( initData(&f_lastPos,"lastPos", "position at which the attach constraint should become inactive") )
    , f_lastDir( initData(&f_lastDir,"lastDir", "direction from lastPos at which the attach coustraint should become inactive") )
    , f_clamp( initData(&f_clamp, false,"clamp", "true to clamp particles at lastPos instead of freeing them.") )
    , f_minDistance( initData(&f_minDistance, (Real)-1,"minDistance", "the constraint become inactive if the distance between the points attached is bigger than minDistance.") )
{
    // default to indice 0
//     f_indices1.beginEdit()->push_back(0);
//     f_indices1.endEdit();
//     f_indices2.beginEdit()->push_back(0);
//     f_indices2.endEdit();
}

#if 0
// Handle topological changes
template <class DataTypes> void AttachConstraint<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->endChange();

    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
}
#endif

template <class DataTypes>
AttachConstraint<DataTypes>::~AttachConstraint()
{
}

template <class DataTypes>
void AttachConstraint<DataTypes>::clearConstraints()
{
    f_indices1.beginEdit()->clear();
    f_indices1.endEdit();
    f_indices2.beginEdit()->clear();
    f_indices2.endEdit();
}

template <class DataTypes>
void AttachConstraint<DataTypes>::addConstraint(unsigned int index1, unsigned int index2)
{
    f_indices1.beginEdit()->push_back(index1);
    f_indices1.endEdit();
    f_indices2.beginEdit()->push_back(index2);
    f_indices2.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void AttachConstraint<DataTypes>::init()
{
    this->core::behavior::PairInteractionProjectiveConstraintSet<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    f_indices1.createTopologicalEngine(topology);
    f_indices1.registerTopologicalData();

    f_indices2.createTopologicalEngine(topology);
    f_indices2.registerTopologicalData();

    constraintReleased.resize(f_indices2.getValue().size());

    if (f_radius.getValue() >= 0 && f_indices1.getValue().size()==0 && f_indices2.getValue().size()==0 && this->mstate1 && this->mstate2)
    {
        const Real maxR = f_radius.getValue();
        const VecCoord& x1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord& x2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();
        for (unsigned int i2=0; i2<x2.size(); ++i2)
        {
            int best = -1;
            Real bestR = maxR;
            for (unsigned int i1=0; i1<x1.size(); ++i1)
            {
                Real r = (x2[i2]-x1[i1]).norm();
                if (r <= bestR)
                {
                    best = i1;
                    bestR = r;
                }
            }
            if (best >= 0)
            {
                addConstraint(best, i2);
            }
        }
    }
#if 0
    // Initialize functions and parameters
    topology::PointSubset my_subset = f_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );
#endif
    activeFlags.resize(f_indices2.getValue().size());
    std::fill(activeFlags.begin(), activeFlags.end(), true);
    if (f_restRotations.getValue())
        calcRestRotations();
}


template <class DataTypes>
void AttachConstraint<DataTypes>::calcRestRotations()
{
}

#ifndef SOFA_FLOAT
template <>
void AttachConstraint<sofa::defaulttype::Rigid3dTypes>::calcRestRotations();
#endif

template <class DataTypes>
void AttachConstraint<DataTypes>::projectPosition(const core::MechanicalParams * /*mparams*/, DataVecCoord& res1_d, DataVecCoord& res2_d)
{
    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const bool freeRotations = f_freeRotations.getValue();
    const bool lastFreeRotation = f_lastFreeRotation.getValue();
    const bool last = (f_lastDir.isSet() && f_lastDir.getValue().norm() > 1.0e-10);
    const bool clamp = f_clamp.getValue();

    VecCoord &res1 = *res1_d.beginEdit();
    VecCoord &res2 = *res2_d.beginEdit();

    // update active flags
    activeFlags.resize(indices2.size());
    if (last)
        lastDist.resize(indices2.size());

    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;
        if (last)
        {
            Coord p = res1[indices1[i]];
            sofa::defaulttype::Vec<3,Real> p3d;
            DataTypes::get(p3d[0],p3d[1],p3d[2],p);
            lastDist[i] = (Real)( (p3d-f_lastPos.getValue())*f_lastDir.getValue());
            if (lastDist[i] > 0.0)
            {
                if (clamp)
                {
                    msg_info_when(activeFlags[i]) << "AttachConstraint: point "
                                                  <<indices1[i]<<" stopped." ;
                }
                else
                {
                    msg_info_when(activeFlags[i]) << "AttachConstraint: point "
                                                  <<indices1[i]<<" is free.";
                }
                active = false;
            }
        }
        activeFlags[i] = active;
    }
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        Coord p = res1[indices1[i]];
        if (activeFlags[i])
        {
            msg_info() << "AttachConstraint: x2["<<indices2[i]<<"] = x1["<<indices1[i]<<"]";

            projectPosition(p, res2[indices2[i]], freeRotations || (lastFreeRotation && (i>=activeFlags.size() || !activeFlags[i+1])), i);
        }
        else if (clamp)
        {
            DataTypes::set(p,f_lastPos.getValue()[0],f_lastPos.getValue()[1],f_lastPos.getValue()[2]);

            msg_info() << "AttachConstraint: x2["<<indices2[i]<<"] = lastPos";

            projectPosition(p, res2[indices2[i]], freeRotations, i);
        }
    }

    res1_d.endEdit();
    res2_d.endEdit();
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectVelocity(const core::MechanicalParams * /*mparams*/, DataVecDeriv& res1_d, DataVecDeriv& res2_d)
{
    VecDeriv &res1 = *res1_d.beginEdit();
    VecDeriv &res2 = *res2_d.beginEdit();

    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const bool freeRotations = f_freeRotations.getValue();
    const bool lastFreeRotation = f_lastFreeRotation.getValue();
    const bool clamp = f_clamp.getValue();

    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;

        if (i < activeFlags.size())
            active = activeFlags[i];

        if (active)
        {
            msg_info() << "AttachConstraint: v2["<<indices2[i]<<"] = v1["<<indices1[i]<<"]" ;

            projectVelocity(res1[indices1[i]], res2[indices2[i]], freeRotations || (lastFreeRotation && (i>=activeFlags.size() || !activeFlags[i+1])), i);
        }
        else if (clamp)
        {
            msg_info() << "AttachConstraint: v2["<<indices2[i]<<"] = 0" ;

            Deriv v = Deriv();
            projectVelocity(v, res2[indices2[i]], freeRotations, i);
        }
    }

    res1_d.endEdit();
    res2_d.endEdit();
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectResponse(const core::MechanicalParams * /*mparams*/, DataVecDeriv& res1_d, DataVecDeriv& res2_d)
{
    VecDeriv &res1 = *res1_d.beginEdit();
    VecDeriv &res2 = *res2_d.beginEdit();

    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const bool twoway = f_twoWay.getValue();
    const bool freeRotations = f_freeRotations.getValue();
    const bool lastFreeRotation = f_lastFreeRotation.getValue();
    const bool clamp = f_clamp.getValue();

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

            projectResponse(res1[indices1[i]], res2[indices2[i]], freeRotations || (lastFreeRotation && (i>=activeFlags.size() || !activeFlags[i+1])), twoway, i);

            msg_info() << " final r2["<<indices2[i]<<"] = "<<res2[indices2[i]]<<"";
        }
        else if (clamp)
        {
            msg_info() << " r2["<<indices2[i]<<"] = 0";

            Deriv v = Deriv();
            projectResponse(v, res2[indices2[i]], freeRotations, false, i);

            msg_info() << " final r2["<<indices2[i]<<"] = "<<res2[indices2[i]]<<"";
        }
    }

    res1_d.endEdit();
    res2_d.endEdit();
}


// Matrix Integration interface
template <class DataTypes>
void AttachConstraint<DataTypes>::applyConstraint(const core::MechanicalParams * /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (f_twoWay.getValue())
        return;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate2);
    if (!r)
        return;

    sofa::defaulttype::BaseMatrix *mat = r.matrix;
    unsigned int offset = r.offset;

    const SetIndexArray & indices = f_indices2.getValue();
    const unsigned int N = Deriv::size();
    const unsigned int NC = DerivConstrainedSize(f_freeRotations.getValue());
    const unsigned int NCLast = DerivConstrainedSize(f_lastFreeRotation.getValue());
    unsigned int i=0;
    const bool clamp = f_clamp.getValue();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it, ++i)
    {
        if (!clamp && i < activeFlags.size() && !activeFlags[i])
            continue;

        msg_info() << "AttachConstraint: apply in matrix column/row "<<(*it);

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
void AttachConstraint<DataTypes>::applyConstraint(const core::MechanicalParams * /*mparams*/, defaulttype::BaseVector* vect, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (f_twoWay.getValue())
        return;

    int o = matrix->getGlobalOffset(this->mstate2);
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
void AttachConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const SetIndexArray & indices1 = f_indices1.getValue();
    const SetIndexArray & indices2 = f_indices2.getValue();
    const VecCoord& x1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        if (activeFlags.size() > i && !activeFlags[i]) continue;
        sofa::helper::gl::glVertexT(x2[indices2[i]]);
    }
    glEnd();
    glPointSize(1);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        if (activeFlags.size() > i && !activeFlags[i]) continue;
        sofa::helper::gl::glVertexT(x1[indices1[i]]);
        sofa::helper::gl::glVertexT(x2[indices2[i]]);
    }
    glEnd();
#endif /* SOFA_NO_OPENGL */
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
