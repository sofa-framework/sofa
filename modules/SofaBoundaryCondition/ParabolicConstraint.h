/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARABOLICCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARABOLICCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologySubsetData.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/** Apply a parabolic trajectory to particles going through 3 points specified by the user.
	The DOFs set in the "indices" list follow the computed parabol from "tBegin" to "tEnd".
	*/
template <class DataTypes>
class ParabolicConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParabolicConstraint,DataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,DataTypes));

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
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
    typedef defaulttype::Vec<3, Real> Vec3R;
    typedef helper::Quater<Real> QuatR;

protected:
    ///indices of the DOFs constraints
    SetIndex m_indices;

    /// the three points defining the parabol
    Data<Vec3R> m_P1;
    Data<Vec3R> m_P2;
    Data<Vec3R> m_P3;

    /// the time steps defining the velocity of the movement
    Data<Real> m_tBegin;
    Data<Real> m_tEnd;

    /// the 3 points projected in the parabol plan
    Vec3R m_locP1;
    Vec3R m_locP2;
    Vec3R m_locP3;

    /// the quaternion doing the projection
    QuatR m_projection;




    ParabolicConstraint();

    ParabolicConstraint(core::behavior::MechanicalState<DataTypes>* mstate);

    ~ParabolicConstraint();
public:
    void addConstraint(unsigned index );

    void setP1(const Vec3R &p) {m_P1.setValue(p);}
    void setP2(const Vec3R &p) {m_P2.setValue(p);}
    void setP3(const Vec3R &p) {m_P3.setValue(p);}

    void setBeginTime(const Real &t) {m_tBegin.setValue(t);}
    void setEndTime(const Real &t) {m_tEnd.setValue(t);}

    Vec3R getP1() {return m_P1.getValue();}
    Vec3R getP2() {return m_P2.getValue();}
    Vec3R getP3() {return m_P3.getValue();}

    Real getBeginTime() {return m_tBegin.getValue();}
    Real getEndTime() {return m_tEnd.getValue();}

    /// -- Constraint interface
    void init() override;
    void reinit() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARABOLICCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class ParabolicConstraint<defaulttype::Rigid3dTypes>;
extern template class ParabolicConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class ParabolicConstraint<defaulttype::Rigid3fTypes>;
extern template class ParabolicConstraint<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
