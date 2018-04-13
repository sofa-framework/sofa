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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_HERMITESPLINECONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_HERMITESPLINECONSTRAINT_H
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

/**
	Impose a trajectory to given Dofs following a Hermite cubic spline constraint.
	Control parameters are :
	  - begin and end points
	  - derivates at this points
	  - acceleration curve on the trajectory
	*/
template <class DataTypes>
class HermiteSplineConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HermiteSplineConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
    typedef typename defaulttype::Vec<3, Real> Vec3R;
    typedef typename defaulttype::Vec<2, Real> Vec2R;
    typedef typename helper::Quater<Real> QuatR;

public:
    ///indices of the DOFs constraints
    SetIndex m_indices;

    /// the time steps defining the duration of the constraint
    Data<Real> m_tBegin;
    Data<Real> m_tEnd; ///< End Time of the motion

    /// control parameters :
    /// first control point
    Data<Vec3R> m_x0;
    /// first derivated control point
    Data<Vec3R> m_dx0;
    /// second control point
    Data<Vec3R> m_x1;
    /// second derivated control point
    Data<Vec3R> m_dx1;
    /// acceleration parameters : the accaleration curve is itself a hermite spline, with first point at (0,0) and second at (1,1)
    /// and derivated on this points are :
    Data<Vec2R> m_sx0;
    Data<Vec2R> m_sx1; ///< second interpolation vector



protected:
    HermiteSplineConstraint();

    HermiteSplineConstraint(core::behavior::MechanicalState<DataTypes>* mstate);

    ~HermiteSplineConstraint();
public:
    void clearConstraints();
    void addConstraint(unsigned index );

    void setBeginTime(const Real &t) {m_tBegin.setValue(t);}
    void setEndTime(const Real &t) {m_tEnd.setValue(t);}

    Real getBeginTime() {return m_tBegin.getValue();}
    Real getEndTime() {return m_tEnd.getValue();}

    void computeHermiteCoefs( const Real u, Real &H00, Real &H10, Real &H01, Real &H11);
    void computeDerivateHermiteCoefs( const Real u, Real &dH00, Real &dH10, Real &dH01, Real &dH11);

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_HERMITESPLINECONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class HermiteSplineConstraint<defaulttype::Rigid3dTypes>;
extern template class HermiteSplineConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class HermiteSplineConstraint<defaulttype::Rigid3fTypes>;
extern template class HermiteSplineConstraint<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
