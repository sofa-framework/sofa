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
#include <sofa/type/vector.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::constraint::projective
{

/**
	Impose a trajectory to given Dofs following a Hermite cubic spline constraint.
	Control parameters are :
	  - begin and end points
	  - derivates at this points
	  - acceleration curve on the trajectory
	*/
template <class DataTypes>
class HermiteSplineProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HermiteSplineProjectiveConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

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
    typedef type::vector<unsigned int> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;
    typedef typename type::Vec<3, Real> Vec3R;
    typedef typename type::Vec<2, Real> Vec2R;
    typedef typename type::Quat<Real> QuatR;

public:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData< sofa::type::vector<sofa::Index> > m_indices;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_tBegin;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_tEnd;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec3R>  m_x0;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec3R>  m_dx0;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec3R> m_x1;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec3R>  m_dx1;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec2R> m_sx0;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec2R> m_sx1;

    ///indices of the DOFs constraints
    SetIndex d_indices;

    /// the time steps defining the duration of the constraint
    Data<Real> d_tBegin;
    Data<Real> d_tEnd; ///< End Time of the motion

    /// control parameters :
    /// first control point
    Data<Vec3R> d_x0;
    /// first derivated control point
    Data<Vec3R> d_dx0;
    /// second control point
    Data<Vec3R> d_x1;
    /// second derivated control point
    Data<Vec3R> d_dx1;
    /// acceleration parameters : the accaleration curve is itself a hermite spline, with first point at (0,0) and second at (1,1)
    /// and derivated on this points are :
    Data<Vec2R> d_sx0;
    Data<Vec2R> d_sx1; ///< second interpolation vector

    /// Link to be set to the topology container in the component graph.
    SingleLink<HermiteSplineProjectiveConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
    
protected:
    explicit HermiteSplineProjectiveConstraint(core::behavior::MechanicalState<DataTypes>* mstate = nullptr);

    ~HermiteSplineProjectiveConstraint();
public:
    void clearConstraints();
    void addConstraint(unsigned index );

    void setBeginTime(const Real &t) {d_tBegin.setValue(t);}
    void setEndTime(const Real &t) {d_tEnd.setValue(t);}

    Real getBeginTime() {return d_tBegin.getValue();}
    Real getEndTime() {return d_tEnd.getValue();}

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
    void projectResponseT(DataDeriv& dx,
        const std::function<void(DataDeriv&, const unsigned int)>& clear);

};

template <>
void SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API HermiteSplineProjectiveConstraint<defaulttype::Rigid3Types>::init();


#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_HERMITESPLINEPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API HermiteSplineProjectiveConstraint<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API HermiteSplineProjectiveConstraint<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::constraint::projective
