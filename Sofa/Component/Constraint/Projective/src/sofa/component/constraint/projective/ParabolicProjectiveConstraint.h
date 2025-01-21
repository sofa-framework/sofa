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

/** Apply a parabolic trajectory to particles going through 3 points specified by the user.
	The DOFs set in the "indices" list follow the computed parabol from "tBegin" to "tEnd".
	*/
template <class DataTypes>
class ParabolicProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParabolicProjectiveConstraint,DataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,DataTypes));

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
    typedef type::vector<sofa::Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;
    typedef type::Vec<3, Real> Vec3R;
    typedef type::Quat<Real> QuatR;

protected:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData< sofa::type::vector<sofa::Index> > m_indices;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec3R> m_P1;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec3R> m_P2;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Vec3R> m_P3;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_tBegin;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_PROJECTIVE()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_tEnd;

    ///indices of the DOFs constraints
    SetIndex d_indices;

    /// the three points defining the parabol
    Data<Vec3R> d_P1;
    Data<Vec3R> d_P2; ///< second point of the parabol
    Data<Vec3R> d_P3; ///< third point of the parabol

    /// the time steps defining the velocity of the movement
    Data<Real> d_tBegin;
    Data<Real> d_tEnd; ///< End Time of the motion

    /// Link to be set to the topology container in the component graph.
    SingleLink<ParabolicProjectiveConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    /// the 3 points projected in the parabol plan
    Vec3R m_locP1;
    Vec3R m_locP2;
    Vec3R m_locP3;

    /// the quaternion doing the projection
    QuatR m_projection;

    explicit ParabolicProjectiveConstraint(core::behavior::MechanicalState<DataTypes>* mstate = nullptr);

    ~ParabolicProjectiveConstraint();
public:
    void addConstraint(unsigned index );

    void setP1(const Vec3R &p) {d_P1.setValue(p);}
    void setP2(const Vec3R &p) {d_P2.setValue(p);}
    void setP3(const Vec3R &p) {d_P3.setValue(p);}

    void setBeginTime(const Real &t) {d_tBegin.setValue(t);}
    void setEndTime(const Real &t) {d_tEnd.setValue(t);}

    Vec3R getP1() {return d_P1.getValue();}
    Vec3R getP2() {return d_P2.getValue();}
    Vec3R getP3() {return d_P3.getValue();}

    Real getBeginTime() {return d_tBegin.getValue();}
    Real getEndTime() {return d_tEnd.getValue();}

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


#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARABOLICPROJECTIVECONSTRAINT_CPP)
extern template class ParabolicProjectiveConstraint<defaulttype::Rigid3Types>;
extern template class ParabolicProjectiveConstraint<defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::constraint::projective
