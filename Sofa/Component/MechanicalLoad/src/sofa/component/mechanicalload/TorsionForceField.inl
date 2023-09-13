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

#include <sofa/component/mechanicalload/TorsionForceField.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>


namespace sofa::component::mechanicalload
{

using sofa::defaulttype::Rigid3Types;
using sofa::linearalgebra::CompressedRowSparseMatrix;

template<typename DataTypes>
TorsionForceField<DataTypes>::TorsionForceField() :
	m_indices(initData(&m_indices, "indices", "indices of the selected points")),
	m_torque(initData(&m_torque, "torque", "torque to apply")),
	m_axis(initData(&m_axis, "axis", "direction of the axis (will be normalized)")),
	m_origin(initData(&m_origin, "origin", "origin of the axis"))
{
    /// Update the normalized axis from m_axis
    this->addUpdateCallback("updateNormalAxis", {&m_axis}, [this](const core::DataTracker& )
    {
        m_u = m_axis.getValue();
        m_u.normalize();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {&m_indices});
}

template<typename DataTypes>
TorsionForceField<DataTypes>::~TorsionForceField()
{

}

template<typename DataTypes>
void TorsionForceField<DataTypes>::addForce(const MechanicalParams* /*params*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /*v*/)
{
	const VecId& indices = m_indices.getValue();
	const VecCoord& q = x.getValue();
	const Real& tau = m_torque.getValue();
	const Pos& o = m_origin.getValue();
	VecDeriv& fq = *f.beginEdit();

	const auto nNodes = indices.size();

	for(Size n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		const Pos lever = tau*m_u;
		fq[id] += lever.cross(q[id] - (o + (q[id] * m_u)*m_u) );
	}

	f.endEdit();
}

template<typename DataTypes>
void TorsionForceField<DataTypes>::addDForce(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
	const VecId& indices = m_indices.getValue();
	const VecDeriv& dq = dx.getValue();
	const Real& tau = m_torque.getValue();
	VecDeriv& dfq = *df.beginEdit();

	const auto nNodes = indices.size();
	const Real& kfact = mparams->kFactor();

	Mat3 D;
	D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
	D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
	D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
	D *= (tau * kfact);

	for(Size n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		dfq[id] += D * dq[id];
	}
}

template<typename DataTypes>
void TorsionForceField<DataTypes>::addKToMatrix(linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset)
{
	const VecId& indices = m_indices.getValue();
	const Real& tau = m_torque.getValue();

	sofa::type::MatNoInit<3,3, Real> D;
	D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
	D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
	D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
	D *= (tau * kFact);

	for (const auto id : indices)
	{
		const unsigned int c = offset + Deriv::total_size * id;
		matrix->add(c, c, D);
	}
}

template <typename DataTypes>
void TorsionForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    const VecId& indices = m_indices.getValue();
    const Real& tau = m_torque.getValue();

    sofa::type::MatNoInit<3,3, Real> D;
    D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
    D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
    D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
    D *= tau;

    for (const auto id : indices)
    {
        const unsigned int c = Deriv::total_size * id;
        dfdx(c, c) += D;
    }
}

template <typename DataTypes>
void TorsionForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<typename DataTypes>
SReal TorsionForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

template<>
void TorsionForceField<Rigid3Types>::addForce(const core::MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &/*v*/)
{
	const VecId& indices = m_indices.getValue();
	const VecCoord& q = x.getValue();
	const Real& tau = m_torque.getValue();
	const Pos& o = m_origin.getValue();
	VecDeriv& fq = *f.beginEdit();

	const auto nNodes = indices.size();

	for(Size n = 0 ; n < nNodes ; ++n)
	{
		const PointId id = indices[n];
		const Pos t = tau*m_u;
		fq[id].getVCenter() += t.cross(q[id].getCenter() - (o + (q[id].getCenter() * m_u)*m_u) );
		fq[id].getVOrientation() += t;
	}

	f.endEdit();
}

template<>
void TorsionForceField<Rigid3Types>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx)
{
	const VecId& indices = m_indices.getValue();
	const VecDeriv& dq = dx.getValue();
	const Real& tau = m_torque.getValue();
	VecDeriv& dfq = *df.beginEdit();

	const auto nNodes = indices.size();
	const Real& kfact = mparams->kFactor();

	Mat3 D;
	D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
	D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
	D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
	D *= (tau * kfact);

	for(Size n = 0 ; n < nNodes ; ++n)
	{
		const PointId id = indices[n];
		dfq[id].getVCenter() += D * dq[id].getVCenter();
	}
}

} // namespace sofa::component::mechanicalload
