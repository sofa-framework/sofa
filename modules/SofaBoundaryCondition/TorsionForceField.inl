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
#include <SofaBoundaryCondition/TorsionForceField.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using sofa::component::linearsolver::CompressedRowSparseMatrix;

template<typename DataTypes>
TorsionForceField<DataTypes>::TorsionForceField() :
	m_indices(initData(&m_indices, "indices", "indices of the selected points")),
	m_torque(initData(&m_torque, "torque", "torque to apply")),
	m_axis(initData(&m_axis, "axis", "direction of the axis (will be normalized)")),
	m_origin(initData(&m_origin, "origin", "origin of the axis"))
{
}

template<typename DataTypes>
TorsionForceField<DataTypes>::~TorsionForceField()
{

}

template<typename DataTypes>
void TorsionForceField<DataTypes>::bwdInit()
{
	m_u = m_axis.getValue();
	m_u.normalize();
}

template<typename DataTypes>
void TorsionForceField<DataTypes>::addForce(const MechanicalParams* /*params*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /*v*/)
{
	const VecId& indices = m_indices.getValue();
	const VecCoord& q = x.getValue();
	const Real& tau = m_torque.getValue();
	const Pos& o = m_origin.getValue();
	VecDeriv& fq = *f.beginEdit();

	const std::size_t nNodes = indices.size();

	for(size_t n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		const Pos lever = tau*m_u;
		fq[id] += lever.cross(q[id] - (o + (q[id] * m_u)*m_u) );
	}

	f.endEdit();
}

template<typename DataTypes>
void TorsionForceField<DataTypes>::addDForce(const MechanicalParams* params, DataVecDeriv& df, const DataVecDeriv& dx)
{
	const VecId& indices = m_indices.getValue();
	const VecDeriv& dq = dx.getValue();
	const Real& tau = m_torque.getValue();
	VecDeriv& dfq = *df.beginEdit();

	const std::size_t nNodes = indices.size();
	const Real& kfact = params->kFactor();

	Mat3 D;
	D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
	D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
	D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
	D *= (tau * kfact);

	for(size_t n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		dfq[id] += D * dq[id];
	}
}

template<typename DataTypes>
void TorsionForceField<DataTypes>::addKToMatrix(defaulttype::BaseMatrix* matrix, double kFact, unsigned int& offset)
{
	const VecId& indices = m_indices.getValue();
	const Real& tau = m_torque.getValue();

	const std::size_t nNodes = indices.size();

	MatrixBlock D;
	D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
	D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
	D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
	D *= (tau * kFact);

	if( CompressedRowSparseMatrix<MatrixBlock>* m = dynamic_cast<CompressedRowSparseMatrix<MatrixBlock>*>(matrix) )
	{

		for(size_t n = 0 ; n < nNodes ; ++n)
		{
			PointId id = indices[n];
			*m->wbloc(id, id, true) += D;
		}
	}
	else
	{
		for(size_t n = 0 ; n < nNodes ; ++n)
		{
			PointId id = indices[n];
			unsigned int c = offset + Deriv::total_size*id;

			matrix->add(c+0, c+0, D(0,0));
			matrix->add(c+0, c+1, D(0,1));
			matrix->add(c+0, c+2, D(0,2));

			matrix->add(c+1, c+0, D(1,0));
			matrix->add(c+1, c+1, D(1,1));
			matrix->add(c+1, c+2, D(1,2));

			matrix->add(c+2, c+0, D(2,0));
			matrix->add(c+2, c+1, D(2,1));
			matrix->add(c+2, c+2, D(2,2));
		}
	}

}

#ifndef SOFA_DOUBLE
template<>
void TorsionForceField<Rigid3fTypes>::addForce(const core::MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &/*v*/)
{
	const VecId& indices = m_indices.getValue();
	const VecCoord& q = x.getValue();
	const Real& tau = m_torque.getValue();
	const Pos& o = m_origin.getValue();
	VecDeriv& fq = *f.beginEdit();

	const std::size_t nNodes = indices.size();

	for(size_t n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		const Pos t = tau*m_u;
		fq[id].getVCenter() += t.cross(q[id].getCenter() - (o + (q[id].getCenter() * m_u)*m_u) );
		fq[id].getVOrientation() += t;
	}

	f.endEdit();
}


template<>
void TorsionForceField<Rigid3fTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx)
{
	const VecId& indices = m_indices.getValue();
	const VecDeriv& dq = dx.getValue();
	const Real& tau = m_torque.getValue();
	VecDeriv& dfq = *df.beginEdit();

	const std::size_t nNodes = indices.size();
	const Real& kfact = mparams->kFactor();

	Mat3 D;
	D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
	D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
	D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
	D *= (tau * kfact);

	for(size_t n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		dfq[id].getVCenter() += D * dq[id].getVCenter();
	}
}

#endif

#ifndef SOFA_FLOAT
template<>
void TorsionForceField<Rigid3dTypes>::addForce(const core::MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &/*v*/)
{
	const VecId& indices = m_indices.getValue();
	const VecCoord& q = x.getValue();
	const Real& tau = m_torque.getValue();
	const Pos& o = m_origin.getValue();
	VecDeriv& fq = *f.beginEdit();

	const std::size_t nNodes = indices.size();

	for(size_t n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		const Pos t = tau*m_u;
		fq[id].getVCenter() += t.cross(q[id].getCenter() - (o + (q[id].getCenter() * m_u)*m_u) );
		fq[id].getVOrientation() += t;
	}

	f.endEdit();
}

template<>
void TorsionForceField<Rigid3dTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx)
{
	const VecId& indices = m_indices.getValue();
	const VecDeriv& dq = dx.getValue();
	const Real& tau = m_torque.getValue();
	VecDeriv& dfq = *df.beginEdit();

	const std::size_t nNodes = indices.size();
	const Real& kfact = mparams->kFactor();

	Mat3 D;
	D(0,0) = 1 - m_u(0)*m_u(0) ;	D(0,1) = -m_u(1)*m_u(0) ;		D(0,2) = -m_u(2)*m_u(0);
	D(1,0) = -m_u(0)*m_u(1) ;		D(1,1) = 1 - m_u(1)*m_u(1) ;	D(1,2) = -m_u(2)*m_u(1);
	D(2,0) = -m_u(0)*m_u(2) ;		D(2,1) = -m_u(1)*m_u(2) ;		D(2,2) = 1 - m_u(3)*m_u(3);
	D *= (tau * kfact);

	for(size_t n = 0 ; n < nNodes ; ++n)
	{
		PointId id = indices[n];
		dfq[id].getVCenter() += D * dq[id].getVCenter();
	}
}

#endif



} // namespace forcefield
} // namespace component
} // namespace sofa
