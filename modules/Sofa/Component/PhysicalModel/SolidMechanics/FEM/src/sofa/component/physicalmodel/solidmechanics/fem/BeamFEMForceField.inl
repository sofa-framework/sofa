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
#ifndef SOFA_COMPONENT_PHYSICALMODEL_SOLIDMECHANICS_FEM_BEAMFEMFORCEFIELD_INL
#define SOFA_COMPONENT_PHYSICALMODEL_SOLIDMECHANICS_FEM_BEAMFEMFORCEFIELD_INL

#include <sofa/component/physicalmodel/solidmechanics/fem/BeamFEMForceField.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>
#include <set>
#include <array>
#include <sofa/helper/system/gl.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace physicalmodel
{

namespace solidmechanics
{

namespace fem
{


template<class DataTypes>
BeamFEMForceField<DataTypes>::BeamFEMForceField()
	: _useSymmetricAssembly(initData(&_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
	, d_youngModulus(initData(&d_youngModulus, "youngModulus","Young Modulus"))
	, d_poissonRatio(initData(&d_poissonRatio,"poissonRatio","Poisson Ratio"))
	, d_r(initData(&d_r,"radius","radius of the sections"))
	, d_innerR(initData(&d_innerR,"innerRadius","inner radius of the sections for hollow beams"))
	, d_L(initData(&d_L,"length","length of the sections"))
	, _epsilon(0.0000001)
{
	d_poissonRatio.setRequired(true);
	d_youngModulus.setReadOnly(true);
}

template<class DataTypes>
BeamFEMForceField<DataTypes>::~BeamFEMForceField()
{ }

template <class DataTypes>
void BeamFEMForceField<DataTypes>::init()
{
	this->core::behavior::ForceField<DataTypes>::init();
	sofa::core::objectmodel::BaseContext* context = this->getContext();

	_topology = context->getMeshTopology();

	if (_topology==nullptr)
	{
		msg_error(this) << "The object must have a BaseMeshTopology (i.e. EdgeSetTopology or MeshTopology).";
		return;
	}
	else if(_topology->getNbEdges()==0)
	{
		msg_error(this) << "The topology is empty.";
		return;
	}

	initInternalData();
}

template <class DataTypes>
void BeamFEMForceField<DataTypes>::bwdInit()
{ }

template <class DataTypes>
void BeamFEMForceField<DataTypes>::reinit()
{
	std::size_t size = _topology->getEdges().size();

	for (std::size_t i = 0; i<size; ++i)
		reinitBeam(i);
}

template <class DataTypes>
void BeamFEMForceField<DataTypes>::initInternalData()
{
	std::size_t size = _topology->getEdges().size();

	_G.resize(size);
	_Iy.resize(size);
	_Iz.resize(size);
	_J.resize(size);
	_A.resize(size);
	_Asy.resize(size);
	_Asz.resize(size);
	_k_loc.resize(size);
	_quat.resize(size);

    sofa::helper::ReadAccessor< Data<VecReal> > radii = d_r;
    sofa::helper::ReadAccessor< Data<VecReal> > innerRadii = d_innerR;

    if(radii.size() == 1)
    {
        Real r = radii[0];
        sofa::helper::WriteAccessor< Data<VecReal> > waRadii = d_r;
        waRadii.resize(size);
        for (size_t i = 0; i < size; i++)
            waRadii[i] = r;
    }
    else
        if (radii.size() != size)
            msg_error() << "number of radii and beams differs";

    if(innerRadii.size() == 1)
    {
        Real ir = innerRadii[0];
        sofa::helper::WriteAccessor< Data<VecReal> > waInRadii = d_innerR;
        waInRadii.resize(size);
        for (size_t i = 0; i < size; i++)
            waInRadii[i] = ir;
    }
    else
        if (innerRadii.size() != n)
            msg_error() << "number of inner radii and beams differs";

	for (std::size_t i = 0; i<size; ++i)
		reinitBeam(i);

	msg_info() << "reinit OK, "<< size <<" elements." ;
}

template <class DataTypes>
void BeamFEMForceField<DataTypes>::reinitBeam(std::size_t i)
{
	Index a = _topology->getEdges()[i][0];
	Index b = _topology->getEdges()[i][1];

	//const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
	//length = (x0[a].getCenter()-x0[b].getCenter()).norm() ;

	const Real E = d_youngModulus.getValue()[i];
	const Real nu = d_poissonRatio.getValue()[i];
	const Real r = d_r.getValue()[i];
	const Real rInner = d_innerR.getValue()[i];

	_G[i] = E/(2.0*(1.0+ nu));
	_Iz[i] = M_PI*(r*r*r*r - rInner*rInner*rInner*rInner)/4.0;

	_Iy[i] = _Iz[i] ;
	_J[i] = _Iz[i] + _Iy[i];
	_A[i] = M_PI*(r*r - rInner*rInner);

	_Asy[i] = 0.0;
	_Asz[i] = 0.0;

	computeStiffness(i);

	initLarge(i,a,b);
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeStiffness(std::size_t i)
{
	Real phiy, phiz;
	Real L = d_L.getValue()[i];
	Real A = _A[i];
	Real nu = d_poissonRatio.getValue()[i];
	Real E = d_youngModulus.getValue()[i];
	Real Iy = _Iy[i];
	Real Iz = _Iz[i];
	Real Asy = _Asy[i];
	Real Asz = _Asz[i];
	Real G = _G[i];
	Real J = _J[i];
	Real L2 = (L * L);
	Real L3 = (L2 * L);
	Real EIy = (E * Iy);
	Real EIz = (E * Iz);

	// Find shear-deformation parameters
	if (Asy == 0)
		phiy = 0.0;
	else
		phiy = (24.0*(1.0+nu)*Iz/(Asy*L2));

	if (Asz == 0)
		phiz = 0.0;
	else
		phiz = (24.0*(1.0+nu)*Iy/(Asz*L2));

	StiffnessMatrix& k_loc = _k_loc[i];

	// Define stiffness matrix 'k' in local coordinates
	k_loc.clear();
	k_loc[6][6]   = k_loc[0][0]   = E*A/L;
	k_loc[7][7]   = k_loc[1][1]   = (Real)(12.0*EIz/(L3*(1.0+phiy)));
	k_loc[8][8]   = k_loc[2][2]   = (Real)(12.0*EIy/(L3*(1.0+phiz)));
	k_loc[9][9]   = k_loc[3][3]   = G*J/L;
	k_loc[10][10] = k_loc[4][4]   = (Real)((4.0+phiz)*EIy/(L*(1.0+phiz)));
	k_loc[11][11] = k_loc[5][5]   = (Real)((4.0+phiy)*EIz/(L*(1.0+phiy)));

	k_loc[4][2]   = (Real)(-6.0*EIy/(L2*(1.0+phiz)));
	k_loc[5][1]   = (Real)( 6.0*EIz/(L2*(1.0+phiy)));
	k_loc[6][0]   = -k_loc[0][0];
	k_loc[7][1]   = -k_loc[1][1];
	k_loc[7][5]   = -k_loc[5][1];
	k_loc[8][2]   = -k_loc[2][2];
	k_loc[8][4]   = -k_loc[4][2];
	k_loc[9][3]   = -k_loc[3][3];
	k_loc[10][2]  = k_loc[4][2];
	k_loc[10][4]  = (Real)((2.0-phiz)*EIy/(L*(1.0+phiz)));
	k_loc[10][8]  = -k_loc[4][2];
	k_loc[11][1]  = k_loc[5][1];
	k_loc[11][5]  = (Real)((2.0-phiy)*EIz/(L*(1.0+phiy)));
	k_loc[11][7]  = -k_loc[5][1];

	for (int i=0; i<=10; i++)
		for (int j=i+1; j<12; j++)
			k_loc[i][j] = k_loc[j][i];
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & /*dataV*/ )
{
	VecDeriv& f = *(dataF.beginEdit());
	const VecCoord& p = dataX.getValue();
	f.resize(p.size());

	std::size_t i = 0;
	for(const Edge& e : _topology->getEdges())
	{
		initLarge(i,e[0],e[1]);
		accumulateForceLarge(f, p, i, e[0], e[1]);
		++i;
	}

	dataF.endEdit();
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams *mparams, DataVecDeriv& datadF , const DataVecDeriv& datadX)
{
	VecDeriv& df = *(datadF.beginEdit());
	const VecDeriv& dx=datadX.getValue();
	Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

	df.resize(dx.size());

	std::size_t i = 0;
	for(const Edge& e : _topology->getEdges())
		applyStiffnessLarge(df, dx, i, e[0], e[1], kFactor);

	datadF.endEdit();
}

////////////// large displacements method
template<class DataTypes>
void BeamFEMForceField<DataTypes>::initLarge(std::size_t i, Index a, Index b)
{
	const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

	Quat quatA, quatB, dQ;
	Vec3 dW;

	quatA = x[a].getOrientation();
	quatB = x[b].getOrientation();

	quatA.normalize();
	quatB.normalize();

	dQ = dQ.quatDiff(quatB, quatA);
	dQ.normalize();

	dW = dQ.quatToRotationVector();     // Use of quatToRotationVector instead of toEulerVector:
	// this is done to keep the old behavior (before the
	// correction of the toEulerVector  function). If the
	// purpose was to obtain the Eulerian vector and not the
	// rotation vector please use the following line instead
	//    dW = dQ.toEulerVector();

	Real Theta = dW.norm();

	if(Theta>_epsilon)
	{
		dW.normalize();

		_quat[i] = quatA*dQ.axisToQuat(dW, Theta/2);
		_quat[i].normalize();
	}
	else
		_quat[i] = quatA;
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::accumulateForceLarge(VecDeriv& f, const VecCoord & x, std::size_t i, Index a, Index b )
{
	const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

	_quat[i] = x[a].getOrientation();
	_quat[i].normalize();

	Vec3 u, P1P2, P1P2_0;
	// local displacement
	Displacement depl;

	// translations //
	P1P2_0 = x0[b].getCenter() - x0[a].getCenter();
	P1P2_0 = x0[a].getOrientation().inverseRotate(P1P2_0);
	P1P2 = x[b].getCenter() - x[a].getCenter();
	P1P2 = x[a].getOrientation().inverseRotate(P1P2);
	u = P1P2 - P1P2_0;

	depl[0] = 0.0; 	depl[1] = 0.0; 	depl[2] = 0.0;
	depl[6] = u[0]; depl[7] = u[1]; depl[8] = u[2];

	// rotations //
	Quat dQ0, dQ;

	// dQ = QA.i * QB ou dQ = QB * QA.i() ??
	dQ0 = dQ0.quatDiff(x0[b].getOrientation(), x0[a].getOrientation()); // x0[a].getOrientation().inverse() * x0[b].getOrientation();
	dQ =  dQ.quatDiff(x[b].getOrientation(), x[a].getOrientation()); // x[a].getOrientation().inverse() * x[b].getOrientation();
	//u = dQ.toEulerVector() - dQ0.toEulerVector(); // Consider to use quatToRotationVector instead of toEulerVector to have the rotation vector

	dQ0.normalize();
	dQ.normalize();

	Quat tmpQ;
	tmpQ = tmpQ.quatDiff(dQ,dQ0);
	tmpQ.normalize();

	u = tmpQ.quatToRotationVector(); //dQ.quatToRotationVector() - dQ0.quatToRotationVector();  // Use of quatToRotationVector instead of toEulerVector:
	// this is done to keep the old behavior (before the
	// correction of the toEulerVector  function). If the
	// purpose was to obtain the Eulerian vector and not the
	// rotation vector please use the following line instead
	//u = tmpQ.toEulerVector(); //dQ.toEulerVector() - dQ0.toEulerVector();

	depl[3] = 0.0; 	depl[4] = 0.0; 	depl[5] = 0.0;
	depl[9] = u[0]; depl[10]= u[1]; depl[11]= u[2];

	// this computation can be optimised: (we know that half of "depl" is null)
	Displacement force = _k_loc[i] * depl;


	// Apply lambda transpose (we use the rotation value of point a for the beam)

	Vec3 fa1 = x[a].getOrientation().rotate(Vec3(force[0],force[1],force[2]));
	Vec3 fa2 = x[a].getOrientation().rotate(Vec3(force[3],force[4],force[5]));

	Vec3 fb1 = x[a].getOrientation().rotate(Vec3(force[6],force[7],force[8]));
	Vec3 fb2 = x[a].getOrientation().rotate(Vec3(force[9],force[10],force[11]));


	f[a] += Deriv(-fa1, -fa2);
	f[b] += Deriv(-fb1, -fb2);

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::applyStiffnessLarge(VecDeriv& df, const VecDeriv& dx, std::size_t i, Index a, Index b, Real fact)
{
	Displacement local_depl;
	Vec3 u;
	Quat& q = _quat[i]; //x[a].getOrientation();
	q.normalize();

	u = q.inverseRotate(getVCenter(dx[a]));
	local_depl[0] = u[0];
	local_depl[1] = u[1];
	local_depl[2] = u[2];

	u = q.inverseRotate(getVOrientation(dx[a]));
	local_depl[3] = u[0];
	local_depl[4] = u[1];
	local_depl[5] = u[2];

	u = q.inverseRotate(getVCenter(dx[b]));
	local_depl[6] = u[0];
	local_depl[7] = u[1];
	local_depl[8] = u[2];

	u = q.inverseRotate(getVOrientation(dx[b]));
	local_depl[9] = u[0];
	local_depl[10] = u[1];
	local_depl[11] = u[2];

	Displacement local_force = _k_loc[i] * local_depl;

	Vec3 fa1 = q.rotate(Vec3(local_force[0],local_force[1] ,local_force[2] ));
	Vec3 fa2 = q.rotate(Vec3(local_force[3],local_force[4] ,local_force[5] ));
	Vec3 fb1 = q.rotate(Vec3(local_force[6],local_force[7] ,local_force[8] ));
	Vec3 fb2 = q.rotate(Vec3(local_force[9],local_force[10],local_force[11]));


	df[a] += Deriv(-fa1,-fa2) * fact;
	df[b] += Deriv(-fb1,-fb2) * fact;
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
	sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
	Real k = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
	defaulttype::BaseMatrix* mat = r.matrix;

	if (r)
	{
		unsigned int &offset = r.offset;

		std::size_t i = 0;
		for(const Edge& e : _topology->getEdges())
		{
			Index a = e[0];
			Index b = e[1];

			Quat& q = _quat[i]; //x[a].getOrientation();
			q.normalize();
			Transformation R,Rt;
			q.toMatrix(R);
			Rt.transpose(R);
			const StiffnessMatrix& K0 = _k_loc[i];
			StiffnessMatrix K;
			bool exploitSymmetry = _useSymmetricAssembly.getValue();

			if (exploitSymmetry) {
				for (int x1=0; x1<12; x1+=3) {
					for (int y1=x1; y1<12; y1+=3)
					{
						Transformation m;
						K0.getsub(x1,y1, m);
						m = R*m*Rt;

						for (int i=0; i<3; i++)
							for (int j=0; j<3; j++) {
								K.elems[i+x1][j+y1] += m[i][j];
								K.elems[j+y1][i+x1] += m[i][j];
							}
						if (x1 == y1)
							for (int i=0; i<3; i++)
								for (int j=0; j<3; j++)
									K.elems[i+x1][j+y1] *= Real(0.5);

					}
				}
			} else  {
				for (int x1=0; x1<12; x1+=3) {
					for (int y1=0; y1<12; y1+=3)
					{
						Transformation m;
						K0.getsub(x1,y1, m);
						m = R*m*Rt;
						K.setsub(x1,y1, m);
					}
				}
			}

			int index[12];
			for (int x1=0; x1<6; x1++)
				index[x1] = offset+a*6+x1;
			for (int x1=0; x1<6; x1++)
				index[6+x1] = offset+b*6+x1;
			for (int x1=0; x1<12; ++x1)
				for (int y1=0; y1<12; ++y1)
					mat->add(index[x1], index[y1], - K(x1,y1)*k);

			++i;
		}


	}

}

template<class DataTypes>
SReal BeamFEMForceField<DataTypes>:: getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
	msg_error(this) << "Get potentialEnergy not implemented";
	return SReal(0.0);
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
	if (!vparams->displayFlags().getShowForceFields()) return;
	if (!this->mstate) return;

	const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

	std::array<std::vector<defaulttype::Vector3>, 3> points;

	std::size_t i = 0;
	for(const Edge& e : _topology->getEdges())
	{
		drawElement(i, e[0], e[1], points, x);
		++i;
	}

	vparams->drawTool()->drawLines(points[0], 1, defaulttype::Vec<4,float>(1,0,0,1));
	vparams->drawTool()->drawLines(points[1], 1, defaulttype::Vec<4,float>(0,1,0,1));
	vparams->drawTool()->drawLines(points[2], 1, defaulttype::Vec<4,float>(0,0,1,1));
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
	if( !onlyVisible ) return;


	static const Real max_real = std::numeric_limits<Real>::max();
	static const Real min_real = std::numeric_limits<Real>::lowest();
	Real maxBBox[3] = {min_real,min_real,min_real};
	Real minBBox[3] = {max_real,max_real,max_real};


	const size_t npoints = this->mstate->getSize();
	const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();

	for (size_t i=0; i<npoints; i++)
	{
		const defaulttype::Vector3 &pt = p[i].getCenter();

		for (int c=0; c<3; c++)
		{
			if (pt[c] > maxBBox[c]) maxBBox[c] = pt[c];
			else if (pt[c] < minBBox[c]) minBBox[c] = pt[c];
		}
	}

	this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::drawElement(std::size_t i, Index a, Index b, std::array<std::vector< defaulttype::Vector3 >, 3>& points, const VecCoord& x)
{
	defaulttype::Vector3 p;
	p = (x[a].getCenter()+x[b].getCenter())*0.5;

	defaulttype::Vector3 beamVec;
	beamVec[0]= d_L.getValue()[i]*0.5; beamVec[1] = 0.0; beamVec[2] = 0.0;

	const Quat& q = _quat[i];
	// axis X
	points[0].push_back(p - q.rotate(beamVec) );
	points[0].push_back(p + q.rotate(beamVec) );
	// axis Y
	beamVec[0]=0.0; beamVec[1] = d_r.getValue()[i]*0.5;
	points[1].push_back(p);
	points[1].push_back(p + q.rotate(beamVec) );
	// axis Z
	beamVec[1]=0.0; beamVec[2] = d_r.getValue()[i]*0.5;
	points[2].push_back(p);
	points[2].push_back(p + q.rotate(beamVec) );
}


} // namespace fem

} // namespace solidmechanics

} // namespace physicalmodel

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_PHYSICALMODEL_SOLIDMECHANICS_FEM_BEAMFEMFORCEFIELD_INL
