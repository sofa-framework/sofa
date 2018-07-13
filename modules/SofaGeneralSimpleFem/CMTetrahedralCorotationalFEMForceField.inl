/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_CMTETRAHEDRALCOROTATIONALFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_CMTETRAHEDRALCOROTATIONALFEMFORCEFIELD_INL

#include "CMTetrahedralCorotationalFEMForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/GridTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/decompose.h>
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/CMTopologyData.inl>
#include <assert.h>
#include <iostream>
#include <set>

namespace sofa
{

namespace component
{

namespace cm_forcefield
{

template< class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronHandler::applyCreateFunction(
		TetrahedronInformation &,
		core::topology::MapTopology::Volume bw,
		const sofa::helper::vector<core::topology::MapTopology::Volume> &,
		const sofa::helper::vector<double> &)
{
	const Volume w(bw);
	if (ff)
	{
		const VecCoord X0 = ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

		auto& attribute = *(ff->_volumeAttribute.beginEdit());
		TetrahedronInformation& info = attribute[w.dart];

		auto& t = ff->_topology->get_dofs(w);
		for (int i=0; i<4; ++i) info.dofs[i] = t[i];

		ff->computeMaterialStiffness(info);

		switch(ff->method)
		{
			case SMALL :
				ff->initSmall(X0, info);
				break;
			case LARGE :
			case PLARGE :
				ff->initLarge(X0, info);
				break;
			case POLAR :
				ff->initPolar(X0, info);
				break;
		}
		ff->_volumeAttribute.endEdit();
	}
}

template< class DataTypes>
CMTetrahedralCorotationalFEMForceField<DataTypes>::CMTetrahedralCorotationalFEMForceField()
	: _volumeAttribute(initData(&_volumeAttribute, "tetrahedronInfo", "Internal tetrahedron data"))
	, f_method(initData(&f_method,std::string("large"),"method","\"small\", \"large\" (by QR) or \"polar\" displacements"))
	, _poissonRatio(core::objectmodel::BaseObject::initData(&_poissonRatio,(Real)0.45f,"poissonRatio","FEM Poisson Ratio"))
	, _youngModulus(core::objectmodel::BaseObject::initData(&_youngModulus,(Real)5000,"youngModulus","FEM Young Modulus"))
	, _localStiffnessFactor(core::objectmodel::BaseObject::initData(&_localStiffnessFactor,"localStiffnessFactor","Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]"))
	, _updateStiffnessMatrix(core::objectmodel::BaseObject::initData(&_updateStiffnessMatrix,false,"updateStiffnessMatrix",""))
	, _assembling(core::objectmodel::BaseObject::initData(&_assembling,false,"computeGlobalMatrix",""))
	, f_drawing(initData(&f_drawing,true,"drawing"," draw the forcefield if true"))
	, drawColor1(initData(&drawColor1,defaulttype::Vec4f(0.0f,0.0f,1.0f,1.0f),"drawColor1"," draw color for faces 1"))
	, drawColor2(initData(&drawColor2,defaulttype::Vec4f(0.0f,0.5f,1.0f,1.0f),"drawColor2"," draw color for faces 2"))
	, drawColor3(initData(&drawColor3,defaulttype::Vec4f(0.0f,1.0f,1.0f,1.0f),"drawColor3"," draw color for faces 3"))
	, drawColor4(initData(&drawColor4,defaulttype::Vec4f(0.5f,1.0f,1.0f,1.0f),"drawColor4"," draw color for faces 4"))
	, tetrahedronHandler(NULL)
{
	this->addAlias(&_assembling, "assembling");
	_poissonRatio.setWidget("poissonRatio");
	tetrahedronHandler = new TetrahedronHandler(this,&_volumeAttribute);

	_poissonRatio.setRequired(true);
	_youngModulus.setRequired(true);
}

template <class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::init()
{
	this->core::behavior::ForceField<DataTypes>::init();

	this->getContext()->get(_topology);

	if( _topology == NULL )
	{
		serr << "ERROR(CMTetrahedralCorotationalFEMForceField): object must have a Topology."<<sendl;
		return;
	}

	// TODO : verify that _topoloy only contains tetrahedron
	if (_topology->nb_cells<VolumeTopology::Volume::ORBIT>() == 0u)
	{
		serr << "ERROR(CMTetrahedralCorotationalFEMForceField): object must have a Tetrahedral Set Topology."<<sendl;
		return;
	}

	auto& attribute = *(_volumeAttribute.beginEdit());
	attribute = _topology->add_attribute<TetrahedronInformation, Volume>("CMTetrahedralCorotationalFEMForceField_tetrahedronInfo");
	_volumeAttribute.endEdit();

	reinit(); // compute per-element stiffness matrices and other precomputed values
}

template <class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::reinit()
{
	if (f_method.getValue() == "small")
		this->setMethod(SMALL);
	else if (f_method.getValue() == "polar")
		this->setMethod(POLAR);
	else if (f_method.getValue() == "plarge")
		this->setMethod(PLARGE);
	else this->setMethod(LARGE);

	// Need to initialize the _stiffnesses vector before using it
	size_t sizeMO=this->mstate->getSize();
	if(_assembling.getValue()) _stiffnesses.resize(sizeMO * 3);

	auto& attribute = *(_volumeAttribute.beginEdit());

	_topology->parallel_foreach_cell([&](Volume w)
	{
		TetrahedronInformation& info = attribute[w.dart];

		tetrahedronHandler->applyCreateFunction(info, BaseVolume(w.dart), helper::vector< BaseVolume >(), helper::vector< double >());
	});

	_volumeAttribute.createTopologicalEngine(_topology,tetrahedronHandler);
	_volumeAttribute.registerTopologicalData();

	_volumeAttribute.endEdit();
}


template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
	sofa::helper::AdvancedTimer::stepBegin("AddForce");
	VecDeriv& f = *d_f.beginEdit();
	const VecCoord& p = d_x.getValue();

	auto& attribute = *(_volumeAttribute.beginEdit());

	switch(method)
	{
		case SMALL :
		{
			//erase the stiffness matrix at each time step
			for(unsigned int i=0; i<_stiffnesses.size(); ++i) _stiffnesses[i].resize(0);

			_topology->foreach_cell([&](Volume w)
			{
				TetrahedronInformation& info = attribute[w.dart];
				accumulateForceSmall(f, p, info);
			});
			break;
		}
		case LARGE :
		{
			//erase the stiffness matrix at each time step
			for(unsigned int i=0; i<_stiffnesses.size(); ++i) _stiffnesses[i].resize(0);

			_topology->foreach_cell([&](Volume w)
			{
				TetrahedronInformation& info = attribute[w.dart];
				accumulateForceLarge(f, p, info);
			});
			break;
		}
			/* Parallelisme activé */
		case PLARGE :
		{
			//erase the stiffness matrix at each time step
			for(unsigned int i=0; i<_stiffnesses.size(); ++i) _stiffnesses[i].resize(0);

			cgogn::uint32 nbThreads = cgogn::thread_pool()->nb_workers();
			unsigned int l = f.size();
			// TODO: adpat to the size of these vector to the real number od threads
			VecDeriv threadF[32];
			for (cgogn::uint32 threadId = 0; threadId < nbThreads; ++threadId)
			{
				threadF[threadId].reserve(l);
				for (unsigned int i = 0; i < l; ++i)
					threadF[threadId][i] = sofa::defaulttype::Vec3d();
			}

			_topology->parallel_foreach_cell([&](Volume w)
			{
				TetrahedronInformation& info = attribute[w.dart];
				VecDeriv& localF = threadF[cgogn::current_thread_index()];
				accumulateForceLarge(localF, p, info);
			});

			for (cgogn::uint32 threadId = 0; threadId < nbThreads; ++threadId)
			{
				for (unsigned int i = 0; i < l; ++i)
					f[i] += threadF[threadId][i];
			}
			break;
		}
		case POLAR :
		{
			_topology->foreach_cell([&](Volume w)
			{
				TetrahedronInformation& info = attribute[w.dart];
				accumulateForcePolar( f, p, info);
			});
			break;
		}
	}
	_volumeAttribute.endEdit();
	d_f.endEdit();
	sofa::helper::AdvancedTimer::stepEnd("AddForce");
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
	sofa::helper::AdvancedTimer::stepBegin("AddDForce");
	VecDeriv& df = *d_df.beginEdit();
	const VecDeriv& dx = d_dx.getValue();

	Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

	const auto& attribute = _volumeAttribute.getValue();

	switch(method)
	{
		case SMALL :
		{
			_topology->foreach_cell([&](Volume w)
			{
				const TetrahedronInformation& info = attribute[w.dart];
				applyStiffnessSmall( df, dx, info, kFactor );
			});
			break;
		}
		case LARGE :
		{
			_topology->foreach_cell([&](Volume w)
			{
				const TetrahedronInformation& info = attribute[w.dart];
				applyStiffnessLarge( df, dx, info, kFactor );
			});
			break;
		}
		case PLARGE :
		{
			cgogn::uint32 nbThreads = cgogn::thread_pool()->nb_workers();
			unsigned int l = df.size();
			// TODO: adpat to the size of these vector to the real number od threads
			VecDeriv threadDF[32];
			for (cgogn::uint32 threadId = 0; threadId < nbThreads; ++threadId)
			{
				threadDF[threadId].reserve(l);
				for (unsigned int i = 0; i < l; ++i) threadDF[threadId][i] = sofa::defaulttype::Vec3d();
			}

			_topology->parallel_foreach_cell([&](Volume w)
			{
				const TetrahedronInformation& info = attribute[w.dart];
				VecDeriv& localDF = threadDF[cgogn::current_thread_index()];
				applyStiffnessLarge( localDF, dx, info, kFactor );
			});

			for (cgogn::uint32 threadId = 0; threadId < nbThreads; ++threadId)
			{
				for (unsigned int i = 0; i < l; ++i) df[i] += threadDF[threadId][i];
			}
			break;
		}
		case POLAR :
		{
			_topology->foreach_cell([&](Volume w)
			{
				const TetrahedronInformation& info = attribute[w.dart];
				applyStiffnessPolar(df, dx, info, kFactor);
			});
			break;
		}
	}

	d_df.endEdit();
	sofa::helper::AdvancedTimer::stepEnd("AddDForce");
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::computeStrainDisplacement( StrainDisplacementTransposed &J, Coord a, Coord b, Coord c, Coord d )
{
	// shape functions matrix
	defaulttype::Mat<2, 3, Real> M;

	M[0][0] = b[1];
	M[0][1] = c[1];
	M[0][2] = d[1];
	M[1][0] = b[2];
	M[1][1] = c[2];
	M[1][2] = d[2];
	J[0][0] = J[1][3] = J[2][5]   = - peudo_determinant_for_coef( M );
	M[0][0] = b[0];
	M[0][1] = c[0];
	M[0][2] = d[0];
	J[0][3] = J[1][1] = J[2][4]   = peudo_determinant_for_coef( M );
	M[1][0] = b[1];
	M[1][1] = c[1];
	M[1][2] = d[1];
	J[0][5] = J[1][4] = J[2][2]   = - peudo_determinant_for_coef( M );

	M[0][0] = c[1];
	M[0][1] = d[1];
	M[0][2] = a[1];
	M[1][0] = c[2];
	M[1][1] = d[2];
	M[1][2] = a[2];
	J[3][0] = J[4][3] = J[5][5]   = peudo_determinant_for_coef( M );
	M[0][0] = c[0];
	M[0][1] = d[0];
	M[0][2] = a[0];
	J[3][3] = J[4][1] = J[5][4]   = - peudo_determinant_for_coef( M );
	M[1][0] = c[1];
	M[1][1] = d[1];
	M[1][2] = a[1];
	J[3][5] = J[4][4] = J[5][2]   = peudo_determinant_for_coef( M );

	M[0][0] = d[1];
	M[0][1] = a[1];
	M[0][2] = b[1];
	M[1][0] = d[2];
	M[1][1] = a[2];
	M[1][2] = b[2];
	J[6][0] = J[7][3] = J[8][5]   = - peudo_determinant_for_coef( M );
	M[0][0] = d[0];
	M[0][1] = a[0];
	M[0][2] = b[0];
	J[6][3] = J[7][1] = J[8][4]   = peudo_determinant_for_coef( M );
	M[1][0] = d[1];
	M[1][1] = a[1];
	M[1][2] = b[1];
	J[6][5] = J[7][4] = J[8][2]   = - peudo_determinant_for_coef( M );

	M[0][0] = a[1];
	M[0][1] = b[1];
	M[0][2] = c[1];
	M[1][0] = a[2];
	M[1][1] = b[2];
	M[1][2] = c[2];
	J[9][0] = J[10][3] = J[11][5]   = peudo_determinant_for_coef( M );
	M[0][0] = a[0];
	M[0][1] = b[0];
	M[0][2] = c[0];
	J[9][3] = J[10][1] = J[11][4]   = - peudo_determinant_for_coef( M );
	M[1][0] = a[1];
	M[1][1] = b[1];
	M[1][2] = c[1];
	J[9][5] = J[10][4] = J[11][2]   = peudo_determinant_for_coef( M );

	// 0
	J[0][1] = J[0][2] = J[0][4] = J[1][0] =  J[1][2] =  J[1][5] =  J[2][0] =  J[2][1] =  J[2][3]  = 0;
	J[3][1] = J[3][2] = J[3][4] = J[4][0] =  J[4][2] =  J[4][5] =  J[5][0] =  J[5][1] =  J[5][3]  = 0;
	J[6][1] = J[6][2] = J[6][4] = J[7][0] =  J[7][2] =  J[7][5] =  J[8][0] =  J[8][1] =  J[8][3]  = 0;
	J[9][1] = J[9][2] = J[9][4] = J[10][0] = J[10][2] = J[10][5] = J[11][0] = J[11][1] = J[11][3] = 0;
}

template<class DataTypes>
typename CMTetrahedralCorotationalFEMForceField<DataTypes>::Real CMTetrahedralCorotationalFEMForceField<DataTypes>::peudo_determinant_for_coef(const defaulttype::Mat<2, 3, Real>& M)
{
	return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::computeStiffnessMatrix(StiffnessMatrix& S, StiffnessMatrix& SR, const MaterialStiffness &K, const StrainDisplacementTransposed &J, const Transformation& Rot)
{
	defaulttype::MatNoInit<6, 12, Real> Jt;
	Jt.transpose(J);

	defaulttype::MatNoInit<12, 12, Real> JKJt;
	JKJt = J*K*Jt;

	defaulttype::MatNoInit<12, 12, Real> RR, RRt;
	RR.clear();
	RRt.clear();
	for(int i=0; i<3; ++i)
		for(int j=0; j<3; ++j)
		{
			RR[i][j]=RR[i+3][j+3]=RR[i+6][j+6]=RR[i+9][j+9]=Rot[i][j];
			RRt[i][j]=RRt[i+3][j+3]=RRt[i+6][j+6]=RRt[i+9][j+9]=Rot[j][i];
		}

	S = RR*JKJt;
	SR = S*RRt;
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::computeMaterialStiffness(TetrahedronInformation& info)
{
	//	const VecReal& localStiffnessFactor = _localStiffnessFactor.getValue();
	SReal factor = 1.0f;
	//	if (!localStiffnessFactor.empty())
	//		factor = localStiffnessFactor[w.dart.index*localStiffnessFactor.size()/_topology->getNbTetrahedra()];

	computeMaterialStiffness(info.materialMatrix, info.dofs[0], info.dofs[1], info.dofs[2], info.dofs[3], factor);
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::computeMaterialStiffness(MaterialStiffness& materialMatrix, Index&a, Index&b, Index&c, Index&d, SReal localStiffnessFactor)
{

	//const VecReal& localStiffnessFactor = _localStiffnessFactor.getValue();
	const Real youngModulus = _youngModulus.getValue()*(Real)localStiffnessFactor;
	const Real poissonRatio = _poissonRatio.getValue();

	materialMatrix[0][0] = materialMatrix[1][1] = materialMatrix[2][2] = 1;
	materialMatrix[0][1] = materialMatrix[0][2] = materialMatrix[1][0] =
			materialMatrix[1][2] = materialMatrix[2][0] = materialMatrix[2][1] = poissonRatio/(1-poissonRatio);
	materialMatrix[0][3] = materialMatrix[0][4] = materialMatrix[0][5] = 0;
	materialMatrix[1][3] = materialMatrix[1][4] = materialMatrix[1][5] = 0;
	materialMatrix[2][3] = materialMatrix[2][4] = materialMatrix[2][5] = 0;
	materialMatrix[3][0] = materialMatrix[3][1] = materialMatrix[3][2] =
			materialMatrix[3][4] = materialMatrix[3][5] = 0;
	materialMatrix[4][0] = materialMatrix[4][1] = materialMatrix[4][2] =
			materialMatrix[4][3] = materialMatrix[4][5] = 0;
	materialMatrix[5][0] = materialMatrix[5][1] = materialMatrix[5][2] =
			materialMatrix[5][3] = materialMatrix[5][4] = 0;
	materialMatrix[3][3] = materialMatrix[4][4] = materialMatrix[5][5] =
			(1-2*poissonRatio)/(2*(1-poissonRatio));
	materialMatrix *= (youngModulus*(1-poissonRatio))/((1+poissonRatio)*(1-2*poissonRatio));

	// divide by 36 times volumes of the element
	const VecCoord X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

	Coord A = (X0)[b] - (X0)[a];
	Coord B = (X0)[c] - (X0)[a];
	Coord C = (X0)[d] - (X0)[a];
	Coord AB = cross(A, B);
	Real volumes6 = fabs( dot( AB, C ) );
	if (volumes6<0)
	{
		serr << "ERROR: Negative volume for tetra "<<a<<','<<b<<','<<c<<','<<d<<"> = "<<volumes6/6<<sendl;
	}
	materialMatrix /= (volumes6*6);
}

template<class DataTypes>
inline void CMTetrahedralCorotationalFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacementTransposed &J )
{
	// Unit of K = unit of youngModulus / unit of volume = Pa / m^3 = kg m^-4 s^-2
	// Unit of J = m^2
	// Unit of JKJt =  kg s^-2
	// Unit of displacement = m
	// Unit of force = kg m s^-2

#if 0
	F = J*(K*(J.multTranspose(Depl)));
#else
	/* We have these zeros
									   K[0][3]   K[0][4]   K[0][5]
									   K[1][3]   K[1][4]   K[1][5]
									   K[2][3]   K[2][4]   K[2][5]
		 K[3][0]   K[3][1]   K[3][2]             K[3][4]   K[3][5]
		 K[4][0]   K[4][1]   K[4][2]   K[4][3]             K[4][5]
		 K[5][0]   K[5][1]   K[5][2]   K[5][3]   K[5][4]


				   J[0][1]   J[0][2]             J[0][4]
		 J[1][0]             J[1][2]                       J[1][5]
		 J[2][0]   J[2][1]             J[2][3]
				   J[3][1]   J[3][2]             J[3][4]
		 J[4][0]             J[4][2]                       J[4][5]
		 J[5][0]   J[5][1]             J[5][3]
				   J[6][1]   J[6][2]             J[6][4]
		 J[7][0]             J[7][2]                       J[7][5]
		 J[8][0]   J[8][1]             J[8][3]
				   J[9][1]   J[9][2]             J[9][4]
		 J[10][0]            J[10][2]                      J[10][5]
		 J[11][0]  J[11][1]            J[11][3]
		 */

	defaulttype::VecNoInit<6,Real> JtD;
	JtD[0] =   J[ 0][0]*Depl[ 0]+/*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
			J[ 3][0]*Depl[ 3]+/*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
			J[ 6][0]*Depl[ 6]+/*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
			J[ 9][0]*Depl[ 9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
	JtD[1] = /*J[ 0][1]*Depl[ 0]+*/J[ 1][1]*Depl[ 1]+/*J[ 2][1]*Depl[ 2]+*/
			/*J[ 3][1]*Depl[ 3]+*/J[ 4][1]*Depl[ 4]+/*J[ 5][1]*Depl[ 5]+*/
			/*J[ 6][1]*Depl[ 6]+*/J[ 7][1]*Depl[ 7]+/*J[ 8][1]*Depl[ 8]+*/
			/*J[ 9][1]*Depl[ 9]+*/J[10][1]*Depl[10] /*J[11][1]*Depl[11]*/;
	JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/J[ 2][2]*Depl[ 2]+
			/*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/J[ 5][2]*Depl[ 5]+
			/*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/J[ 8][2]*Depl[ 8]+
			/*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/J[11][2]*Depl[11]  ;
	JtD[3] =   J[ 0][3]*Depl[ 0]+  J[ 1][3]*Depl[ 1]+/*J[ 2][3]*Depl[ 2]+*/
			J[ 3][3]*Depl[ 3]+  J[ 4][3]*Depl[ 4]+/*J[ 5][3]*Depl[ 5]+*/
			J[ 6][3]*Depl[ 6]+  J[ 7][3]*Depl[ 7]+/*J[ 8][3]*Depl[ 8]+*/
			J[ 9][3]*Depl[ 9]+  J[10][3]*Depl[10] /*J[11][3]*Depl[11]*/;
	JtD[4] = /*J[ 0][4]*Depl[ 0]+*/J[ 1][4]*Depl[ 1]+  J[ 2][4]*Depl[ 2]+
			/*J[ 3][4]*Depl[ 3]+*/J[ 4][4]*Depl[ 4]+  J[ 5][4]*Depl[ 5]+
			/*J[ 6][4]*Depl[ 6]+*/J[ 7][4]*Depl[ 7]+  J[ 8][4]*Depl[ 8]+
			/*J[ 9][4]*Depl[ 9]+*/J[10][4]*Depl[10]+  J[11][4]*Depl[11]  ;
	JtD[5] =   J[ 0][5]*Depl[ 0]+/*J[ 1][5]*Depl[ 1]*/ J[ 2][5]*Depl[ 2]+
			J[ 3][5]*Depl[ 3]+/*J[ 4][5]*Depl[ 4]*/ J[ 5][5]*Depl[ 5]+
			J[ 6][5]*Depl[ 6]+/*J[ 7][5]*Depl[ 7]*/ J[ 8][5]*Depl[ 8]+
			J[ 9][5]*Depl[ 9]+/*J[10][5]*Depl[10]*/ J[11][5]*Depl[11];

	defaulttype::VecNoInit<6,Real> KJtD;
	KJtD[0] =   K[0][0]*JtD[0]+  K[0][1]*JtD[1]+  K[0][2]*JtD[2]
			/*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
	KJtD[1] =   K[1][0]*JtD[0]+  K[1][1]*JtD[1]+  K[1][2]*JtD[2]
			/*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
	KJtD[2] =   K[2][0]*JtD[0]+  K[2][1]*JtD[1]+  K[2][2]*JtD[2]
			/*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
	KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
			K[3][3]*JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
	KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
			/*K[4][3]*JtD[3]+*/K[4][4]*JtD[4] /*K[4][5]*JtD[5]*/;
	KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
			/*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/K[5][5]*JtD[5]  ;

	F[ 0] =   J[ 0][0]*KJtD[0]+/*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
			J[ 0][3]*KJtD[3]+/*J[ 0][4]*KJtD[4]+*/J[ 0][5]*KJtD[5]  ;
	F[ 1] = /*J[ 1][0]*KJtD[0]+*/J[ 1][1]*KJtD[1]+/*J[ 1][2]*KJtD[2]+*/
			J[ 1][3]*KJtD[3]+  J[ 1][4]*KJtD[4] /*J[ 1][5]*KJtD[5]*/;
	F[ 2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/J[ 2][2]*KJtD[2]+
			/*J[ 2][3]*KJtD[3]+*/J[ 2][4]*KJtD[4]+  J[ 2][5]*KJtD[5]  ;
	F[ 3] =   J[ 3][0]*KJtD[0]+/*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
			J[ 3][3]*KJtD[3]+/*J[ 3][4]*KJtD[4]+*/J[ 3][5]*KJtD[5]  ;
	F[ 4] = /*J[ 4][0]*KJtD[0]+*/J[ 4][1]*KJtD[1]+/*J[ 4][2]*KJtD[2]+*/
			J[ 4][3]*KJtD[3]+  J[ 4][4]*KJtD[4] /*J[ 4][5]*KJtD[5]*/;
	F[ 5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/J[ 5][2]*KJtD[2]+
			/*J[ 5][3]*KJtD[3]+*/J[ 5][4]*KJtD[4]+  J[ 5][5]*KJtD[5]  ;
	F[ 6] =   J[ 6][0]*KJtD[0]+/*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
			J[ 6][3]*KJtD[3]+/*J[ 6][4]*KJtD[4]+*/J[ 6][5]*KJtD[5]  ;
	F[ 7] = /*J[ 7][0]*KJtD[0]+*/J[ 7][1]*KJtD[1]+/*J[ 7][2]*KJtD[2]+*/
			J[ 7][3]*KJtD[3]+  J[ 7][4]*KJtD[4] /*J[ 7][5]*KJtD[5]*/;
	F[ 8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/J[ 8][2]*KJtD[2]+
			/*J[ 8][3]*KJtD[3]+*/J[ 8][4]*KJtD[4]+  J[ 8][5]*KJtD[5]  ;
	F[ 9] =   J[ 9][0]*KJtD[0]+/*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
			J[ 9][3]*KJtD[3]+/*J[ 9][4]*KJtD[4]+*/J[ 9][5]*KJtD[5]  ;
	F[10] = /*J[10][0]*KJtD[0]+*/J[10][1]*KJtD[1]+/*J[10][2]*KJtD[2]+*/
			J[10][3]*KJtD[3]+  J[10][4]*KJtD[4] /*J[10][5]*KJtD[5]*/;
	F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/J[11][2]*KJtD[2]+
			/*J[11][3]*KJtD[3]+*/J[11][4]*KJtD[4]+  J[11][5]*KJtD[5]  ;
	//        serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, D = "<<Depl<<sendl;
	//        serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, JtD = "<<JtD<<sendl;
	//        serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, K = "<<K<<sendl;
	//        serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, KJtD = "<<KJtD<<sendl;
	//        serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, F = "<<F<<sendl;
#endif
}

template<class DataTypes>
inline void CMTetrahedralCorotationalFEMForceField<DataTypes>::computeForce(Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacementTransposed &J, SReal fact)
{
	// Unit of K = unit of youngModulus / unit of volume = Pa / m^3 = kg m^-4 s^-2
	// Unit of J = m^2
	// Unit of JKJt =  kg s^-2
	// Unit of displacement = m
	// Unit of force = kg m s^-2

#if 0
	F = J*(K*(J.multTranspose(Depl)));
	F *= fact;
#else
	/* We have these zeros
									   K[0][3]   K[0][4]   K[0][5]
									   K[1][3]   K[1][4]   K[1][5]
									   K[2][3]   K[2][4]   K[2][5]
		 K[3][0]   K[3][1]   K[3][2]             K[3][4]   K[3][5]
		 K[4][0]   K[4][1]   K[4][2]   K[4][3]             K[4][5]
		 K[5][0]   K[5][1]   K[5][2]   K[5][3]   K[5][4]


				   J[0][1]   J[0][2]             J[0][4]
		 J[1][0]             J[1][2]                       J[1][5]
		 J[2][0]   J[2][1]             J[2][3]
				   J[3][1]   J[3][2]             J[3][4]
		 J[4][0]             J[4][2]                       J[4][5]
		 J[5][0]   J[5][1]             J[5][3]
				   J[6][1]   J[6][2]             J[6][4]
		 J[7][0]             J[7][2]                       J[7][5]
		 J[8][0]   J[8][1]             J[8][3]
				   J[9][1]   J[9][2]             J[9][4]
		 J[10][0]            J[10][2]                      J[10][5]
		 J[11][0]  J[11][1]            J[11][3]
		 */

	defaulttype::VecNoInit<6,Real> JtD;
	JtD[0] =   J[ 0][0]*Depl[ 0]+/*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
			J[ 3][0]*Depl[ 3]+/*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
			J[ 6][0]*Depl[ 6]+/*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
			J[ 9][0]*Depl[ 9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
	JtD[1] = /*J[ 0][1]*Depl[ 0]+*/J[ 1][1]*Depl[ 1]+/*J[ 2][1]*Depl[ 2]+*/
			/*J[ 3][1]*Depl[ 3]+*/J[ 4][1]*Depl[ 4]+/*J[ 5][1]*Depl[ 5]+*/
			/*J[ 6][1]*Depl[ 6]+*/J[ 7][1]*Depl[ 7]+/*J[ 8][1]*Depl[ 8]+*/
			/*J[ 9][1]*Depl[ 9]+*/J[10][1]*Depl[10] /*J[11][1]*Depl[11]*/;
	JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/J[ 2][2]*Depl[ 2]+
			/*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/J[ 5][2]*Depl[ 5]+
			/*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/J[ 8][2]*Depl[ 8]+
			/*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/J[11][2]*Depl[11]  ;
	JtD[3] =   J[ 0][3]*Depl[ 0]+  J[ 1][3]*Depl[ 1]+/*J[ 2][3]*Depl[ 2]+*/
			J[ 3][3]*Depl[ 3]+  J[ 4][3]*Depl[ 4]+/*J[ 5][3]*Depl[ 5]+*/
			J[ 6][3]*Depl[ 6]+  J[ 7][3]*Depl[ 7]+/*J[ 8][3]*Depl[ 8]+*/
			J[ 9][3]*Depl[ 9]+  J[10][3]*Depl[10] /*J[11][3]*Depl[11]*/;
	JtD[4] = /*J[ 0][4]*Depl[ 0]+*/J[ 1][4]*Depl[ 1]+  J[ 2][4]*Depl[ 2]+
			/*J[ 3][4]*Depl[ 3]+*/J[ 4][4]*Depl[ 4]+  J[ 5][4]*Depl[ 5]+
			/*J[ 6][4]*Depl[ 6]+*/J[ 7][4]*Depl[ 7]+  J[ 8][4]*Depl[ 8]+
			/*J[ 9][4]*Depl[ 9]+*/J[10][4]*Depl[10]+  J[11][4]*Depl[11]  ;
	JtD[5] =   J[ 0][5]*Depl[ 0]+/*J[ 1][5]*Depl[ 1]*/ J[ 2][5]*Depl[ 2]+
			J[ 3][5]*Depl[ 3]+/*J[ 4][5]*Depl[ 4]*/ J[ 5][5]*Depl[ 5]+
			J[ 6][5]*Depl[ 6]+/*J[ 7][5]*Depl[ 7]*/ J[ 8][5]*Depl[ 8]+
			J[ 9][5]*Depl[ 9]+/*J[10][5]*Depl[10]*/ J[11][5]*Depl[11];
	//         serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, D = "<<Depl<<sendl;
	//         serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, JtD = "<<JtD<<sendl;

	defaulttype::VecNoInit<6,Real> KJtD;
	KJtD[0] =   K[0][0]*JtD[0]+  K[0][1]*JtD[1]+  K[0][2]*JtD[2]
			/*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
	KJtD[1] =   K[1][0]*JtD[0]+  K[1][1]*JtD[1]+  K[1][2]*JtD[2]
			/*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
	KJtD[2] =   K[2][0]*JtD[0]+  K[2][1]*JtD[1]+  K[2][2]*JtD[2]
			/*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
	KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
			K[3][3]*JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
	KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
			/*K[4][3]*JtD[3]+*/K[4][4]*JtD[4] /*K[4][5]*JtD[5]*/;
	KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
			/*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/K[5][5]*JtD[5]  ;

	KJtD *= fact;

	F[ 0] =   J[ 0][0]*KJtD[0]+/*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
			J[ 0][3]*KJtD[3]+/*J[ 0][4]*KJtD[4]+*/J[ 0][5]*KJtD[5]  ;
	F[ 1] = /*J[ 1][0]*KJtD[0]+*/J[ 1][1]*KJtD[1]+/*J[ 1][2]*KJtD[2]+*/
			J[ 1][3]*KJtD[3]+  J[ 1][4]*KJtD[4] /*J[ 1][5]*KJtD[5]*/;
	F[ 2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/J[ 2][2]*KJtD[2]+
			/*J[ 2][3]*KJtD[3]+*/J[ 2][4]*KJtD[4]+  J[ 2][5]*KJtD[5]  ;
	F[ 3] =   J[ 3][0]*KJtD[0]+/*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
			J[ 3][3]*KJtD[3]+/*J[ 3][4]*KJtD[4]+*/J[ 3][5]*KJtD[5]  ;
	F[ 4] = /*J[ 4][0]*KJtD[0]+*/J[ 4][1]*KJtD[1]+/*J[ 4][2]*KJtD[2]+*/
			J[ 4][3]*KJtD[3]+  J[ 4][4]*KJtD[4] /*J[ 4][5]*KJtD[5]*/;
	F[ 5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/J[ 5][2]*KJtD[2]+
			/*J[ 5][3]*KJtD[3]+*/J[ 5][4]*KJtD[4]+  J[ 5][5]*KJtD[5]  ;
	F[ 6] =   J[ 6][0]*KJtD[0]+/*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
			J[ 6][3]*KJtD[3]+/*J[ 6][4]*KJtD[4]+*/J[ 6][5]*KJtD[5]  ;
	F[ 7] = /*J[ 7][0]*KJtD[0]+*/J[ 7][1]*KJtD[1]+/*J[ 7][2]*KJtD[2]+*/
			J[ 7][3]*KJtD[3]+  J[ 7][4]*KJtD[4] /*J[ 7][5]*KJtD[5]*/;
	F[ 8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/J[ 8][2]*KJtD[2]+
			/*J[ 8][3]*KJtD[3]+*/J[ 8][4]*KJtD[4]+  J[ 8][5]*KJtD[5]  ;
	F[ 9] =   J[ 9][0]*KJtD[0]+/*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
			J[ 9][3]*KJtD[3]+/*J[ 9][4]*KJtD[4]+*/J[ 9][5]*KJtD[5]  ;
	F[10] = /*J[10][0]*KJtD[0]+*/J[10][1]*KJtD[1]+/*J[10][2]*KJtD[2]+*/
			J[10][3]*KJtD[3]+  J[10][4]*KJtD[4] /*J[10][5]*KJtD[5]*/;
	F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/J[11][2]*KJtD[2]+
			/*J[11][3]*KJtD[3]+*/J[11][4]*KJtD[4]+  J[11][5]*KJtD[5]  ;
#endif
}

//////////////////////////////////////////////////////////////////////
////////////////////  small displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::initSmall(const VecCoord& X0, TetrahedronInformation& info)
{
	computeStrainDisplacement(info.strainDisplacementTransposedMatrix,
							  (X0)[info.dofs[0]], (X0)[info.dofs[1]], (X0)[info.dofs[2]], (X0)[info.dofs[3]] );
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall(Vector& f, const Vector & p, TetrahedronInformation& info)
{
	const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

	// displacements
	Displacement D;
	D[0]  = 0.0f;
	D[1]  = 0.0f;
	D[2]  = 0.0f;
	for (int i=3; i<12; ++i)
		D[i]  = (X0)[info.dofs[i/3]][i%3] - (X0)[info.dofs[0]][i%3] - p[info.dofs[i/3]][i%3] + p[info.dofs[0]][i%3];

	// compute force on element
	Displacement F;

	if(!_assembling.getValue())
	{
		computeForce(F, D, info.materialMatrix, info.strainDisplacementTransposedMatrix);
	}
	else
	{
		Transformation Rot;
		Rot[0][0] = Rot[1][1] = Rot[2][2] = 1;
		Rot[0][1] = Rot[0][2] = 0;
		Rot[1][0] = Rot[1][2] = 0;
		Rot[2][0] = Rot[2][1] = 0;

		StiffnessMatrix JKJt;
		StiffnessMatrix tmp;
		computeStiffnessMatrix(JKJt, tmp, info.materialMatrix, info.strainDisplacementTransposedMatrix, Rot);

		for(int i=0; i<12; ++i)
		{
			int row = info.dofs[i/3]*3+i%3;

			for(int j=0; j<12; ++j)
			{
				if(JKJt[i][j]!=0)
				{
					int col = info.dofs[j/3]*3+j%3;

					// search if the vertex is already take into account by another element
					typename CompressedValue::iterator result = _stiffnesses[row].end();
					for(typename CompressedValue::iterator it=_stiffnesses[row].begin(); it!=_stiffnesses[row].end()&&result==_stiffnesses[row].end(); ++it)
					{
						if( (*it).first == col )
							result = it;
					}

					if( result==_stiffnesses[row].end() )
						_stiffnesses[row].push_back( Col_Value(col,JKJt[i][j] )  );
					else
						(*result).second += JKJt[i][j];
				}
			}
		}
		F = JKJt * D;
	}

	for (int i=0; i<12; i+=3)
		f[info.dofs[i/3]] += Deriv(F[i], F[i+1], F[i+2]);
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessSmall(Vector& f, const Vector& x, const TetrahedronInformation& info, SReal fact )
{
	Displacement X;
	for (int i = 0; i<12; ++i)
		X[i] = x[info.dofs[i/3]][i%3];

	Displacement F;
	computeForce( F, X,info.materialMatrix,info.strainDisplacementTransposedMatrix, fact);

	for(int i=0; i<12; i+=3)
		f[info.dofs[i/3]] += Deriv(-F[i], -F[i+1], -F[i+2]);
}

//////////////////////////////////////////////////////////////////////
////////////////////  large displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
inline void CMTetrahedralCorotationalFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c)
{
	// first vector on first edge
	// second vector in the plane of the two first edges
	// third vector orthogonal to first and second

	Coord edgex = p[b]-p[a];
	edgex.normalize();

	Coord edgey = p[c]-p[a];
	edgey.normalize();

	Coord edgez = cross( edgex, edgey );
	edgez.normalize();

	edgey = cross( edgez, edgex );
	edgey.normalize();

	r[0][0] = edgex[0];
	r[0][1] = edgex[1];
	r[0][2] = edgex[2];
	r[1][0] = edgey[0];
	r[1][1] = edgey[1];
	r[1][2] = edgey[2];
	r[2][0] = edgez[0];
	r[2][1] = edgez[1];
	r[2][2] = edgez[2];
}

template <class DataTypes>
inline void CMTetrahedralCorotationalFEMForceField<DataTypes>::getElementRotation(Transformation& R, unsigned int elementIdx)
{
	auto& tetraInf = *(_volumeAttribute.beginEdit());
	TetrahedronInformation *tinfo = &tetraInf[elementIdx];
	Transformation r01,r21;
	r01=tinfo->initialTransformation;
	r21=tinfo->rotation*r01;
	R=r21;
}

template <class DataTypes>
inline void CMTetrahedralCorotationalFEMForceField<DataTypes>::getRotation(Transformation& R, Vertex vertex)
{
	const auto& tetrahedronInf = _volumeAttribute.getValue();

	//    int numNeiTetra=_topology->getTetrahedraAroundVertex(w).size();
	Transformation r;
	r.clear();

	unsigned int nb_neighbors = 0u;

	_topology->foreach_incident_volume(vertex, [&](Volume w) {
		const TetrahedronInformation *tinfo = &tetrahedronInf[w.dart];
		Transformation r01,r21;
		r01=tinfo->initialTransformation;
		r21=tinfo->rotation*r01;
		r+=r21;
		++nb_neighbors;
	});

	R=r*(1.0f/nb_neighbors);

	//orthogonalization
	Coord ex,ey,ez;
	for(int i=0; i<3; i++)
	{
		ex[i]=R[0][i];
		ey[i]=R[1][i];
	}
	ex.normalize();
	ey.normalize();

	ez=cross(ex,ey);
	ez.normalize();

	ey=cross(ez,ex);
	ey.normalize();

	for(int i=0; i<3; i++)
	{
		R[0][i]=ex[i];
		R[1][i]=ey[i];
		R[2][i]=ez[i];
	}
}

template <class DataTypes>
inline void CMTetrahedralCorotationalFEMForceField<DataTypes>::getElementStiffnessMatrix(Real* stiffness, Volume w)
{
	const auto& tetrahedronInfData = _volumeAttribute.getValue();
	const auto& tetrahedronInf = tetrahedronInfData[w.dart];

	Transformation Rot;
	StiffnessMatrix JKJt,tmp;
	Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
	Rot[0][1]=Rot[0][2]=0;
	Rot[1][0]=Rot[1][2]=0;
	Rot[2][0]=Rot[2][1]=0;
	computeStiffnessMatrix(JKJt,tmp,tetrahedronInf.materialMatrix, tetrahedronInf.strainDisplacementTransposedMatrix,Rot);
	for(int i=0; i<12; i++)
	{
		for(int j=0; j<12; j++)
			stiffness[i*12+j]=JKJt(i,j);
	}
}

template <class DataTypes>
inline void CMTetrahedralCorotationalFEMForceField<DataTypes>::getElementStiffnessMatrix(Real* stiffness, const core::topology::BaseMeshTopology::Tetrahedron& t)
{
	const VecCoord X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

	Index a = t[0];
	Index b = t[1];
	Index c = t[2];
	Index d = t[3];

	Transformation R_0_1;
	computeRotationLarge( R_0_1, X0, a, b, c);

	MaterialStiffness	materialMatrix;
	StrainDisplacementTransposed	strainMatrix;
	helper::fixed_array<Coord,4> rotatedInitialElements;

	rotatedInitialElements[0] = R_0_1*X0[a];
	rotatedInitialElements[1] = R_0_1*X0[b];
	rotatedInitialElements[2] = R_0_1*X0[c];
	rotatedInitialElements[3] = R_0_1*X0[d];

	rotatedInitialElements[1] -= rotatedInitialElements[0];
	rotatedInitialElements[2] -= rotatedInitialElements[0];
	rotatedInitialElements[3] -= rotatedInitialElements[0];
	rotatedInitialElements[0] = Coord(0,0,0);

	computeMaterialStiffness(materialMatrix,a,b,c,d);
	computeStrainDisplacement(strainMatrix, rotatedInitialElements[0], rotatedInitialElements[1], rotatedInitialElements[2], rotatedInitialElements[3]);

	Transformation Rot;
	StiffnessMatrix JKJt,tmp;
	Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
	Rot[0][1]=Rot[0][2]=0;
	Rot[1][0]=Rot[1][2]=0;
	Rot[2][0]=Rot[2][1]=0;
	computeStiffnessMatrix(JKJt, tmp, materialMatrix, strainMatrix, Rot);
	for(int i=0; i<12; i++)
	{
		for(int j=0; j<12; j++)
			stiffness[i*12+j]=JKJt(i,j);
	}
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::initLarge(const VecCoord& X0, TetrahedronInformation& info)
{
	Transformation R_0_1;
	computeRotationLarge(R_0_1, X0, info.dofs[0], info.dofs[1], info.dofs[2]);
	info.initialTransformation = R_0_1;

	info.rotatedInitialElements[0] = R_0_1*(X0)[info.dofs[0]];
	info.rotatedInitialElements[1] = R_0_1*(X0)[info.dofs[1]];
	info.rotatedInitialElements[2] = R_0_1*(X0)[info.dofs[2]];
	info.rotatedInitialElements[3] = R_0_1*(X0)[info.dofs[3]];

	info.rotatedInitialElements[1] -= info.rotatedInitialElements[0];
	info.rotatedInitialElements[2] -= info.rotatedInitialElements[0];
	info.rotatedInitialElements[3] -= info.rotatedInitialElements[0];
	info.rotatedInitialElements[0] = Coord(0,0,0);

	computeStrainDisplacement(info.strainDisplacementTransposedMatrix,info.rotatedInitialElements[0], info.rotatedInitialElements[1],info.rotatedInitialElements[2],info.rotatedInitialElements[3] );
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceLarge( Vector& f, const Vector & p, TetrahedronInformation& info)
{
	// Rotation matrix (deformed and displaced Tetrahedron/world)
	Transformation R_0_2;
	computeRotationLarge( R_0_2, p, info.dofs[0], info.dofs[1], info.dofs[2]);
	info.rotation.transpose(R_0_2);

	// positions of the deformed and displaced Tetrahedron in its frame
	helper::fixed_array<Coord,4> deforme;
	deforme[0] = R_0_2*p[info.dofs[0]];
	deforme[1] = R_0_2*p[info.dofs[1]];
	deforme[2] = R_0_2*p[info.dofs[2]];
	deforme[3] = R_0_2*p[info.dofs[3]];

	deforme[1][0] -= deforme[0][0];
	deforme[2][0] -= deforme[0][0];
	deforme[2][1] -= deforme[0][1];
	deforme[3] -= deforme[0];

	// displacement
	Displacement D;
	D[0] = 0;
	D[1] = 0;
	D[2] = 0;
	D[3] = info.rotatedInitialElements[1][0] - deforme[1][0];
	D[4] = 0;
	D[5] = 0;
	D[6] = info.rotatedInitialElements[2][0] - deforme[2][0];
	D[7] = info.rotatedInitialElements[2][1] - deforme[2][1];
	D[8] = 0;
	D[9] = info.rotatedInitialElements[3][0] - deforme[3][0];
	D[10] = info.rotatedInitialElements[3][1] - deforme[3][1];
	D[11] =info.rotatedInitialElements[3][2] - deforme[3][2];

	Displacement F;
	if(_updateStiffnessMatrix.getValue())
	{
		StrainDisplacementTransposed& J = info.strainDisplacementTransposedMatrix;
		J[0][0] = J[1][3] = J[2][5]   = ( - deforme[2][1]*deforme[3][2] );
		J[1][1] = J[0][3] = J[2][4]   = ( deforme[2][0]*deforme[3][2] - deforme[1][0]*deforme[3][2] );
		J[2][2] = J[0][5] = J[1][4]   = ( deforme[2][1]*deforme[3][0] - deforme[2][0]*deforme[3][1] + deforme[1][0]*deforme[3][1] - deforme[1][0]*deforme[2][1] );

		J[3][0] = J[4][3] = J[5][5]   = ( deforme[2][1]*deforme[3][2] );
		J[4][1] = J[3][3] = J[5][4]  = ( - deforme[2][0]*deforme[3][2] );
		J[5][2] = J[3][5] = J[4][4]   = ( - deforme[2][1]*deforme[3][0] + deforme[2][0]*deforme[3][1] );

		J[7][1] = J[6][3] = J[8][4]  = ( deforme[1][0]*deforme[3][2] );
		J[8][2] = J[6][5] = J[7][4]   = ( - deforme[1][0]*deforme[3][1] );

		J[11][2] = J[9][5] = J[10][4] = ( deforme[1][0]*deforme[2][1] );
	}

	if(!_assembling.getValue())
	{
		// compute force on element
		computeForce( F, D, info.materialMatrix, info.strainDisplacementTransposedMatrix);
		for(int i=0; i<12; i+=3)
			f[info.dofs[i/3]] += info.rotation * Deriv(F[i], F[i+1], F[i+2]);
	}
	else
	{
		info.strainDisplacementTransposedMatrix[6][0] = 0;
		info.strainDisplacementTransposedMatrix[9][0] = 0;
		info.strainDisplacementTransposedMatrix[10][1] = 0;

		StiffnessMatrix RJKJt, RJKJtRt;
		computeStiffnessMatrix(RJKJt,RJKJtRt,info.materialMatrix, info.strainDisplacementTransposedMatrix,info.rotation);

		for(int i=0; i<12; ++i)
		{
			int row = info.dofs[i/3]*3+i%3;

			for(int j=0; j<12; ++j)
			{
				int col = info.dofs[j/3]*3+j%3;

				// search if the vertex is already take into account by another element
				typename CompressedValue::iterator result = _stiffnesses[row].end();
				for(typename CompressedValue::iterator it=_stiffnesses[row].begin(); it!=_stiffnesses[row].end()&&result==_stiffnesses[row].end(); ++it)
				{
					if( (*it).first == col )
					{
						result = it;
					}
				}

				if( result==_stiffnesses[row].end() )
				{
					_stiffnesses[row].push_back( Col_Value(col,RJKJtRt[i][j] )  );
				}
				else
				{
					(*result).second += RJKJtRt[i][j];
				}
			}
		}

		F = RJKJt*D;

		for(int i=0; i<12; i+=3)
			f[info.dofs[i/3]] += Deriv( F[i], F[i+1],  F[i+2] );
	}
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessLarge(Vector& f, const Vector& x, const TetrahedronInformation& info, SReal fact)
{
	Transformation R_0_2;
	R_0_2.transpose(info.rotation);

	Displacement X;
	Coord x_2;

	x_2 = R_0_2*x[info.dofs[0]];
	X[0] = x_2[0];
	X[1] = x_2[1];
	X[2] = x_2[2];

	x_2 = R_0_2*x[info.dofs[1]];
	X[3] = x_2[0];
	X[4] = x_2[1];
	X[5] = x_2[2];

	x_2 = R_0_2*x[info.dofs[2]];
	X[6] = x_2[0];
	X[7] = x_2[1];
	X[8] = x_2[2];

	x_2 = R_0_2*x[info.dofs[3]];
	X[9] = x_2[0];
	X[10] = x_2[1];
	X[11] = x_2[2];

	Displacement F;
	computeForce(F, X, info.materialMatrix, info.strainDisplacementTransposedMatrix, fact);
	for(int i=0; i<12; i+=3)
		f[info.dofs[i/3]] += info.rotation * Deriv(-F[i], -F[i+1], -F[i+2]);
}

//////////////////////////////////////////////////////////////////////
////////////////////  polar decomposition method  ////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::initPolar(const VecCoord& X0, TetrahedronInformation& info)
{
	Transformation A;
	A[0] = (X0)[info.dofs[1]]-(X0)[info.dofs[0]];
	A[1] = (X0)[info.dofs[2]]-(X0)[info.dofs[0]];
	A[2] = (X0)[info.dofs[3]]-(X0)[info.dofs[0]];
	info.initialTransformation = A;

	Transformation R_0_1;
	helper::Decompose<Real>::polarDecomposition(A, R_0_1);

	info.rotatedInitialElements[0] = R_0_1*(X0)[info.dofs[0]];
	info.rotatedInitialElements[1] = R_0_1*(X0)[info.dofs[1]];
	info.rotatedInitialElements[2] = R_0_1*(X0)[info.dofs[2]];
	info.rotatedInitialElements[3] = R_0_1*(X0)[info.dofs[3]];

	computeStrainDisplacement(info.strainDisplacementTransposedMatrix,info.rotatedInitialElements[0], info.rotatedInitialElements[1],info.rotatedInitialElements[2],info.rotatedInitialElements[3]);
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::accumulateForcePolar(Vector& f, const Vector & p, TetrahedronInformation& info)
{
	Transformation A;
	A[0] = p[info.dofs[1]] - p[info.dofs[0]];
	A[1] = p[info.dofs[2]] - p[info.dofs[0]];
	A[2] = p[info.dofs[3]] - p[info.dofs[0]];

	Transformation R_0_2;
	defaulttype::MatNoInit<3,3,Real> S;
	helper::Decompose<Real>::polarDecomposition(A, R_0_2);

	info.rotation.transpose( R_0_2 );

	// positions of the deformed and displaced Tetrahedre in its frame
	helper::fixed_array<Coord, 4> deforme;
	for(int i=0; i<4; ++i)
		deforme[i] = R_0_2 * p[info.dofs[i]];

	// displacement
	Displacement D;
	for (int i=0; i<12; ++i)
		D[i] = info.rotatedInitialElements[i/3][i%3] - deforme[i/3][i%3];

	Displacement F;
	if(_updateStiffnessMatrix.getValue())
	{
		// shape functions matrix
		computeStrainDisplacement(info.strainDisplacementTransposedMatrix, deforme[0], deforme[1], deforme[2], deforme[3]);
	}

	if(!_assembling.getValue())
	{
		computeForce( F, D, info.materialMatrix, info.strainDisplacementTransposedMatrix );
		for(int i=0; i<12; i+=3)
			f[info.dofs[i/3]] += info.rotation * Deriv( F[i], F[i+1],  F[i+2] );
	}
	else
	{
		serr << "TODO(TetrahedralCorotationalFEMForceField): support for assembling system matrix when using polar method."<<sendl;
	}
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessPolar(Vector& f, const Vector& x, const TetrahedronInformation& info, SReal fact )
{
	Transformation R_0_2;
	R_0_2.transpose(info.rotation);

	Displacement X;
	Coord x_2;

	x_2 = R_0_2*x[info.dofs[0]];
	X[0] = x_2[0];
	X[1] = x_2[1];
	X[2] = x_2[2];

	x_2 = R_0_2*x[info.dofs[1]];
	X[3] = x_2[0];
	X[4] = x_2[1];
	X[5] = x_2[2];

	x_2 = R_0_2*x[info.dofs[2]];
	X[6] = x_2[0];
	X[7] = x_2[1];
	X[8] = x_2[2];

	x_2 = R_0_2*x[info.dofs[3]];
	X[9] = x_2[0];
	X[10] = x_2[1];
	X[11] = x_2[2];

	Displacement F;
	computeForce(F, X, info.materialMatrix, info.strainDisplacementTransposedMatrix, fact);

	for(int i=0; i<12; i+=3)
		f[info.dofs[i/3]] -= info.rotation * Deriv(F[i], F[i+1], F[i+2]);
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
	if (!vparams->displayFlags().getShowForceFields()) return;
	if (!this->mstate) return;
	if (!f_drawing.getValue()) return;

	const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

	if (vparams->displayFlags().getShowWireFrame())
		vparams->drawTool()->setPolygonMode(0,true);

	std::vector< defaulttype::Vector3 > points[4];
	_topology->foreach_cell([&](Volume w)
	{
		const auto& t = _topology->get_dofs(w);

		Index a = t[0];
		Index b = t[1];
		Index c = t[2];
		Index d = t[3];
		Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
		Coord pa = (x[a]+center)*(Real)0.666667;
		Coord pb = (x[b]+center)*(Real)0.666667;
		Coord pc = (x[c]+center)*(Real)0.666667;
		Coord pd = (x[d]+center)*(Real)0.666667;

		points[0].push_back(pa);
		points[0].push_back(pb);
		points[0].push_back(pc);

		points[1].push_back(pb);
		points[1].push_back(pc);
		points[1].push_back(pd);

		points[2].push_back(pc);
		points[2].push_back(pd);
		points[2].push_back(pa);

		points[3].push_back(pd);
		points[3].push_back(pa);
		points[3].push_back(pb);
	});

	vparams->drawTool()->drawTriangles(points[0], drawColor1.getValue());
	vparams->drawTool()->drawTriangles(points[1], drawColor2.getValue());
	vparams->drawTool()->drawTriangles(points[2], drawColor3.getValue());
	vparams->drawTool()->drawTriangles(points[3], drawColor4.getValue());

	if (vparams->displayFlags().getShowWireFrame())
		vparams->drawTool()->setPolygonMode(0,false);
}

// TODO: Use CGoGN foreach instead of SeqTetrahedra
template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *mat, SReal k, unsigned int &offset)
{
	// Build Matrix Block for this ForceField
	unsigned int i,j,n1, n2, row, column, ROW, COLUMN;

	Transformation Rot;
	StiffnessMatrix JKJt,tmp;

	const auto& tetrahedronInf = _volumeAttribute.getValue();

	Index noeud1, noeud2;

	Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
	Rot[0][1]=Rot[0][2]=0;
	Rot[1][0]=Rot[1][2]=0;
	Rot[2][0]=Rot[2][1]=0;
	const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetras = _topology->getTetrahedra();
	for(int IT=0 ; IT != (int)tetras.size() ; ++IT)
	{
		if (method == SMALL)
			computeStiffnessMatrix(JKJt,tmp,tetrahedronInf[IT].materialMatrix,tetrahedronInf[IT].strainDisplacementTransposedMatrix,Rot);
		else
			computeStiffnessMatrix(JKJt,tmp,tetrahedronInf[IT].materialMatrix,tetrahedronInf[IT].strainDisplacementTransposedMatrix,tetrahedronInf[IT].rotation);
		const core::topology::BaseMeshTopology::Tetrahedron t=tetras[IT];

		// find index of node 1
		for (n1=0; n1<4; n1++)
		{
			noeud1 = t[n1];

			for(i=0; i<3; i++)
			{
				ROW = offset+3*noeud1+i;
				row = 3*n1+i;
				// find index of node 2
				for (n2=0; n2<4; n2++)
				{
					noeud2 = t[n2];

					for (j=0; j<3; j++)
					{
						COLUMN = offset+3*noeud2+j;
						column = 3*n2+j;
						mat->add(ROW, COLUMN, - tmp[row][column]*k);
					}
				}
			}
		}
	}
}

template<class DataTypes>
void CMTetrahedralCorotationalFEMForceField<DataTypes>::printStiffnessMatrix(Volume w)
{
	const auto& attribute = _volumeAttribute.getValue();
	const TetrahedronInformation& info = attribute[w.dart];

	Transformation Rot;
	StiffnessMatrix JKJt,tmp;

	Rot[0][0]=Rot[1][1]=Rot[2][2]=1;
	Rot[0][1]=Rot[0][2]=0;
	Rot[1][0]=Rot[1][2]=0;
	Rot[2][0]=Rot[2][1]=0;

	computeStiffnessMatrix(JKJt,tmp,info.materialMatrix,info.strainDisplacementTransposedMatrix,Rot);
}

} // namespace cm_forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CMTETRAHEDRALCOROTATIONALFEMFORCEFIELD_INL
