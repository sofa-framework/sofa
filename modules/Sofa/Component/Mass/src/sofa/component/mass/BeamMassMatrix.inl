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
#ifndef SOFA_COMPONENT_MASS_BEAMMASSMATRIX_INL
#define SOFA_COMPONENT_MASS_BEAMMASSMATRIX_INL

#include <sofa/component/mass/BeamMassMatrix.h>

namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class MassType>
BeamMassMatrix<DataTypes, MassType>::BeamMassMatrix()
	//: vertexMassInfo( initData(&vertexMassInfo, "vertexMass", "values of the particles masses on vertices") )
	: m_massDensity(initData(&m_massDensity, "massDensity", "mass density that allows to compute the  particles masses from a mesh topology and geometry.\nOnly used if > 0") )
	, d_r(initData(&d_r,"radius","radius of the sections"))
	, d_innerR(initData(&d_innerR,"innerRadius","inner radius of the sections for hollow beams"))
	, d_L(initData(&d_L,"length","length of the sections"))
{

}

template <class DataTypes, class MassType>
BeamMassMatrix<DataTypes, MassType>::~BeamMassMatrix()
{ }


template <class DataTypes, class MassType>
void BeamMassMatrix<DataTypes, MassType>::init()
{
	Inherited::init();

	_topology = this->getContext()->getMeshTopology();

	if(_topology == nullptr)
		return;

	this->getContext()->get(edgeGeo);

	std::size_t size = _topology->getEdges().size();

	if(m_massDensity.getValue().size() == 1)
	{
		Real rho = m_massDensity.getValue()[0];
		helper::WriteAccessor<Data<sofa::helper::vector<Real> > > massDensity = m_massDensity;
		massDensity.clear();
		massDensity.resize(size, rho);
	}

	_Iy.resize(size);
	_Iz.resize(size);
	_J.resize(size);
	_A.resize(size);

	M00.resize(size);
	M11.resize(size);
	M22.resize(size);
	M33.resize(size);
	M44.resize(size);
	M55.resize(size);
	M66.resize(size);
	M77.resize(size);
	M88.resize(size);
	M99.resize(size);
	M1010.resize(size);
	M1111.resize(size);
	M24.resize(size);
	M15.resize(size);
	M06.resize(size);
	M17.resize(size);
	M57.resize(size);
	M28.resize(size);
	M48.resize(size);
	M39.resize(size);
	M210.resize(size);
	M410.resize(size);
	M810.resize(size);
	M111.resize(size);
	M511.resize(size);
	M711.resize(size);

	for (std::size_t i = 0; i<size; ++i)
	{
		const Real r = d_r.getValue()[i];
		const Real rInner = d_innerR.getValue()[i];

		_Iz[i] = M_PI*(r*r*r*r - rInner*rInner*rInner*rInner)/4.0;
		_Iy[i] = _Iz[i] ;
		_J[i] = _Iz[i] + _Iy[i];
		_A[i] = M_PI*(r*r - rInner*rInner);
	}

	massInitialization();
}

template <class DataTypes, class MassType>
void BeamMassMatrix<DataTypes, MassType>::reinit()
{ }

template <class DataTypes, class MassType>
void BeamMassMatrix<DataTypes, MassType>::clear()
{ }

template <class DataTypes, class MassType>
void BeamMassMatrix<DataTypes, MassType>::massInitialization()
{
	/// prepare to store info in the vertex array
	//helper::vector<MassType>& my_vertexMassInfo = *vertexMassInfo.beginEdit();

	//my_vertexMassInfo.resize(ndof);

	size_t nbEdges=_topology->getNbEdges();
	size_t v0, v1;
	Real _L;

	for (unsigned int j=0; j < nbEdges; ++j)
	{
		v0=_topology->getEdge(j)[0];
		v1=_topology->getEdge(j)[1];

		_L = d_L.getValue()[j];

		const Real rhoAL = m_massDensity.getValue()[j] * _A[j] * _L;

		M00[j] = rhoAL * 1./3.;
		M11[j] = rhoAL * ((13./35.) + (6.*_Iz[j]) / (5.*_A[j]*_L*_L)) ;
		M22[j] = rhoAL * ((13./35.) + (6.*_Iy[j]) / (5.*_A[j]*_L*_L)) ;
		M33[j] = rhoAL * (_J[j] / (3.* _A[j]));
		M44[j] = rhoAL * ((_L*_L) / 105. + (2. * _Iy[j]) / (15. * _A[j]));
		M55[j] = rhoAL * ((_L*_L) / 105. + (2. * _Iz[j]) / (15. * _A[j]));

		M66[j] = rhoAL * 1./3.;
		M77[j] = rhoAL * ((13./35.) + (6.*_Iz[j]) / (5.*_A[j]*_L*_L)) ;
		M88[j] = rhoAL * ((13./35.) + (6.*_Iy[j]) / (5.*_A[j]*_L*_L)) ;
		M99[j] = rhoAL * (_J[j] / (3.* _A[j]));
		M1010[j] = rhoAL * ((_L*_L) / 105. + (2. * _Iy[j]) / (15. * _A[j]));
		M1111[j] = rhoAL * ((_L*_L) / 105. + (2. * _Iz[j]) / (15. * _A[j]));

		M24[j] = rhoAL * -(11. * _L) / 210. - _Iy[j] / (10. * _A[j]* _L);

		M15[j] = rhoAL * (11. * _L) / 210. + _Iz[j] / (10. * _A[j]* _L);

		M06[j] = rhoAL * 1./6.;
		M17[j] = rhoAL * 9. / 70. - (6. * _Iz[j]) / (5. * _A[j] * _L * _L);
		M57[j] = rhoAL * (13. * _L) / 420. - (_Iz[j] / (10. * _A[j]* _L));
		M28[j] = rhoAL * 9. / 70. - (6. * _Iy[j]) / (5. * _A[j] * _L * _L);
		M48[j] = rhoAL * (- (13. * _L) / 420. + (_Iy[j] / (10. * _A[j]* _L)));
		M39[j] = rhoAL * _J[j] / (6. * _A[j]);
		M210[j] = rhoAL * (13. * _L) / 420. - (_Iy[j] / (10. * _A[j] * _L));
		M410[j] = rhoAL * (- (_L * _L) / 140. - (_Iy[j] / (30. * _A[j])));
		M810[j] = rhoAL * (11. * _L) / 210. + _Iy[j] / (10. * _A[j] * _L);
		M111[j] = rhoAL * (-(13. * _L) / 420. + (_Iz[j] / (10. * _A[j] * _L)));
		M511[j] = rhoAL * (- (_L * _L) / 140. - (_Iz[j] / (30. * _A[j])));
		M711[j] = rhoAL * (-(11. * _L) / 210. - _Iz[j] / (10. * _A[j] * _L));
	}

}

// -- Mass interface
template <class DataTypes, class MassType>
void BeamMassMatrix<DataTypes, MassType>::addMDx(const core::MechanicalParams*, DataVecDeriv& vres, const DataVecDeriv& vdx, SReal factor)
{
	helper::WriteAccessor< DataVecDeriv > res = vres;
	helper::ReadAccessor< DataVecDeriv > dx = vdx;

	size_t nbEdges=_topology->getNbEdges();
	size_t v0, v1;

	for (unsigned int j=0; j < nbEdges; ++j)
	{
		v0=_topology->getEdge(j)[0];
		v1=_topology->getEdge(j)[1];

		res[v0][0] += dx[v0][0] * M00[j];
		res[v0][1] += dx[v0][1] * M11[j] ;
		res[v0][2] += dx[v0][2] * M22[j] ;
		res[v0][3] += dx[v0][3] * M33[j];
		res[v0][4] += dx[v0][4] * M44[j];
		res[v0][5] += dx[v0][5] * M55[j];

		res[v1][0] += dx[v1][0] * M66[j];
		res[v1][1] += dx[v1][1] * M77[j];
		res[v1][2] += dx[v1][2] * M88[j];
		res[v1][3] += dx[v1][3] * M99[j];
		res[v1][4] += dx[v1][4] * M1010[j];
		res[v1][5] += dx[v1][5] * M1111[j];

		res[v0][2] += dx[v0][2] * M24[j];
		res[v0][4] += dx[v0][4] * M24[j];

		res[v0][1] += dx[v0][1] * M15[j];
		res[v0][5] += dx[v0][5] * M15[j];

		res[v0][0] += dx[v1][0] * M06[j];
		res[v1][0] += dx[v0][0] * M06[j];

		res[v0][1] += dx[v1][1] * M17[j];
		res[v1][1] += dx[v0][1] * M17[j];

		res[v0][5] += dx[v1][1] * M57[j];
		res[v1][1] += dx[v0][5] * M57[j];

		res[v0][2] += dx[v1][2] * M28[j];
		res[v1][2] += dx[v0][2] * M28[j];

		res[v0][4] += dx[v1][2] * M48[j];
		res[v1][2] += dx[v0][4] * M48[j];

		res[v0][3] += dx[v1][3] * M39[j];
		res[v1][3] += dx[v0][3] * M39[j];

		res[v0][2] += dx[v1][4] * M210[j];
		res[v1][4] += dx[v0][2] * M210[j];

		res[v0][4] += dx[v1][4] * M410[j];
		res[v1][4] += dx[v0][4] * M410[j];

		res[v1][2] += dx[v1][4] * M810[j];
		res[v1][4] += dx[v1][2] * M810[j];

		res[v0][1] += dx[v1][5] * M111[j];
		res[v1][5] += dx[v0][1] * M111[j];

		res[v0][5] += dx[v1][5] * M511[j];
		res[v1][5] += dx[v0][5] * M511[j];

		res[v1][1] += dx[v1][5] * M711[j];
		res[v1][5] += dx[v1][1] * M711[j];
	}
}

template <class DataTypes, class MassType>
void BeamMassMatrix<DataTypes, MassType>::addForce(const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& , const DataVecDeriv& )
{

}

} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MASS_BEAMMASSMATRIX_INL
