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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <fstream>

namespace sofa
{

namespace component
{

namespace topology
{
	
const unsigned int verticesInHexahedronArray[2][2][2]=  {{{0,4},{3,7}},{{1,5},{2,6}}};


template< class DataTypes>
NumericalIntegrationDescriptor<typename HexahedronSetGeometryAlgorithms< DataTypes >::Real,3> &HexahedronSetGeometryAlgorithms< DataTypes >::getHexahedronNumericalIntegrationDescriptor()
{
	// initialize the cubature table only if needed.
	if (initializedHexahedronCubatureTables==false) {
		initializedHexahedronCubatureTables=true;
		defineHexahedronCubaturePoints();
	}
	return hexahedronNumericalIntegration;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::defineHexahedronCubaturePoints() {
	typedef typename NumericalIntegrationDescriptor<typename HexahedronSetGeometryAlgorithms< DataTypes >::Real,3>::QuadraturePoint QuadraturePoint;
	typedef typename NumericalIntegrationDescriptor<typename HexahedronSetGeometryAlgorithms< DataTypes >::Real,3>::BarycentricCoordinatesType BarycentricCoordinatesType;
	// Gauss method
	typename NumericalIntegrationDescriptor<typename HexahedronSetGeometryAlgorithms< DataTypes >::Real,3>::QuadratureMethod m=NumericalIntegrationDescriptor<typename HexahedronSetGeometryAlgorithms< DataTypes >::Real,3>::GAUSS_LEGENDRE_METHOD;
	typename NumericalIntegrationDescriptor<typename HexahedronSetGeometryAlgorithms< DataTypes >::Real,3>::QuadraturePointArray qpa;
	typename NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1>::QuadraturePointArray qpa1D;

	BarycentricCoordinatesType v;
	Real w;


	NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1> &nide=this->getEdgeNumericalIntegrationDescriptor();

	/// create gauss points as tensor product of Gauss Legendre points in 1D
	/// create integration method up to order 8 (could go up to 12 if needed) where the number of gauss points is the cube of the number of 1D Gauss points
	size_t o,i,j,k;
	for (o=1;o<8;++o) {
		qpa.clear();
		qpa1D=nide.getQuadratureMethod(NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1>::GAUSS_LEGENDRE_METHOD,o);
		for (i=0;i<qpa1D.size();++i) {
			for (j=0;j<qpa1D.size();++j) {
				for (k=0;k<qpa1D.size();++k) {
					v=BarycentricCoordinatesType(qpa1D[i].first[0],qpa1D[j].first[0],qpa1D[k].first[0]);
					w=qpa1D[i].second*qpa1D[j].second*qpa1D[k].second;
					qpa.push_back(QuadraturePoint(v,(Real)w));
				}
			}
		}
		hexahedronNumericalIntegration.addQuadratureMethod(m,o,qpa);
	}
	/*
	/// consider non tensor product rules : taken from getfem++ file Hexahedron_5.im
	 m=NumericalIntegrationDescriptor<typename HexahedronSetGeometryAlgorithms< DataTypes >::Real,3>::GAUSS_CUBE_METHOD;
	/// integration with  accuracy of order 5 with 14 gauss points.
	Real a=0.8979112128771107316322744102380675;
	Real b=0.5;
	w=0.1108033240997229916897506925207755;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		v[i]=a;
		qpa.push_back(QuadraturePoint(v,w));
		v[i]=1-a;
		qpa.push_back(QuadraturePoint(v,w));
	}
	a=0.8793934553196640731345171390561335;
	w=0.0418975069252077562326869806094182;
	for (i=0;i<2;++i) {
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				/// barycentric coordinates are either a or 1-a
				v=BarycentricCoordinatesType(a+i*(1-2*a),a+j*(1-2*a),a+k*(1-2*a));
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	hexahedronNumericalIntegration.addQuadratureMethod(m,5,qpa);


	/// consider non tensor product rules : taken from getfem++ file Hexahedron_9.im
	/// integration with  accuracy of order 9 with 58 gauss points.

	/// 6 points
	a=.8068407347958544969174424448702780;
	b=0.5;
	w=.0541593744687068178762288491492902;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		v[i]=a;
		qpa.push_back(QuadraturePoint(v,w));
		v[i]=1-a;
		qpa.push_back(QuadraturePoint(v,w));
	}
	// 12 points
	a=0.9388435616288391432433878794971660;
	b=0.5;
	w=0.0114737257670222052714055736149557;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		for (j=0;j<2;++j) {
			v[(i+1)%3]=a+j*(1-2*a);
			for (k=0;k<2;++k) {
				v[(i+2)%3]=a+k*(1-2*a);
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	/// 8 points
	a=0.7820554035100150271333094993315360;
	w=0.0248574797680029375401085898232011;
	for (i=0;i<2;++i) {
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				/// barycentric coordinates are either a or 1-a
				v=BarycentricCoordinatesType(a+i*(1-2*a),a+j*(1-2*a),a+k*(1-2*a));
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	/// 8 points
	a=0.9350498923309879588075319044319620;
	w=0.0062685994124186287334314359655827;
	for (i=0;i<2;++i) {
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				/// barycentric coordinates are either a or 1-a
				v=BarycentricCoordinatesType(a+i*(1-2*a),a+j*(1-2*a),a+k*(1-2*a));
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	/// 24 points 
	a=0.7161339513154310822080124307584715;
	b=0.9692652109323358726644884348015390;
	w=0.0120146004391716708040599923089382;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		for (j=0;j<2;++j) {
			v[(i+1)%3]=a+j*(1-2*a);
			for (k=0;k<2;++k) {
				v[i]=b;
				v[(i+2)%3]=a+k*(1-2*a);
				qpa.push_back(QuadraturePoint(v,w));
				v[i]=1-b;
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	hexahedronNumericalIntegration.addQuadratureMethod(m,9,qpa);
	
	
	/// consider non tensor product rules : taken from getfem++ file Hexahedron_11.im
	/// integration with  accuracy of order 11 with 90 gauss points.

	/// 6 points
	a=.9063071670498132481961877986898720;
	b=0.5;
	w=.0253096342016000238231671413708773;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		v[i]=a;
		qpa.push_back(QuadraturePoint(v,w));
		v[i]=1-a;
		qpa.push_back(QuadraturePoint(v,w));
	}
	// 12 points
	a=.8673341434985040086731923849337745;
	b=0.5;
	w=0.0181499182325144622865632250992823;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		for (j=0;j<2;++j) {
			v[(i+1)%3]=a+j*(1-2*a);
			for (k=0;k<2;++k) {
				v[(i+2)%3]=a+k*(1-2*a);
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	/// 8 points
	a=.6566967022580273605228866152789755;
	w=.0269990056568711411641833332980551;
	for (i=0;i<2;++i) {
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				/// barycentric coordinates are either a or 1-a
				v=BarycentricCoordinatesType(a+i*(1-2*a),a+j*(1-2*a),a+k*(1-2*a));
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	/// 8 points
	a=.8008376320991313508172065028926585;
	w= .0146922934945570350487414755013352;
	for (i=0;i<2;++i) {
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				/// barycentric coordinates are either a or 1-a
				v=BarycentricCoordinatesType(a+i*(1-2*a),a+j*(1-2*a),a+k*(1-2*a));
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
		/// 8 points
	a=.9277278805088799923375457353451730;
	w= .0055804890098536552051251442852662;
	for (i=0;i<2;++i) {
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				/// barycentric coordinates are either a or 1-a
				v=BarycentricCoordinatesType(a+i*(1-2*a),a+j*(1-2*a),a+k*(1-2*a));
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	/// 24 points 
	a=.9706224286053016319555750788155670;
	b=.6769514072983150674551564354064455;
	w=.0028267870173527355278995288336230;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		for (j=0;j<2;++j) {
			v[(i+1)%3]=a+j*(1-2*a);
			for (k=0;k<2;++k) {
				v[i]=b;
				v[(i+2)%3]=a+k*(1-2*a);
				qpa.push_back(QuadraturePoint(v,w));
				v[i]=1-b;
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	/// 24 points 
	a=.7253999675572547151889421728651345;
	b=.9825498327563551314651409115626720;
	w=.0076802492622294169003437555791307;
	for (i=0;i<3;++i) {
		v=BarycentricCoordinatesType(b,b,b);
		for (j=0;j<2;++j) {
			v[(i+1)%3]=a+j*(1-2*a);
			for (k=0;k<2;++k) {
				v[i]=b;
				v[(i+2)%3]=a+k*(1-2*a);
				qpa.push_back(QuadraturePoint(v,w));
				v[i]=1-b;
				qpa.push_back(QuadraturePoint(v,w));
			}
		}
	}
	hexahedronNumericalIntegration.addQuadratureMethod(m,11,qpa);
	*/
	
}
template< class DataTypes>
bool HexahedronSetGeometryAlgorithms< DataTypes >::isHexahedronAffine(const HexaID hx, const VecCoord& p, const Real tolerance) const
{
	/// check that the hexahedron is a parallelepiped returns true if it is the case and false otherwise.
	/// given 4 points of binary coordinates 000 010 100 001 checks that the 4 other points are translated versions
	const Hexahedron &h = this->m_topology->getHexahedron(hx);
	Coord dpos;
	dpos=(p[h[verticesInHexahedronArray[1][0][1]]]-p[h[verticesInHexahedronArray[1][0][0]]])-(p[h[verticesInHexahedronArray[0][0][1]]]-p[h[verticesInHexahedronArray[0][0][0]]]);
	if (dpos.norm()>tolerance)
		return false;
	else {
		dpos=(p[h[verticesInHexahedronArray[1][1][0]]]-p[h[verticesInHexahedronArray[1][0][0]]])-(p[h[verticesInHexahedronArray[0][1][0]]]-p[h[verticesInHexahedronArray[0][0][0]]]);
		if (dpos.norm()>tolerance)
			return false;
		else {
			dpos=(p[h[verticesInHexahedronArray[1][0][1]]]-p[h[verticesInHexahedronArray[0][0][1]]])-(p[h[verticesInHexahedronArray[0][0][1]]]-p[h[verticesInHexahedronArray[0][0][0]]]);
			if (dpos.norm()>tolerance)
				return false;
			else {
				dpos=(p[h[verticesInHexahedronArray[1][1][1]]]-p[h[verticesInHexahedronArray[0][1][1]]])-(p[h[verticesInHexahedronArray[1][0][1]]]-p[h[verticesInHexahedronArray[0][0][1]]]);
				if (dpos.norm()>tolerance)
					return false;
				else
					return true;
			}
		}
	}
}
template< class DataTypes>
typename  DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeShapeFunction(const LocalCoord nc,const HexahedronBinaryIndex bi) const 
{
	return((bi[0] ? nc[0] : 1-nc[0])*(bi[1] ? nc[1] : 1-nc[1])*(bi[2] ? nc[2] : 1-nc[2]));
}
template< class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms< DataTypes >::computeNodalValue(const HexaID hx,const LocalCoord nc,const VecCoord& p) const
{
	 const Hexahedron &h = this->m_topology->getHexahedron(hx);
	 size_t i,j,k;
	 Coord pos[8];
	 for (i=0;i<8;++i) 
		 pos[i]=p[h[i]];
	 Coord res;

	 for (i=0;i<2;++i) {
		 for (j=0;j<2;++j) {
			 for (k=0;k<2;++k) {
				 res+= (i ? nc[0] : 1-nc[0])*(j ? nc[1] : 1-nc[1])*(k ? nc[2] : 1-nc[2])*pos[h[verticesInHexahedronArray[i][j][k]]];
			 }
		 }
	 }
/*
    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz)); */

    return res;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::computePositionDerivative(const HexaID hx,const LocalCoord nc,const VecCoord& p,  Coord dpos[3]) const
{
	 const Hexahedron &h = this->m_topology->getHexahedron(hx);
	 size_t i,j,k;
	 size_t ind[3];

	 Coord pos[8];
	 for (i=0;i<8;++i) 
		 pos[i]=p[h[i]];
	 Coord res;

	 for (i=0;i<3;++i) {
		 Coord pos0,pos1;
		 for (j=0;j<2;++j) {
			 for (k=0;k<2;++k) {
				 ind[i]=1;
				 ind[(i+1)%3]=j;
				 ind[(i+2)%3]=k;
				 pos1+= (j ? nc[(i+1)%3] : 1-nc[(i+1)%3])*(k ? nc[(i+2)%3] : 1-nc[(i+2)%3])*p[h[verticesInHexahedronArray[ind[0]][ind[1]][ind[2]]]];
				 ind[i]=0;
				 pos0+= (j ? nc[(i+1)%3] : 1-nc[(i+1)%3])*(k ? nc[(i+2)%3] : 1-nc[(i+2)%3])*p[h[verticesInHexahedronArray[ind[0]][ind[1]][ind[2]]]];
			 }
		 }
		 dpos[i]=pos1-pos0;
	 }

}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeJacobian(const HexaID hx,const LocalCoord nc,const VecCoord& p) const
{
	Coord dpos[3];
	this->computePositionDerivative(hx,nc,p,dpos);
	return (tripleProduct(dpos[0],dpos[1],dpos[2]));
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronAABB(const HexaID h, Coord& minCoord, Coord& maxCoord) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<3; ++i)
    {
        minCoord[i] = std::min( std::min(std::min(p[t[0]][i], p[t[1]][i]), std::min(p[t[2]][i], p[t[3]][i])),
                std::min(std::min(p[t[4]][i], p[t[5]][i]), std::min(p[t[6]][i], p[t[7]][i])));

        maxCoord[i] = std::max( std::max(std::max(p[t[0]][i], p[t[1]][i]), std::max(p[t[2]][i], p[t[3]][i])),
                std::max(std::max(p[t[4]][i], p[t[5]][i]), std::max(p[t[6]][i], p[t[7]][i])));
    }
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronRestAABB(const HexaID h, Coord& minCoord, Coord& maxCoord) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    for(unsigned int i=0; i<3; ++i)
    {
        minCoord[i] = std::min( std::min(std::min(p[t[0]][i], p[t[1]][i]), std::min(p[t[2]][i], p[t[3]][i])),
                std::min(std::min(p[t[4]][i], p[t[5]][i]), std::min(p[t[6]][i], p[t[7]][i])));

        maxCoord[i] = std::max( std::max(std::max(p[t[0]][i], p[t[1]][i]), std::max(p[t[2]][i], p[t[3]][i])),
                std::max(std::max(p[t[4]][i], p[t[5]][i]), std::max(p[t[6]][i], p[t[7]][i])));
    }
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronCenter(const HexaID h) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]] + p[t[4]] + p[t[5]] + p[t[6]] + p[t[7]]) * (Real) 0.125;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronRestCenter(const HexaID h) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]] + p[t[4]] + p[t[5]] + p[t[6]] + p[t[7]]) * (Real) 0.125;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::getHexahedronVertexCoordinates(const HexaID h, Coord pnt[8]) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<8; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::getRestHexahedronVertexCoordinates(const HexaID h, Coord pnt[8]) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    for(unsigned int i=0; i<8; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getRestPointPositionInHexahedron(const HexaID h,
        const Real baryC[3]) const
{
    Coord	p[8];
    getRestHexahedronVertexCoordinates(h, p);

    const Real &fx = baryC[0];
    const Real &fy = baryC[1];
    const Real &fz = baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getRestPointPositionInHexahedron(const HexaID h,
        const sofa::defaulttype::Vector3& baryC) const
{
    Coord	p[8];
    getRestHexahedronVertexCoordinates(h, p);

    const Real fx = (Real) baryC[0];
    const Real fy = (Real) baryC[1];
    const Real fz = (Real) baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getPointPositionInHexahedron(const HexaID h,
        const Real baryC[3]) const
{
    Coord	p[8];
    getHexahedronVertexCoordinates(h, p);

    const Real &fx = baryC[0];
    const Real &fy = baryC[1];
    const Real &fz = baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getPointPositionInHexahedron(const HexaID h,
        const sofa::defaulttype::Vector3& baryC) const
{
    Coord	p[8];
    getHexahedronVertexCoordinates(h, p);

    const Real fx = (Real) baryC[0];
    const Real fy = (Real) baryC[1];
    const Real fz = (Real) baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
sofa::defaulttype::Vector3 HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronRestBarycentricCoeficients(const HexaID h,
        const Coord& pos) const
{
    Coord	p[8];
    getRestHexahedronVertexCoordinates(h, p);

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    sofa::defaulttype::Vector3 origin, p1, p3, p4, pnt;
    for( unsigned int w=0 ; w<max_spatial_dimensions ; ++w )
    {
        origin[w] = p[0][w];
        p1[w] = p[1][w];
        p3[w] = p[3][w];
        p4[w] = p[4][w];
        pnt[w] = pos[w];
    }

    sofa::defaulttype::Mat3x3d		m, mt, base;
    m[0] = p1-origin;
    m[1] = p3-origin;
    m[2] = p4-origin;
    mt.transpose(m);
    base.invert(mt);

    return base * (pnt - origin);
}

template<class DataTypes>
sofa::defaulttype::Vector3 HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronBarycentricCoeficients(const HexaID h,
        const Coord& pos) const
{
    // Warning: this is only correct if the hexahedron is not deformed
    // as only 3 perpendicular edges are considered as a base
    // other edges are assumed to be parallel to the respective base edge (and have the same length)

    Coord	p[8];
    getHexahedronVertexCoordinates(h, p);

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    sofa::defaulttype::Vector3 origin, p1, p3, p4, pnt;
    for( unsigned int w=0 ; w<max_spatial_dimensions ; ++w )
    {
        origin[w] = p[0][w];
        p1[w] = p[1][w];
        p3[w] = p[3][w];
        p4[w] = p[4][w];
        pnt[w] = pos[w];
    }

    sofa::defaulttype::Mat3x3d		m, mt, base;
    m[0] = p1-origin;
    m[1] = p3-origin;
    m[2] = p4-origin;
    mt.transpose(m);
    base.invert(mt);

    return base * (pnt - origin);
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeElementDistanceMeasure(const HexaID h, const Coord pos) const
{
    typedef typename DataTypes::Real Real;

    const sofa::defaulttype::Vector3 v = computeHexahedronBarycentricCoeficients(h, pos);

    Real d = (Real) std::max(std::max(-v[0], -v[1]), std::max(std::max(-v[2], v[0]-1), std::max(v[1]-1, v[2]-1)));

    if(d>0)
        d = (pos - computeHexahedronCenter(h)).norm2();

    return d;
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeElementRestDistanceMeasure(const HexaID h, const Coord pos) const
{
    typedef typename DataTypes::Real Real;

    const sofa::defaulttype::Vector3 v = computeHexahedronRestBarycentricCoeficients(h, pos);

    Real d = (Real) std::max(std::max(-v[0], -v[1]), std::max(std::max(-v[2], v[0]-1), std::max(v[1]-1, v[2]-1)));

    if(d>0)
        d = (pos - computeHexahedronRestCenter(h)).norm2();

    return d;
}

template< class DataTypes>
int HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElement(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    int index=-1;
    distance = 1e10;

    for(int c=0; c<this->m_topology->getNbHexahedra(); ++c)
    {
        const Real d = computeElementDistanceMeasure(c, pos);

        if(d<distance)
        {
            distance = d;
            index = c;
        }
    }

    if(index != -1)
        baryC = computeHexahedronBarycentricCoeficients(index, pos);

    return index;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElements(const VecCoord& pos,
        helper::vector<int>& elem,
        helper::vector<defaulttype::Vector3>& baryC,
        helper::vector<Real>& dist) const
{
    for(unsigned int i=0; i<pos.size(); ++i)
    {
        elem[i] = findNearestElement(pos[i], baryC[i], dist[i]);
    }
}

template< class DataTypes>
int HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    int index=-1;
    distance = 1e10;

    for(int c=0; c<this->m_topology->getNbHexahedra(); ++c)
    {
        const Real d = computeElementRestDistanceMeasure(c, pos);

        if(d<distance)
        {
            distance = d;
            index = c;
        }
    }

    if(index != -1)
        baryC = computeHexahedronRestBarycentricCoeficients(index, pos);

    return index;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElementsInRestPos( const VecCoord& pos, helper::vector<int>& elem, helper::vector<defaulttype::Vector3>& baryC, helper::vector<Real>& dist) const
{
    for(unsigned int i=0; i<pos.size(); ++i)
    {
        elem[i] = findNearestElementInRestPos(pos[i], baryC[i], dist[i]);
    }
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronVolume( const HexaID hexa) const
{
 const Hexahedron &h = this->m_topology->getHexahedron(hexa);
    const VecCoord& p = (this->object->read(core::ConstVecCoordId::position())->getValue());
	Coord dp[3];
	unsigned char i,j,k,ind[3];
	Real volume;
	for (i=0;i<3;++i) {
		dp[i]=Coord();
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				ind[i]=1;
				ind[(i+1)%3]=j;
				ind[(i+2)%3]=k;
				dp[i]+=p[h[verticesInHexahedronArray[ind[0]][ind[1]][ind[2]]]];
				ind[i]=0;
				dp[i]-=p[h[verticesInHexahedronArray[ind[0]][ind[1]][ind[2]]]];
			}
		}
	}
	volume=tripleProduct(dp[0],dp[1],dp[2])/48.0f;
	dp[0]=p[h[verticesInHexahedronArray[0][1][1]]]-p[h[verticesInHexahedronArray[0][0][0]]];
	dp[1]=p[h[verticesInHexahedronArray[1][0][1]]]-p[h[verticesInHexahedronArray[0][0][0]]];
	dp[2]=p[h[verticesInHexahedronArray[1][1][0]]]-p[h[verticesInHexahedronArray[0][0][0]]];
	volume-=tripleProduct(dp[0],dp[1],dp[2])/12.0f;
	dp[0]=p[h[verticesInHexahedronArray[1][0][0]]]-p[h[verticesInHexahedronArray[1][1][1]]];
	dp[1]=p[h[verticesInHexahedronArray[0][1][0]]]-p[h[verticesInHexahedronArray[1][1][1]]];
	dp[2]=p[h[verticesInHexahedronArray[0][0][1]]]-p[h[verticesInHexahedronArray[1][1][1]]];
	volume+=tripleProduct(dp[0],dp[1],dp[2])/12.0f;
    return volume;
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeRestHexahedronVolume( const HexaID hexa) const
{
    const Hexahedron &h = this->m_topology->getHexahedron(hexa);
    const VecCoord& p =  (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
	Coord dp[3];
	size_t i,j,k,ind[3];
	Real volume;
	for (i=0;i<3;++i) {
		dp[i]=Coord();
		for (j=0;j<2;++j) {
			for (k=0;k<2;++k) {
				ind[i]=1;
				ind[(i+1)%3]=j;
				ind[(i+2)%3]=k;
				dp[i]+=p[h[verticesInHexahedronArray[ind[0]][ind[1]][ind[2]]]];
				ind[i]=0;
				dp[i]-=p[h[verticesInHexahedronArray[ind[0]][ind[1]][ind[2]]]];
			}
		}
	}
	volume=tripleProduct(dp[0],dp[1],dp[2])/48.0f;
	dp[0]=p[h[verticesInHexahedronArray[0][1][1]]]-p[h[verticesInHexahedronArray[0][0][0]]];
	dp[1]=p[h[verticesInHexahedronArray[1][0][1]]]-p[h[verticesInHexahedronArray[0][0][0]]];
	dp[2]=p[h[verticesInHexahedronArray[1][1][0]]]-p[h[verticesInHexahedronArray[0][0][0]]];
	volume-=tripleProduct(dp[0],dp[1],dp[2])/12.0f;
	dp[0]=p[h[verticesInHexahedronArray[1][0][0]]]-p[h[verticesInHexahedronArray[1][1][1]]];
	dp[1]=p[h[verticesInHexahedronArray[0][1][0]]]-p[h[verticesInHexahedronArray[1][1][1]]];
	dp[2]=p[h[verticesInHexahedronArray[0][0][1]]]-p[h[verticesInHexahedronArray[1][1][1]]];
	volume+=tripleProduct(dp[0],dp[1],dp[2])/12.0f;
    return volume;
}

template<class DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronVolume( BasicArrayInterface<Real> &ai) const
{
    //const sofa::helper::vector<Hexahedron> &ta=this->m_topology->getHexahedra();
    //const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    for(int i=0; i<this->m_topology->getNbHexahedra(); ++i)
    {
        //const Hexahedron &t=this->m_topology->getHexahedron(i); //ta[i];
        ai[i]=(Real)(0.0); /// @todo : implementation of computeHexahedronVolume
    }
}

/// Write the current mesh into a msh file
template <typename DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::writeMSHfile(const char *filename) const
{
    std::ofstream myfile;
    myfile.open (filename);

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const size_t numVertices = vect_c.size();

    myfile << "$NOD\n";
    myfile << numVertices <<"\n";

    for(size_t i=0; i<numVertices; ++i)
    {
        double x = (double) vect_c[i][0];
        double y = (double) vect_c[i][1];
        double z = (double) vect_c[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    const sofa::helper::vector<Hexahedron> hea = this->m_topology->getHexahedra();

    myfile << hea.size() <<"\n";

    for(unsigned int i=0; i<hea.size(); ++i)
    {
        myfile << i+1 << " 5 1 1 8 " << hea[i][4]+1 << " " << hea[i][5]+1 << " "
                << hea[i][1]+1 << " " << hea[i][0]+1 << " "
                << hea[i][7]+1 << " " << hea[i][6]+1 << " "
                << hea[i][2]+1 << " " << hea[i][3]+1 << "\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}

template<class DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    QuadSetGeometryAlgorithms<DataTypes>::draw(vparams);

    // Draw Hexa indices
    if (d_showHexaIndices.getValue())
    {

        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
        const sofa::defaulttype::Vec3f& color = d_drawColorHexahedra.getValue();
        sofa::defaulttype::Vec4f color4(color[0], color[1], color[2], 1.0);

        float scale = this->getIndicesScale();

        //for hexa:
        scale = scale/2;

        const sofa::helper::vector<Hexahedron> &hexaArray = this->m_topology->getHexahedra();

        helper::vector<defaulttype::Vector3> positions;
        for (unsigned int i =0; i<hexaArray.size(); i++)
        {

            Hexahedron the_hexa = hexaArray[i];
            sofa::defaulttype::Vec3f center;

            for (unsigned int j = 0; j<8; j++)
            {
                defaulttype::Vector3 vertex; vertex = DataTypes::getCPos(coords[ the_hexa[j] ]);
                center += vertex;
            }

            center = center/8;
            positions.push_back(center);
        }

        vparams->drawTool()->draw3DText_Indices(positions, scale, color4);
    }


    //Draw hexahedra
    if (d_drawHexahedra.getValue())
    {
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, true);

        const sofa::helper::vector<Hexahedron> &hexaArray = this->m_topology->getHexahedra();

        const sofa::defaulttype::Vec3f& color = d_drawColorHexahedra.getValue();
        sofa::defaulttype::Vec4f color4(color[0], color[1], color[2], 1.0f);

        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

        sofa::helper::vector <sofa::defaulttype::Vector3> hexaCoords;

        for (unsigned int i = 0; i<hexaArray.size(); i++)
        {
            const Hexahedron& H = hexaArray[i];

            for (unsigned int j = 0; j<8; j++)
            {
                sofa::defaulttype::Vector3 p; p = DataTypes::getCPos(coords[H[j]]);

                hexaCoords.push_back(p);
            }
        }

        const float& scale = d_drawScaleHexahedra.getValue();

        if(scale >= 1.0 && scale < 0.001)
            vparams->drawTool()->drawHexahedra(hexaCoords, color4);
        else
            vparams->drawTool()->drawScaledHexahedra(hexaCoords, color4, scale);

        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, false);
           
    }
}




} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
