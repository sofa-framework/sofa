/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

template<typename PFP>
bool EarTriangulation<PFP>::inTriangle(const typename PFP::VEC3& P, const typename PFP::VEC3& normal, const typename PFP::VEC3& Ta,  const typename PFP::VEC3& Tb, const typename PFP::VEC3& Tc)
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename VEC3::DATA_TYPE T ;

	if (Geom::tripleProduct(P-Ta, (Tb-Ta), normal) >= T(0))
		return false;

	if (Geom::tripleProduct(P-Tb, (Tc-Tb), normal) >= T(0))
		return false;

	if (Geom::tripleProduct(P-Tc, (Ta-Tc), normal) >= T(0))
		return false;

	return true;
}

template<typename PFP>
void EarTriangulation<PFP>::recompute2Ears( Dart d, const typename PFP::VEC3& normalPoly, bool convex)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart d2 = m_map.phi_1(d);
	Dart d_p = m_map.phi_1(d2);
	Dart d_n = m_map.phi1(d);

	const VEC3& Ta = m_position[d2];
	const VEC3& Tb = m_position[d];
	const VEC3& Tc = m_position[d_p];
	const VEC3& Td = m_position[d_n];

	// compute angle
	VEC3 v1= Tb - Ta;
	VEC3 v2= Tc - Ta;
	VEC3 v3= Td - Tb;

	v1.normalize();
	v2.normalize();
	v3.normalize();

//	float dotpr1 = 1.0f - (v1*v2);
//	float dotpr2 = 1.0f + (v1*v3);
	float dotpr1 = acos(v1*v2) / (M_PI/2.0f);
	float dotpr2 = acos(-(v1*v3)) / (M_PI/2.0f);

	if (!convex)	// if convex no need to test if vertex is an ear (yes)
	{
		VEC3 nv1 = v1^v2;
		VEC3 nv2 = v1^v3;

		if (nv1*normalPoly < 0.0)
			dotpr1 = 10.0f - dotpr1;// not an ears  (concave)
		if (nv2*normalPoly < 0.0)
			dotpr2 = 10.0f - dotpr2;// not an ears  (concave)

		bool finished = (dotpr1>=5.0f) && (dotpr2>=5.0f);
		for (typename VPMS::reverse_iterator it = m_ears.rbegin(); (!finished)&&(it != m_ears.rend())&&(it->angle > 5.0f); ++it)
		{
			Dart dx = it->dart;
			const VEC3& P = m_position[dx];

			if ((dotpr1 < 5.0f) && (d != d_p))
				if (inTriangle(P, normalPoly,Tb,Tc,Ta))
					dotpr1 = 5.0f;// not an ears !

			if ((dotpr2 < 5.0f) && (d != d_n) )
				if (inTriangle(P, normalPoly,Td,Ta,Tb))
					dotpr2 = 5.0f;// not an ears !

			finished = ((dotpr1 >= 5.0f)&&(dotpr2 >= 5.0f));
		}
	}

	float length = (Tb-Tc).norm2();
	m_dartEars[d2] = m_ears.insert(VertexPoly(d2,dotpr1,length));

	length = (Td-Ta).norm2();
	m_dartEars[d] = m_ears.insert(VertexPoly(d,dotpr2,length));
}

template<typename PFP>
float EarTriangulation<PFP>::computeEarInit(Dart d, const typename PFP::VEC3& normalPoly, float& val)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart e =  m_map.phi1(d);
	Dart f =  m_map.phi1(e);

	const VEC3& Ta = m_position[e];
	const VEC3& Tb = m_position[f];
	const VEC3& Tc = m_position[d];

	VEC3 v1 = Tc-Ta;
	VEC3 v2 = Tb-Ta;
	v1.normalize();
	v2.normalize();

//	val = 1.0f - (v1*v2);
	val = acos(v1*v2) / (M_PI/2.0f);

	VEC3 vn = v1^v2;
	if (vn*normalPoly > 0.0f)
		val = 10.0f - val; 		// not an ears  (concave, store at the end for optimized use for intersections)

	if (val>5.0f)
		return 0.0f;

	//INTERSECTION
	f =  m_map.phi1(f);
	while (f != d)
	{
		if (inTriangle(m_position[f], normalPoly,Tb,Tc,Ta))
		{
			val = 5.0f;
			return 0.0f;
		}
		f =  m_map.phi1(f);
	}

	return (Tb-Tc).norm2();
}

template<typename PFP>
//void EarTriangulation<PFP>::trianguleFace(Dart d, DartMarker& mark)
void EarTriangulation<PFP>::trianguleFace(Dart d)
{
	// compute normal to polygon
	typename PFP::VEC3 normalPoly = Algo::Surface::Geometry::newellNormal<PFP>(m_map, d, m_position);

	// first pass create polygon in chained list witht angle computation
	unsigned int nbv = 0;
	unsigned int nbe = 0;
	Dart a = d;

	if (m_map.template phi<111>(d) ==d)
	{
//		mark.markOrbit<FACE>(d);	// mark the face
		return;
	}

	do
	{
		float val;
		float length = computeEarInit(a,normalPoly,val);
		a = m_map.phi1(a);	// phi here because ears is next of a
		m_dartEars[a] = m_ears.insert(VertexPoly(a,val,length));
		if (length!=0)
			nbe++;
		nbv++;
	}while (a!=d);

	// NO WE HAVE THE POLYGON AND EARS
	// LET'S REMOVE THEM

	bool convex = nbe==nbv;

	while (nbv>3)
	{
		// take best (and valid!) ear
		typename VPMS::iterator be_it = m_ears.begin(); // best ear
		Dart d_e = be_it->dart;
		Dart e1 = m_map.phi1(d_e);
		Dart e2 = m_map.phi_1(d_e);

		m_map.splitFace(e1,e2);
//		mark.markOrbit<FACE>(d_e);
		nbv--;

		if (nbv>3)	// do not recompute if only one triangle left
		{
			//remove ears and two sided ears

			m_ears.erase(be_it);					// from map of ears
			m_ears.erase(m_dartEars[e1]);
			m_ears.erase(m_dartEars[e2]);

			recompute2Ears(e1,normalPoly,convex);

			convex = (m_ears.rbegin()->angle) < 5.0f;
		}
//		else
//			mark.markOrbit<FACE>(e1);	// mark last face
	}
	m_ears.clear();
}

template<typename PFP>
void EarTriangulation<PFP>::triangule(unsigned int thread)
{
//	DartMarker m(m_map, thread);
//
//	for(Dart d = m_map.begin(); d != m_map.end(); m_map.next(d))
//	{
//		if(!m.isMarked(d))
//		{
//			Dart e = m_map.template phi<111>(d);
//			if (e!=d)
//				trianguleFace(d, m);
//		}
//	}
//	m.unmarkAll();

	TraversorF<typename PFP::MAP> trav(m_map,thread);

	for(Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		Dart e = m_map.template phi<111>(d);
		if (e!=d)
			trianguleFace(d);
	}
}

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
