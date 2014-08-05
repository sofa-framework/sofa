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

/**
 * create a n-sided pyramid
 */
template <typename PFP>
Dart createPyramid(typename PFP::MAP& map, unsigned int n, bool withBoundary)
{
	Dart dres = Dart::nil();
	std::vector<Dart> m_tableVertDarts;
	m_tableVertDarts.reserve(n);

	// creation of triangles around circunference and storing vertices
	for (unsigned int i = 0; i < n; ++i)
	{
		Dart d = map.newFace(3, false);
		m_tableVertDarts.push_back(d);
	}

	// sewing the triangles
	for (unsigned int i = 0; i < n-1; ++i)
	{
		Dart d = m_tableVertDarts[i];
		d = map.phi_1(d);
		Dart e = m_tableVertDarts[i+1];
		e = map.phi1(e);
		map.sewFaces(d, e, false);
	}
	//sewing the last with the first
	map.sewFaces(map.phi1(m_tableVertDarts[0]), map.phi_1(m_tableVertDarts[n-1]), false);

	//sewing the bottom face
	Dart base = map.newFace(n, false);
	dres = base;
	for(unsigned int i = 0; i < n ; ++i)
	{
		map.sewFaces(m_tableVertDarts[i], base, false);
		base = map.phi1(base);
	}

	if(map.dimension() == 3 && withBoundary)
		map.closeMap();

	//return a dart from the base
	return dres;
}

/**
 * create a n-sided prism
 */
template <typename PFP>
Dart createPrism(typename PFP::MAP& map, unsigned int n, bool withBoundary)
{
	Dart dres = Dart::nil();
	unsigned int nb = n*2;
	std::vector<Dart> m_tableVertDarts;
	m_tableVertDarts.reserve(nb);

	// creation of quads around circunference and storing vertices
	for (unsigned int i = 0; i < n; ++i)
	{
		Dart d = map.newFace(4, false);
		m_tableVertDarts.push_back(d);
	}

	// storing a dart from the vertex pointed by phi1(phi1(d))
	for (unsigned int i = 0; i < n; ++i)
	{
		m_tableVertDarts.push_back(map.phi1(map.phi1(m_tableVertDarts[i])));
	}

	// sewing the quads
	for (unsigned int i = 0; i < n-1; ++i)
	{
		Dart d = m_tableVertDarts[i];
		d = map.phi_1(d);
		Dart e = m_tableVertDarts[i+1];
		e = map.phi1(e);
		map.sewFaces(d, e, false);
	}
	//sewing the last with the first
	map.sewFaces(map.phi1(m_tableVertDarts[0]), map.phi_1(m_tableVertDarts[n-1]), false);

	//sewing the top & bottom faces
	Dart top = map.newFace(n, false);
	Dart bottom = map.newFace(n, false);
	dres = top;
	for(unsigned int i = 0; i < n ; ++i)
	{
		map.sewFaces(m_tableVertDarts[i], top, false);
		map.sewFaces(m_tableVertDarts[n+i], bottom, false);
		top = map.phi1(top);
		bottom = map.phi_1(bottom);
	}

	if(map.dimension() == 3 && withBoundary)
		map.closeMap();

	//return a dart from the base
	return dres;
}

/**
 * create a n-sided diamond
 */
template <typename PFP>
Dart createDiamond(typename PFP::MAP& map, unsigned int nbSides, bool withBoundary)
{
	unsigned int nbt = 2*nbSides -1 ; // -1 for computation optimization
	std::vector<Dart> m_tableVertDarts;
	m_tableVertDarts.reserve(nbSides);
	
	
	// creation of triangles around circunference and storing vertices
	for (unsigned int i = 0; i <= nbt; ++i)
	{
		Dart d = map.newFace(3, false);
		m_tableVertDarts.push_back(d);
	}

	// sewing the triangles
	for (unsigned int i = 0; i < nbSides-1; ++i)
	{
		Dart d = m_tableVertDarts[i];
		d = map.phi_1(d);
		Dart e = m_tableVertDarts[i+1];
		e = map.phi1(e);
		map.sewFaces(d, e, false);
	}
	//sewing the last with the first
	map.sewFaces(map.phi1(m_tableVertDarts[0]), map.phi_1(m_tableVertDarts[nbSides-1]), false);

	for (unsigned int i = nbSides; i < nbt; ++i)
	{
		Dart d = m_tableVertDarts[i];
		d = map.phi_1(d);
		Dart e = m_tableVertDarts[i+1];
		e = map.phi1(e);
		map.sewFaces(d, e, false);
	}
	//sewing the last with the first
	map.sewFaces(map.phi1(m_tableVertDarts[nbSides]), map.phi_1(m_tableVertDarts[nbt]), false);

	//sewing the the two opened pyramids together
	for(unsigned int i = 0; i < nbSides ; ++i)
	{
		map.sewFaces(m_tableVertDarts[i], m_tableVertDarts[nbt-i], false);
	}

	if(map.dimension() == 3 && withBoundary)
		map.closeMap();

	//return a dart from the base
	return m_tableVertDarts[0];
}


/**
 * create a 3-sided prism
 */
template <typename PFP>
Dart createTriangularPrism(typename PFP::MAP& map, bool withBoundary)
{
	return createPrism<PFP>(map, 3, withBoundary);
}

/**
 * create a hexahedron
 */
template <typename PFP>
Dart createHexahedron(typename PFP::MAP& map, bool withBoundary)
{
	return createPrism<PFP>(map, 4, withBoundary);
}

/**
 * create a tetrahedron
 */
template <typename PFP>
Dart createTetrahedron(typename PFP::MAP& map, bool withBoundary)
{
	return createPyramid<PFP>(map, 3, withBoundary);
}

/**
 * create a 4-sided pyramid
 */
template <typename PFP>
Dart createQuadrangularPyramid(typename PFP::MAP& map, bool withBoundary)
{
	return createPyramid<PFP>(map, 4, withBoundary);
}

/**
 * create an octahedron (i.e. 4-sided diamond)
 */
template <typename PFP>
Dart createOctahedron(typename PFP::MAP& map, bool withBoundary)
{
	return createDiamond<PFP>(map,4, withBoundary);
}

template <typename PFP>
Dart embedPrism(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int n, bool withBoundary, float bottom_radius, float top_radius, float height)
{
	typedef typename PFP::VEC3 VEC3 ;

	unsigned int m_nz = 1;

	Dart dres = Dart::nil();
	unsigned int nb = n*2;
	std::vector<Dart> m_tableVertDarts;
	m_tableVertDarts.reserve(nb);

	// creation of quads around circunference and storing vertices
	for (unsigned int i = 0; i < n; ++i)
	{
		Dart d = map.newFace(4, false);
		m_tableVertDarts.push_back(d);
	}

	// storing a dart from the vertex pointed by phi1(phi1(d))
	for (unsigned int i = 0; i < n; ++i)
	{
		//m_tableVertDarts.push_back(map.phi1(map.phi1(m_tableVertDarts[i])));
		m_tableVertDarts.push_back(map.phi_1(m_tableVertDarts[i]));
	}

	// sewing the quads
	for (unsigned int i = 0; i < n-1; ++i)
	{
		Dart d = m_tableVertDarts[i];
		d = map.phi_1(d);
		Dart e = m_tableVertDarts[i+1];
		e = map.phi1(e);
		map.sewFaces(d, e, false);
	}
	//sewing the last with the first
	map.sewFaces(map.phi1(m_tableVertDarts[0]), map.phi_1(m_tableVertDarts[n-1]), false);

	//sewing the top & bottom faces
	Dart top = map.newFace(n, false);
	Dart bottom = map.newFace(n, false);
	dres = top;
	for(unsigned int i = 0; i < n ; ++i)
	{
		map.sewFaces(m_tableVertDarts[i], top, false);
		map.sewFaces(map.phi_1(m_tableVertDarts[n+i]), bottom, false);
		top = map.phi1(top);
		bottom = map.phi_1(bottom);
	}

	if(map.dimension() == 3 && withBoundary)
		map.closeMap();

	float alpha = float(2.0*M_PI/n);
	float dz = height/float(m_nz);

	for(unsigned int i = 0; i <= m_nz; ++i)
	{
		float a = float(i)/float(m_nz);
		float radius = a*top_radius + (1.0f-a)*bottom_radius;
		for(unsigned int j = 0; j < n; ++j)
		{

			float x = radius*cos(alpha*float(j));
			float y = radius*sin(alpha*float(j));
			position[ m_tableVertDarts[i*n+j] ] = VEC3(x, y, -height/2 + dz*float(i));
		}
	}

	return dres;
}

template <typename PFP>
Dart embedPyramid(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, unsigned int n, bool withBoundary, float radius, float height)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart dres = Dart::nil();
	std::vector<Dart> m_tableVertDarts;
	m_tableVertDarts.reserve(n);

	// creation of triangles around circunference and storing vertices
	for (unsigned int i = 0; i < n; ++i)
	{
		Dart d = map.newFace(3, false);
		m_tableVertDarts.push_back(d);
	}

	// sewing the triangles
	for (unsigned int i = 0; i < n-1; ++i)
	{
		Dart d = m_tableVertDarts[i];
		d = map.phi_1(d);
		Dart e = m_tableVertDarts[i+1];
		e = map.phi1(e);
		map.sewFaces(d, e, false);
	}
	//sewing the last with the first
	map.sewFaces(map.phi1(m_tableVertDarts[0]), map.phi_1(m_tableVertDarts[n-1]), false);

	//sewing the bottom face
	Dart base = map.newFace(n, false);
	dres = base;
	for(unsigned int i = 0; i < n ; ++i)
	{
		map.sewFaces(m_tableVertDarts[i], base, false);
		base = map.phi1(base);
	}

	if(map.dimension() == 3 && withBoundary)
		map.closeMap();

	float alpha = float(2.0*M_PI/n);

	for(unsigned int j = 0; j < n; ++j)
	{
		float rad = radius;
		float h = -height/2;
		float x = rad*cos(alpha*float(j));
		float y = rad*sin(alpha*float(j));

		position[ m_tableVertDarts[j] ] = VEC3(x, y, h);
	}

	//  top always closed in cone
	position[ map.phi_1(m_tableVertDarts[0]) ] = VEC3(0.0f, 0.0f, height/2 );

	//return a dart from the base
	return dres;
}

template <typename PFP>
bool isPyra(typename PFP::MAP& map, Dart d, unsigned int thread)
{
	unsigned int nbFacesT = 0;
	unsigned int nbFacesQ = 0;

	//Test the number of faces end its valency
	Traversor3WF<typename PFP::MAP> travWF(map, d, false, thread);
	for(Dart dit = travWF.begin() ; dit != travWF.end(); dit = travWF.next())
	{
		//increase the number of faces
		if(map.faceDegree(dit) == 3)
			nbFacesT++;
		else if(map.faceDegree(dit) == 4)
			nbFacesQ++;
		else
			return false;
	}

	if((nbFacesT != 4) || (nbFacesQ != 1))	//too much faces
		return false;

	return true;
}

template <typename PFP>
bool isPrism(typename PFP::MAP& map, Dart d, unsigned int thread)
{
	unsigned int nbFacesT = 0;
	unsigned int nbFacesQ = 0;

	//Test the number of faces end its valency
	Traversor3WF<typename PFP::MAP> travWF(map, d, false, thread);
	for(Dart dit = travWF.begin() ; dit != travWF.end(); dit = travWF.next())
	{
		//increase the number of faces
		if(map.faceDegree(dit) == 3)
			nbFacesT++;
		else if(map.faceDegree(dit) == 4)
			nbFacesQ++;
		else
			return false;
	}

	if((nbFacesT != 2) || (nbFacesQ != 3))	//too much faces
		return false;

	return true;
}


template <typename PFP>
void explodPolyhedron(typename PFP::MAP& map, Dart d,  VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	map.unsewVertexUmbrella(d);
	unsigned int newFaceDegree = map.closeHole(map.phi1(d));

	if(newFaceDegree != 3)
	{
		// chercher le brin ou demarrer

		std::multimap<Dart, int> edges ;
		typename std::multimap<Dart, int>::iterator it ;

		Dart d12 = map.phi2(map.phi1(d));
		Dart fit = d12;
		unsigned int i;

		do
		{
			i = map.faceDegree(map.phi2(fit));
			std::cout << "edge(" << fit << "," << i << ")" << std::endl;
			edges.insert(std::make_pair(fit, i));
			fit = map.phi1(fit);
		}
		while(fit != d12);

		do
		{
			//44 44
			if(edges.find(fit)->second == 4 && edges.find(map.phi1(fit))->second == 4
				&& !map.sameFace(map.phi2(fit), map.phi2(map.phi1(fit))))
			{
				map.splitFace(fit, map.phi1(map.phi1(fit)));
				fit = map.phi2(map.phi_1(fit));
				int i = map.faceDegree(fit);
				edges.insert(std::make_pair(fit, i));

//				Dart fit2 = map.phi2(fit) ;
//				typename PFP::VEC3 p1 = position[fit] ;
//				typename PFP::VEC3 p2 = position[fit2] ;
//
//				map.cutEdge(fit) ;
//				position[map.phi1(fit)] = typename PFP::REAL(0.5) * (p1 + p2);


				std::cout << "flip cas quad quad " << std::endl;
			}

			//3 3
			if(edges.find(fit)->second == 3 && edges.find(map.phi1(fit))->second == 3
				&& !map.sameFace(map.phi2(fit), map.phi2(map.phi1(fit))))
			{
				map.splitFace(fit, map.phi1(fit));
				fit = map.phi2(map.phi_1(fit));
				int i = map.faceDegree(fit);
				edges.insert(std::make_pair(fit, i));

				std::cout << "flip cas tri tri" << std::endl;
			}

			//3 44 ou 44 3
			if( ((edges.find(fit)->second == 4 && edges.find(map.phi1(fit))->second == 3)
				|| (edges.find(fit)->second == 3 && edges.find(map.phi1(fit))->second == 4))
					&& !map.sameFace(map.phi2(fit), map.phi2(map.phi1(fit))))
			{
				map.splitFace(fit, map.phi1(map.phi1(fit)));
				fit = map.phi2(map.phi_1(fit));
				int i = map.faceDegree(fit);
				edges.insert(std::make_pair(fit, i));

				std::cout << "flip cas quad tri" << std::endl;
			}

			fit = map.phi1(fit);
		}
		while(map.faceDegree(fit) > 4 && fit != d12);
	}
}

template <typename PFP>
void quads2TrianglesCC(typename PFP::MAP& the_map, Dart primd)
{
	DartMarker<typename PFP::MAP> m(the_map);

	// list of faces to process and processed(before pos iterator)
	std::list<Dart> ld;
	ld.push_back(primd);
	// current position in list
	typename std::list<Dart>::iterator pos = ld.begin();
	do
	{
	   Dart d = *pos;

	   // cut the face of first dart of list
	   Dart d1 = the_map.phi1(d);
	   Dart e = the_map.phi1(d1);
	   Dart e1 = the_map.phi1(e);
	   Dart f = the_map.phi1(e1);
	   if (f == d) // quad
	   {
		   the_map.splitFace(d,e);
		   // mark the face
		   m.template markOrbit<FACE>(d);
		   m.template markOrbit<FACE>(e);
	   }
	   else m.template markOrbit<FACE>(d);

	   // and store neighbours faces in the list
	   d = the_map.phi2(d);
	   e = the_map.phi2(e);
	   d1 = the_map.phi1(the_map.phi2(d1));
	   e1 = the_map.phi1(the_map.phi2(e1));

	   if (!m.isMarked(d))
		   ld.push_back(d);
	   if (!m.isMarked(e))
		   ld.push_back(e);
	   if (!m.isMarked(d1))
		   ld.push_back(d1);
	   if ((f == d) && (!m.isMarked(e1)))
		   ld.push_back(e1);
	   pos++;
	} while (pos!=ld.end()); // stop when no more face to process
}

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
