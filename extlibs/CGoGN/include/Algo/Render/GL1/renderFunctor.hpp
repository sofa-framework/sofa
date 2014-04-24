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

#include <GL/glew.h>

#include "Algo/Geometry/normal.h"
#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL1
{

/**
* @param lighted use normal (face or vertex)
* @param smooth use per vertex normal
* @param nbe number of vertex per primitive (3 for triangles, 4 for quads, -1 for polygons)
* @param expl exploding coefficient
* @param stor shall we store faces that are not of the good primitive type
* @param vn the vertex normal vector (indiced by dart label)
*/
template <typename PFP>
FunctorGLFace<PFP>::FunctorGLFace(MAP& map, bool lighted, bool smooth, int nbe, float expl, bool stor,
		const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3>& normals):
	FunctorMap<MAP>(map),
	m_smooth(smooth),
	m_lighted(lighted),
	m_nbEdges(nbe),
	m_explode(expl),
	m_storing(stor),
	m_positions(posi),
	m_normals(normals)
{
}

/**
* get back the vector of darts of faces that have not been treated
*/
template<typename PFP>
std::vector<Dart>& FunctorGLFace<PFP>::getPolyDarts()
{
	return m_poly;
}

/**
* operator applied on each face:
* if the face has the right number of edges
* it is rendered (glVertex). Other are stored
* if needed.
*/
template<typename PFP>
bool FunctorGLFace<PFP>::operator() (Dart d)
{
		if (m_explode == 1.0f)
			renderFace(d);
		else
			renderFaceExplode(d);
	return false;
}

/**
* Render a face without exploding
*/
template<typename PFP>
void FunctorGLFace<PFP>::renderFace(Dart d)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart dd = d;

	// count the number of edges of the face
	unsigned nbEdges = this->m_map.faceDegree(d);

	// First case, it is a polygon to be render with a triangle Fan
	if (m_nbEdges == TRIFAN)
	{
		VEC3 norm;
		if (m_lighted && !m_smooth)	// use face normal
		{
			norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}

		// compute center
		VEC3 center = Algo::Surface::Geometry::faceCentroid<PFP>(this->m_map, d, m_positions);
		VEC3 centerNormal = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, dd, m_positions);

		if (m_smooth) 				// use vertex normal
		{
			glNormal3fv(centerNormal.data());
			glVertex3fv(center.data());
		}
		else
		{
			if (m_lighted)	// use face normal
				glNormal3fv(norm.data());
			glVertex3fv(center.data());
		}

		//traverse vertices of polygon
		dd = d;
		if (m_smooth) {					// use vertex normal
			for(unsigned i=0; i<=nbEdges; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else {
			for(unsigned i=0; i<=nbEdges; ++i)
			{
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	// case of polygons to be rendered with a polygon
	else if (m_nbEdges==POLYGONS)
	{
		if (m_lighted && !m_smooth)	// use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}

		//traverse vertices of polygons
		dd = d;
		if (m_smooth)			// use vertex normal
		{
			for(unsigned i=0; i<nbEdges; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else					// do not use normal
		{
			for(unsigned i=0; i<nbEdges; ++i)
			{
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	else 
	if ( (m_nbEdges == QUADS) && (nbEdges == QUADS))
	{
		if (m_lighted && !m_smooth)	// use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}
		// vertices
		dd = d;
		if (m_smooth)				// use vertex normal
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
			dd = this->m_map.phi_1(dd);
			for(unsigned i = 0; i < 3; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
			
		}
		else
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
			dd = this->m_map.phi_1(dd);
			for(unsigned i = 0; i < 3; ++i)
			{

				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}

	}
	// now the case of triangles (first pass)
	else if (nbEdges == TRIANGLES)
	{
		if (m_lighted && !m_smooth)	// use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}
		// vertices
		dd = d;
		if (m_smooth)				// use vertex normal
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	else if (m_storing)				// face is not rendered, if needed store it
	{
		m_poly.push_back(d);
	}
}

/**
* Render a face with exploding
*/
template<typename PFP>
void FunctorGLFace<PFP>::renderFaceExplode(Dart d)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart dd = d;
	
	// TODO deplacer les 2 vecteurs (et reserve) dans l'objet (et le constructeur)
	//std::vector<typename PFP::VEC3> vecEmb;
	//vecEmb.reserve(64);				// reserve for 64 edges
	std::vector<VEC3> vecPos;

	if (m_nbEdges==POLYGONS)
	{
		m_nbEdges=0;
		do
		{
			//vecEmb.push_back(m_positions[dd]);
			dd = this->m_map.phi1(dd);
			m_nbEdges++;
		} while (dd != d);
	}
	else
	{
		for (unsigned i = 0; i < m_nbEdges; ++i)
		{
			//vecEmb.push_back(m_positions[dd]);
			dd = this->m_map.phi1(dd);
		}
	}
//
	// test if face has to be rendered
	if (dd == d)
	{
		//CGoGNout << "POLY: "<<m_nbEdges<<CGoGNendl;
		VEC3 center = Algo::Surface::Geometry::faceCentroid<PFP>(this->m_map, d, m_positions);

		// modify vertex position (explode faces)
		vecPos.resize(m_nbEdges);
		float opp_expl = 1.0f - m_explode;
		
		for(unsigned i=0; i< m_nbEdges; ++i)
		{
			vecPos[i] = m_explode * m_positions[dd] + opp_expl * center;
			//CGoGNout<< "POS: "<<vecPos[i] << CGoGNendl;
			dd= this->m_map.phi1(dd);
		}
		if ( m_lighted && !m_smooth ) // use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			//if (m_inverted) norm *= -1.0f ;
			glNormal3fv(norm.data());
		}
		// vertices
		dd = d;
		if (m_smooth) 				// use vertex normal
		{
			for(unsigned i=0; i<m_nbEdges; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glVertex3fv(vecPos[i].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else
		{
			for(unsigned i=0; i<m_nbEdges; ++i)
			{
				glVertex3fv(vecPos[i].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	else if (m_storing) // face is not render, if needed store it
	{
		m_poly.push_back(d);
		//CGoGNout << "store dart"<<CGoGNendl;

	}
}

template<typename PFP>
FunctorGLNormal<PFP>::FunctorGLNormal(MAP& map, const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3>& normals, float scale):
	FunctorMap<MAP>(map),
	m_positions(posi),
	m_normals(normals),
	m_scale(scale)
{
}

template<typename PFP>
bool FunctorGLNormal<PFP>::operator() (Dart d)
{

		typename PFP::VEC3 p = m_positions[d];
		glVertex3fv(p.data());
		p += m_scale * m_normals[d];
		glVertex3fv(p.data());

	return false;
}

template<typename PFP>
FunctorGLFrame<PFP>::FunctorGLFrame(MAP& map, const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3> frames[3], float scale):
	FunctorMap<MAP>(map),
	m_positions(posi),
	m_frames(frames),
	m_scale(scale)
{
}

template<typename PFP>
bool FunctorGLFrame<PFP>::operator() (Dart d)
{

		typename PFP::VEC3 p = m_positions[d] ;
		for (unsigned int i = 0 ; i < 3 ; ++i) {
			glVertex3fv(p.data());
			typename PFP::VEC3 q ;
			q = p ;
			q += m_scale * m_frames[i][d] ;
			glVertex3fv(q.data());
		}
	return false;
}

template <typename PFP>
FunctorGLFaceColor<PFP>::FunctorGLFaceColor(MAP& map, bool lighted, bool smooth, int nbe, float expl, bool stor,
		const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3>& normals, const VertexAttribute<typename PFP::VEC3>& colors):
	FunctorMap<MAP>(map),
	m_smooth(smooth),
	m_lighted(lighted),
	m_nbEdges(nbe),
	m_explode(expl),
	m_storing(stor),
	m_positions(posi),
	m_normals(normals),
	m_colors(colors)
{
}
template<typename PFP>
std::vector<Dart>& FunctorGLFaceColor<PFP>::getPolyDarts()
{
	return m_poly;
}

template<typename PFP>
bool FunctorGLFaceColor<PFP>::operator() (Dart d)
{

		if (m_explode == 1.0f)
			renderFace(d);
		else
			renderFaceExplode(d);

	return false;
}

template<typename PFP>
void FunctorGLFaceColor<PFP>::renderFace(Dart d)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart dd = d;

	// count the number of edges of the face
	unsigned nbEdges = this->m_map.faceDegree(d);

	// First case, it is a polygon to be render with a triangle Fan
	if (m_nbEdges == TRIFAN)
	{
		VEC3 norm;
		if (m_lighted && !m_smooth)	// use face normal
		{
			norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}

		// compute center
		VEC3 center = Algo::Surface::Geometry::faceCentroid<PFP>(this->m_map, d, m_positions);
		VEC3 centerNormal = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, dd, m_positions);
		VEC3 centerColor = Algo::Surface::Geometry::faceCentroid<PFP>(this->m_map, d, m_colors);

		if (m_smooth) 				// use vertex normal
		{
			glNormal3fv(centerNormal.data());
			glColor3fv(centerColor.data());
			glVertex3fv(center.data());
		}
		else
		{
			if (m_lighted)	// use face normal
				glNormal3fv(norm.data());
			glColor3fv(centerColor.data());
			glVertex3fv(center.data());
		}

		//traverse vertices of polygon
		dd = d;
		if (m_smooth) {					// use vertex normal
			for(unsigned i=0; i<=nbEdges; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else {
			for(unsigned i=0; i<=nbEdges; ++i)
			{
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	// case of polygons to be rendered with a polygon
	else if (m_nbEdges==POLYGONS)
	{
		if (m_lighted && !m_smooth)	// use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}

		//traverse vertices of polygons
		dd = d;
		if (m_smooth)			// use vertex normal
		{
			for(unsigned i=0; i<nbEdges; ++i)
			{
				glColor3fv(m_colors[dd].data());
				glNormal3fv(m_normals[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else					// do not use normal
		{
			for(unsigned i=0; i<nbEdges; ++i)
			{
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	else
	if ( (m_nbEdges == QUADS) && (nbEdges == QUADS))
	{
		if (m_lighted && !m_smooth)	// use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}
		// vertices
		dd = d;
		if (m_smooth)				// use vertex normal
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
			dd = this->m_map.phi_1(dd);
			for(unsigned i = 0; i < 3; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}

		}
		else
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
			dd = this->m_map.phi_1(dd);
			for(unsigned i = 0; i < 3; ++i)
			{
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}

	}
	// now the case of triangles (first pass)
	else if (nbEdges == TRIANGLES)
	{
		if (m_lighted && !m_smooth)	// use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			glNormal3fv(norm.data());
		}
		// vertices
		dd = d;
		if (m_smooth)				// use vertex normal
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glColor3fv(m_colors[dd].data());
				glVertex3fv(m_positions[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else
		{
			for(unsigned i = 0; i < 3; ++i)
			{
				glVertex3fv(m_positions[dd].data());
				glColor3fv(m_colors[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	else if (m_storing)				// face is not rendered, if needed store it
	{
		m_poly.push_back(d);
	}
}

template<typename PFP>
void FunctorGLFaceColor<PFP>::renderFaceExplode(Dart d)
{
	typedef typename PFP::VEC3 VEC3 ;

	Dart dd = d;

	// TODO deplacer les 2 vecteurs (et reserve) dans l'objet (et le constructeur)
	//std::vector<typename PFP::VEC3> vecEmb;
	//vecEmb.reserve(64);				// reserve for 64 edges
	std::vector<VEC3> vecPos;

	if (m_nbEdges==POLYGONS)
	{
		m_nbEdges=0;
		do
		{
			//vecEmb.push_back(m_positions[dd]);
			dd = this->m_map.phi1(dd);
			m_nbEdges++;
		} while (dd != d);
	}
	else
	{
		for (unsigned i = 0; i < m_nbEdges; ++i)
		{
			//vecEmb.push_back(m_positions[dd]);
			dd = this->m_map.phi1(dd);
		}
	}
//
	// test if face has to be rendered
	if (dd == d)
	{
		//CGoGNout << "POLY: "<<m_nbEdges<<CGoGNendl;
		VEC3 center = Algo::Surface::Geometry::faceCentroid<PFP>(this->m_map, d, m_positions);

		// modify vertex position (explode faces)
		vecPos.resize(m_nbEdges);
		float opp_expl = 1.0f - m_explode;

		for(unsigned i=0; i< m_nbEdges; ++i)
		{
			vecPos[i] = m_explode * m_positions[dd] + opp_expl * center;
			//CGoGNout<< "POS: "<<vecPos[i] << CGoGNendl;
			dd= this->m_map.phi1(dd);
		}
		if ( m_lighted && !m_smooth ) // use face normal
		{
			VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(this->m_map, d, m_positions);
			//if (m_inverted) norm *= -1.0f ;
			glNormal3fv(norm.data());
		}
		// vertices
		dd = d;
		if (m_smooth) 				// use vertex normal
		{
			for(unsigned i=0; i<m_nbEdges; ++i)
			{
				glNormal3fv(m_normals[dd].data());
				glColor3fv(m_colors[dd].data());
				glVertex3fv(vecPos[i].data());
				dd = this->m_map.phi1(dd);
			}
		}
		else
		{
			for(unsigned i=0; i<m_nbEdges; ++i)
			{
				glVertex3fv(vecPos[i].data());
				glColor3fv(m_colors[dd].data());
				dd = this->m_map.phi1(dd);
			}
		}
	}
	else if (m_storing) // face is not render, if needed store it
	{
		m_poly.push_back(d);
		//CGoGNout << "store dart"<<CGoGNendl;
	}
}

} // namespace GL1

} // namespace Render

} // namespace Algo

} // namespace CGoGN
