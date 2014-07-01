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

#include <limits>
#include "Topology/generic/autoAttributeHandler.h"

#include "Geometry/vector_gen.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL1
{

template <typename PFP>
void renderTopoMD2(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, bool drawPhi1, bool drawPhi2, float ke, float kf)
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;
	
	DartAutoAttribute<VEC3, MAP> fv1(map);
	DartAutoAttribute<VEC3, MAP> fv11(map);
	DartAutoAttribute<VEC3, MAP> fv2(map);
	DartAutoAttribute<VEC3, MAP> vert(map);

	glLineWidth(2.0f);
	glColor3f(0.9f,0.9f,0.9f);
	glBegin(GL_LINES);

	DartMarker<MAP> mf(map);
	for(Dart d = map.begin(); d!= map.end(); map.next(d))
	{
		if (!mf.isMarked(d))
		{
			std::vector<VEC3> vecPos;
			vecPos.reserve(16);

			std::vector<VEC3> vecF1;
			vecF1.reserve(16);

			// store the face & center
			VEC3 center(0.0f,0.0f,0.0f);
			Dart dd = d;
			do
			{
				const VEC3& P = positions[d];
				vecPos.push_back(positions[d]);
				center += P;
				d = map.phi1(d);
			} while (d != dd);
			center /= REAL(vecPos.size());

			//shrink the face
			unsigned int nb = vecPos.size();
			float k = 1.0f - kf;
			for (unsigned int i=0; i<nb; ++i)
			{
				vecPos[i] = center*k + vecPos[i]*kf;
			}
			vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

			k = 1.0f - ke;
			for (unsigned int i=0; i<nb; ++i)
			{
				VEC3 P = vecPos[i]*ke + vecPos[i+1]*k;
				VEC3 Q = vecPos[i+1]*ke + vecPos[i]*k;
				glVertex3fv(P.data());
				glVertex3fv(Q.data());
				vert[d] = P;
				VEC3 f = P*0.5f + Q*0.5f;
				fv2[d] = f;
				f = P*0.1f + Q*0.9f;
				fv1[d] = f;
				f = P*0.9f + Q*0.1f;
				fv11[d] = f;
				d = map.phi1(d);
			}
			mf.markOrbit<FACE>(d);
		}
	}

	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		Dart e = map.phi2(d);
		if ((d<e) && drawPhi2)
		{
			glColor3f(1.0,0.0,0.0);
			glVertex3fv(fv2[d].data());
			glVertex3fv(fv2[e].data());
		}
		if (drawPhi1)
		{
			e = map.phi1(d);
			glColor3f(0.0f,1.0f,1.0f);
			glVertex3fv(fv1[d].data());
			glVertex3fv(fv11[e].data());
		}
	}
	
	glEnd(); // LINES

	glPointSize(5.0f);
	glColor3f(0.0f,0.0f,0.0f);
	glBegin(GL_POINTS);
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		glVertex3fv(vert[d].data());
	}
	glEnd();

}

template <typename PFP>
void renderTopoMD3(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, bool drawPhi1, bool drawPhi2, bool drawPhi3, float ke, float kf, float kv)
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	DartAutoAttribute<VEC3, MAP> fv1(map);
	DartAutoAttribute<VEC3, MAP> fv11(map);
	DartAutoAttribute<VEC3, MAP> fv2(map);
	DartAutoAttribute<VEC3, MAP> fv2x(map);
	DartAutoAttribute<VEC3, MAP> vert(map);

	int m_nbDarts = 0;

	// table of center of volume
	std::vector<VEC3> vecCenters;
	vecCenters.reserve(1000);
	// table of nbfaces per volume
	std::vector<unsigned int> vecNbFaces;
	vecNbFaces.reserve(1000);
	// table of face (one dart of each)
	std::vector<Dart> vecDartFaces;
	vecDartFaces.reserve(map.getNbDarts()/4);

	DartMarker<MAP> mark(map);					// marker for darts
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		CellMarkerStore<MAP, VERTEX> markVert(map);		//marker for vertices
		VEC3 center(0, 0, 0);
		unsigned int nbv = 0;
		unsigned int nbf = 0;
		std::list<Dart> visitedFaces;	// Faces that are traversed
		visitedFaces.push_back(d);		// Start with the face of d

		// For every face added to the list
		for (std::list<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
		{
			if (!mark.isMarked(*face))		// Face has not been visited yet
			{
				// store a dart of face
				vecDartFaces.push_back(*face);
				nbf++;
				Dart dNext = *face ;
				do
				{
					if (!markVert.isMarked(dNext))
					{
						markVert.mark(dNext);
						center += positions[dNext];
						nbv++;
					}
					mark.mark(dNext);					// Mark
					m_nbDarts++;
					Dart adj = map.phi2(dNext);				// Get adjacent face
					if (adj != dNext && !mark.isMarked(adj))
						visitedFaces.push_back(adj);	// Add it
					dNext = map.phi1(dNext);
				} while(dNext != *face);
			}
		}
		center /= typename PFP::REAL(nbv);
		vecCenters.push_back(center);
		vecNbFaces.push_back(nbf);
	}

 	glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f,1.0f,1.0f);

	std::vector<Dart>::iterator face = vecDartFaces.begin();
	for (unsigned int iVol=0; iVol<vecNbFaces.size(); ++iVol)
	{
		for (unsigned int iFace = 0; iFace < vecNbFaces[iVol]; ++iFace)
		{
			Dart d = *face++;

			std::vector<VEC3> vecPos;
			vecPos.reserve(16);

			// store the face & center
			VEC3 center(0, 0, 0);
			Dart dd = d;
			do
			{
				const VEC3& P = positions[d];
				vecPos.push_back(P);
				//m_attIndex[d] = posDBI;
				center += P;
				d = map.phi1(d);
			} while (d != dd);
			center /= REAL(vecPos.size());

			//shrink the face
			unsigned int nb = vecPos.size();
			float okf = 1.0f - kf;
			float okv = 1.0f - kv;
			for (unsigned int i = 0; i < nb; ++i)
			{
				vecPos[i] = vecCenters[iVol]*okv + vecPos[i]*kv;
				vecPos[i] = center*okf + vecPos[i]*kf;
			}
			vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

			// compute position of points to use for drawing topo
			float oke = 1.0f - ke;
			for (unsigned int i = 0; i < nb; ++i)
			{
				VEC3 P = vecPos[i]*ke + vecPos[i+1]*oke;
				VEC3 Q = vecPos[i+1]*ke + vecPos[i]*oke;

				vert[d] = P;

				glVertex3fv(P.data());
				glVertex3fv(Q.data());

				fv1[d] = P*0.1f + Q*0.9f;
				fv11[d] = P*0.9f + Q*0.1f;

				fv2[d] = P*0.52f + Q*0.48f;
				fv2x[d] = P*0.48f + Q*0.52f;

				d = map.phi1(d);
			}

		}
	}

	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		Dart e = map.phi2(d);
		if ((d<e) && drawPhi2)
		{
			glColor3f(1.0,0.0,0.0);
			glVertex3fv(fv2[d].data());
			glVertex3fv(fv2[e].data());
		}

		e = map.phi3(d);
		if(fv2[d] != VEC3(0.0f) && fv2[e] != VEC3(0.0f))
		{
		if ((d<e) && drawPhi3)
		{
			glColor3f(1.0,1.0,0.0);
			glVertex3fv(fv2[d].data());
			glVertex3fv(fv2[e].data());
		}
		}
		if (drawPhi1)
		{
			e = map.phi1(d);
			glColor3f(0.0f,1.0f,1.0f);
			glVertex3fv(fv1[d].data());
			glVertex3fv(fv11[e].data());
		}
	}

	glEnd(); // LINES

	glPointSize(5.0f);
	glColor3f(0.0f,0.0f,0.0f);
	glBegin(GL_POINTS);
	for(Dart d = map.begin(); d!= map.end(); map.next(d))
	{
		glVertex3fv(vert[d].data());
	}
	glEnd();

}

template <typename PFP>
void renderTopoGMD2(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, bool drawBeta0, bool drawBeta1, bool drawBeta2, float kd, float ke, float kf)
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	DartAutoAttribute<VEC3, MAP> posBeta1(map);
	DartAutoAttribute<VEC3, MAP> posBeta2(map);
	DartAutoAttribute<VEC3, MAP> vert(map);

	glLineWidth(2.0f);
	glColor3f(0.9f,0.9f,0.9f);
	glBegin(GL_LINES);

	DartMarker<MAP> mf(map);
	//draw all darts and potentially all beta0
	for(Dart d = map.begin(); d!= map.end(); map.next(d))
	{
		if (!mf.isMarked(d))
		{
			std::vector<VEC3> vecPos;
			vecPos.reserve(32);

			// store the face & center
			VEC3 center(0);
			Dart dd = d;
			do
			{
				const VEC3& P = positions[dd];
				vecPos.push_back(positions[dd]);
				center += P;
				dd = map.phi1(dd);
			} while (dd!=d);
			center /= REAL(vecPos.size());

			//shrink the face
			unsigned int nb = vecPos.size();
			float k = 1.0f - kf;
			for (unsigned int i=0; i<nb; ++i)
				vecPos[i] = center*k + vecPos[i]*kf;

			vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

			k = 1.0f - ke;
			for (unsigned int i=0; i<nb; ++i)
			{
				VEC3 P = vecPos[i]*ke + vecPos[i+1]*k;
				VEC3 Q = vecPos[i+1]*ke + vecPos[i]*k;
				VEC3 P_mid = P+(Q-P)*kd*0.5f;
				VEC3 Q_mid = Q+(P-Q)*kd*0.5f;
				glVertex3fv(P.data());
				glVertex3fv(P_mid.data());

				if(drawBeta0)
				{
					glColor3f(0.0f,0.0f,1.0f);
					glVertex3fv(P_mid.data());
					glVertex3fv(Q_mid.data());
				}

				glColor3f(0.9f,0.9f,0.9f);
				glVertex3fv(Q.data());
				glVertex3fv(Q_mid.data());
				vert[d] = P;

				posBeta2[d] = P*0.5f + P_mid*0.5f;

				posBeta2[map.beta0(d)] = Q*0.5f + Q_mid*0.5f;

				posBeta1[d] = P*0.9f + P_mid*0.1f;
				posBeta1[map.beta0(d)] = Q*0.9f + Q_mid*0.1f;
				d = map.phi1(d);
			}
			mf.template markOrbit<FACE>(d);
		}
	}

	//draw links beta 1 and beta 2
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		Dart e = map.beta2(d);
		if (drawBeta2)
		{
			glColor3f(1.0,0.0,0.0);
			glVertex3fv(posBeta2[d].data());
			glVertex3fv(posBeta2[e].data());
		}

		e = map.beta1(d);
		if ((d<e) && drawBeta1)
		{
			glColor3f(0.0f,1.0f,1.0f);
			glVertex3fv(posBeta1[d].data());
			glVertex3fv(posBeta1[e].data());
		}
	}
	glEnd(); // LINES

	//draw vertices
	glPointSize(5.0f);
	glColor3f(0.0f,0.0f,0.0f);
	glBegin(GL_POINTS);
	for(Dart d = map.begin(); d!= map.end(); map.next(d))
	{
		glVertex3fv(vert[d].data());
	}
	glEnd();
}

template <typename PFP>
void renderTopoGMD3(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, bool drawBeta0, bool drawBeta1, bool drawBeta2, bool drawBeta3, float kd, float ke, float kf, float kv)
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	DartAutoAttribute<VEC3, MAP> posBeta1(map);
	DartAutoAttribute<VEC3, MAP> posBeta2(map); //beta 3 link is represented at the same location as beta2
	DartAutoAttribute<VEC3, MAP> vert(map);

	// table of face (one dart of each)
	std::vector<Dart> vecDartFaces;
	vecDartFaces.reserve(map.getNbDarts()/6); //6 = nb of darts for tri mesh

	// table of degree of faces
	std::vector<unsigned int> vecNbFaces;
	vecNbFaces.reserve(vecDartFaces.size());

	// table of center of volume (to explode volumes)
	std::vector<VEC3> vecVolCenters;
	vecVolCenters.reserve(vecDartFaces.size()/4); // = nb of volumes for a tetra mesh

	DartMarker<MAP> mark(map);					// marker for darts
	CellMarker<MAP, VOLUME> mVol(map);
//	DartMarker mVol(map);

	//compute barycenter and get a dart by face
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if(!mVol.isMarked(d))
		{
			mVol.mark(d);
//			mVol.markOrbit(VOLUME,d);

			CellMarkerStore<MAP, VERTEX> markVert(map);		//marker for vertices
			VEC3 center(0);
			unsigned int nbVertices = 0;
			unsigned int nbf = 0;
			std::list<Dart> visitedFaces;	// Faces that are traversed
			visitedFaces.push_back(d);		// Start with the face of d

			// For every face added to the list
			for (std::list<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
			{
				if (!mark.isMarked(*face))		// Face has not been visited yet
				{
					// store a dart of face
					vecDartFaces.push_back(*face);
					nbf++;
					Dart dNext = *face ;
					do
					{
						mark.mark(dNext); // Mark

						if (!markVert.isMarked(dNext))
						{
							markVert.mark(dNext);
							center += positions[dNext];
							nbVertices++;
						}

						Dart adj = map.phi2(dNext); // add adjacent face if not done already
						if (adj != dNext && !mark.isMarked(adj))
							visitedFaces.push_back(adj);

						dNext = map.phi1(dNext);
					} while(dNext != *face);
				}
			}
			center /= typename PFP::REAL(nbVertices);
			vecVolCenters.push_back(center);
			vecNbFaces.push_back(nbf);
		}
	}

 	glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f,1.0f,1.0f);

	std::vector<Dart>::iterator face = vecDartFaces.begin();
	//for each volume
	for (unsigned int iVol=0; iVol<vecNbFaces.size(); ++iVol)
	{
		//for each face
		for (unsigned int iFace = 0; iFace < vecNbFaces[iVol]; ++iFace)
		{
			Dart d = *face++;

			std::vector<VEC3> vecPos;
			vecPos.reserve(16);

			// store the face & center
			VEC3 center(0);
			Dart dd = d;
			do
			{
				const VEC3& P = positions[dd];
				vecPos.push_back(P);
				center += P;
				dd = map.phi1(dd);
			} while (dd != d);
			center /= REAL(vecPos.size());

			//shrink the face
			unsigned int nb = vecPos.size();
			float okf = 1.0f - kf;
			float okv = 1.0f - kv;
			for (unsigned int i = 0; i < nb; ++i)
			{
				vecPos[i] = vecVolCenters[iVol]*okv + vecPos[i]*kv;
				vecPos[i] = center*okf + vecPos[i]*kf;
			}
			vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

			// compute position of points to use for drawing topo
			float oke = 1.0f - ke;
			for (unsigned int i = 0; i < nb; ++i)
			{
				VEC3 P = vecPos[i]*ke + vecPos[i+1]*oke;
				VEC3 Q = vecPos[i+1]*ke + vecPos[i]*oke;

				VEC3 P_mid = P+(Q-P)*kd*0.5f;
				VEC3 Q_mid = Q+(P-Q)*kd*0.5f;

				vert[d] = P;

				glVertex3fv(P.data());
				glVertex3fv(P_mid.data());

				if(drawBeta0)
				{
					glColor3f(0.0f,0.0f,1.0f);
					glVertex3fv(P_mid.data());
					glVertex3fv(Q_mid.data());
					glColor3f(1.0f,1.0f,1.0f);
				}

				glVertex3fv(Q.data());
				glVertex3fv(Q_mid.data());

				posBeta1[d] = P*0.9f + P_mid*0.1f;
				posBeta1[map.beta0(d)] = Q*0.9f + Q_mid*0.1f;

				posBeta2[d] = P*0.5f + P_mid*0.5f;
				posBeta2[map.beta0(d)] = Q*0.5f + Q_mid*0.5f;

				d = map.phi1(d);
			}

		}
	}

	//draw beta1, beta2, beta3 if required
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		Dart e = map.beta2(d);
		if ((d<e) && drawBeta2)
		{
			glColor3f(1.0,0.0,0.0);
			glVertex3fv(posBeta2[d].data());
			glVertex3fv(posBeta2[e].data());
		}

		e = map.beta3(d);
		if ((d<e) && drawBeta3)
		{
			glColor3f(1.0,1.0,0.0);
			glVertex3fv(posBeta2[d].data());
			glVertex3fv(posBeta2[e].data());
		}

		e = map.beta1(d);
		if ((d<e) && drawBeta1)
		{
			e = map.beta1(d);
			glColor3f(0.0f,1.0f,1.0f);
			glVertex3fv(posBeta1[d].data());
			glVertex3fv(posBeta1[e].data());
		}
	}

	glEnd(); // LINES

	//draw points of the map
	glPointSize(5.0f);
	glColor3f(1.0f,1.0f,1.0f);
	glBegin(GL_POINTS);
	for(Dart d = map.begin(); d!= map.end(); map.next(d))
	{
		glVertex3fv(vert[d].data());
	}
	glEnd();
}

} // namespace GL1

} // namespace Render

} // namespace Algo

} // namespace CGoGN
