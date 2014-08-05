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

#include "Utils/gl_def.h"
#include "Geometry/transfo.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

template<typename PFP>
Algo::Surface::Tilings::Tiling<PFP>* revolution_prim(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& center,
	const typename PFP::VEC3& axis,
	bool profile_closed,
	int nbSides)
{
	typedef typename PFP::VEC3 VEC3 ;

	// find circle center
	float k = (axis * (center-profile[0])) / (axis*axis);
	VEC3 circCenter = center + k*axis;

	// compute vector base plane for the circle
	VEC3 U = profile[0] - circCenter;
	VEC3 V = axis^U;
	V.normalize();
	V *=  U.norm();

	// create the path:
	std::vector<typename PFP::VEC3> path;
	path.reserve(nbSides);
	for(int i=0; i< nbSides; ++i)
	{
		float alpha = float(2.0*M_PI/nbSides*i);
		VEC3 P = circCenter + cos(alpha)*V + sin(alpha)*U;
		path.push_back(P);
	}
	// do the extrusion with good parameters
	return extrusion_prim<PFP>(the_map, position, profile, path[0], U, profile_closed, path, true);
}

template<typename PFP>
Dart revolution(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& center,
	const typename PFP::VEC3& axis,
	bool profile_closed,
	int nbSides)
{
    Algo::Surface::Tilings::Tiling<PFP> *prim = revolution_prim<PFP>(the_map, position, profile, center, axis, profile_closed, nbSides);
	Dart d = prim->getDart();
	delete prim;
	return d;
}


template<typename PFP>
Dart extrusion_scale(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normalProfile,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed,
	const std::vector<float>& scalePath)
{
    Algo::Surface::Tilings::Tiling<PFP> *prim = extrusion_scale_prim<PFP>(the_map, position, profile, centerProfile, normalProfile, profile_closed, path, path_closed,scalePath);
	Dart d = prim->getDart();
	delete prim;
	return d;
}

template<typename PFP>
Dart extrusion(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normalProfile,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed)
{
	std::vector<float> scalePath;
    Algo::Surface::Tilings::Tiling<PFP> *prim = extrusion_scale_prim<PFP>(the_map, position, profile, centerProfile, normalProfile, profile_closed, path, path_closed,scalePath);
	Dart d = prim->getDart();
	delete prim;
	return d;
}

template<typename PFP>
Algo::Surface::Tilings::Tiling<PFP>* extrusion_prim(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normalProfile,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed)
{
	std::vector<float> scalePath;
	return extrusion_scale_prim<PFP>(the_map, position, profile, centerProfile, normalProfile, profile_closed, path, path_closed,scalePath);
}

template<typename PFP>
Algo::Surface::Tilings::Tiling<PFP>* extrusion_scale_prim(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const std::vector<typename PFP::VEC3>& profile,
	const typename PFP::VEC3& centerProfile,
	const typename PFP::VEC3& normal,
	bool profile_closed,
	const std::vector<typename PFP::VEC3>& path,
	bool path_closed,
	const std::vector<float>& scalePath)
{
	// topological creation
    Algo::Surface::Tilings::Tiling<PFP>* prim;

    //Dart grid;

	if (profile_closed)
	{
		if (path_closed)
            prim = new Algo::Surface::Tilings::Square::Tore<PFP>(the_map, profile.size(), path.size());
		else
			prim = new Algo::Surface::Tilings::Square::Cylinder<PFP>(the_map, profile.size(), path.size()-1, false, false);
	}
	else
	{
		if (path_closed)
		{
			prim = new Algo::Surface::Tilings::Square::Grid<PFP>(the_map, profile.size()-1, path.size(), true);
			// sewing boundaries correponding to path boundaries
			std::vector<Dart>& darts = prim->getVertexDarts();
			int index = profile.size()*path.size();
			for (unsigned int i=0;i<profile.size()-1;++i)
			{
				Dart d = the_map.phi_1(darts[index++]);
				the_map.sewFaces(d,darts[i]);
			}
			// remove (virtually) the last row of darts that is no more need due to sewing
			darts.resize(darts.size() - profile.size());

		}
		else
			prim = new Algo::Surface::Tilings::Square::Grid<PFP>(the_map, profile.size()-1, path.size()-1, true);
	}

	glPushMatrix();
	// embedding
	std::vector<Dart>& vertD = prim->getVertexDarts();
	typename PFP::VEC3 normalObj(normal);
	normalObj.normalize();

	// put profile at the beginning of path
	std::vector<typename PFP::VEC3> localObj;
	localObj.reserve(profile.size());
	for(typename std::vector<typename PFP::VEC3>::const_iterator ip=profile.begin(); ip!=profile.end(); ++ip)
	{
		typename PFP::VEC3 P = *ip + path[0] - centerProfile;
		localObj.push_back(P);
	}

	int index=0;
	for(unsigned int i=0; i<path.size(); ++i)
	{
		typename PFP::VEC3 rot(0.0f,0.0f,0.0f);
		typename PFP::VEC3 VP;
		if (i==0) //begin
		{
			// vector on path
			VP = path[i+1] - path[i];
			// computing axis of rotation
			VP.normalize();
			rot = normalObj^VP;
			normalObj= VP;
		}
		else if (i==(path.size()-1)) //end
		{
			if (!path_closed)
			{
				// vector on path
				VP = path[i] - path[i-1];
				// computing axis of rotation
				VP.normalize();
				rot=normalObj^VP;
			}
			else
			{
				typename PFP::VEC3 V1 =  path[0] - path[i];
				typename PFP::VEC3 V2 =  path[i] - path[i-1];
				V1.normalize();
				V2.normalize();
				// vector on path
				VP = V1+V2;
				// computing axis of rotation
				VP.normalize();
				rot=normalObj^VP;
			}
		}
		else // middle nodes
		{
			typename PFP::VEC3 V1 =  path[i+1] - path[i];
			typename PFP::VEC3 V2 =  path[i] - path[i-1];

			V1.normalize();
			V2.normalize();
			// vector on path
			VP = V1+V2;
			// computing axis of rotation
			VP.normalize();
			rot = normalObj^VP;

			normalObj= V1;
		}

		// computing angle of rotation
		float pscal = normalObj*VP;
		float asinAlpha = rot.normalize();
		float alpha;
		if (pscal>=0)
			alpha = asin(asinAlpha);
		else
			alpha = float(M_PI) - asin(asinAlpha);
		// creation of transformation matrix
		Geom::Matrix44f transf;
		transf.identity();
		if (alpha>0.00001f)
		{
			Geom::translate(-path[i][0],-path[i][1],-path[i][2],transf);
			Geom::rotate(rot[0],rot[1],rot[2],alpha,transf);
			Geom::translate(path[i][0],path[i][1],path[i][2],transf);
		}

		CGoGNout << "PATH: "<< i<< CGoGNendl;
		// apply transfo on object to embed Polyhedron.
		for(typename std::vector<typename PFP::VEC3>::iterator ip = localObj.begin(); ip != localObj.end(); ++ip)
		{
			if (i!=0) //exept for first point of path
			{
				(*ip) += (path[i]-path[i-1]);
			}

			(*ip)= Geom::transform((*ip), transf);

			unsigned int em = the_map.template newCell<VERTEX>();
//			positions[em] = (*ip);
			typename PFP::VEC3 P = (*ip); //positions.at(em);

			if (!scalePath.empty())
				P = path[i] + (scalePath[i]*(P-path[i]));

			CGoGNout << "P: "<< P<< CGoGNendl;

			// compute the scale factor for angle deformation
			float coef = 1.0f/(float(sin(M_PI/2.0f - alpha))); // warning here is angle/2 but alpha is half of angle we want to use
			if (fabs(coef-1.0f)>0.00001f)
			{
				// projection of path point on plane define par P and the rot vector
				float k = (rot*(P-path[i])) / (rot*rot);
				typename PFP::VEC3 X = path[i] + k*rot;
				//and scale in the plane
				position[em] = X + coef*(P-X);
			}
			else position[em] = P;

			Dart d = vertD[index++];
			Algo::Topo::setOrbitEmbedding<VERTEX>(the_map, d, em);

			// rotate again to put profile in the good position along the path
//			pos4=Geom::Vec4f ((*ip)[0],(*ip)[1],(*ip)[2], 1.0f);
//			np4=Geom::Vec4f( (pos4 * tv1), (pos4  * tv2), (pos4 * tv3), (pos4 * tv4));
//			(*ip)[0] = np4[0]/np4[3];
//			(*ip)[1] = np4[1]/np4[3];
//			(*ip)[2] = np4[2]/np4[3];
			(*ip)= Geom::transform((*ip), transf);
		}
	}
	glPopMatrix();
	return prim;
}

template <typename PFP>
Dart extrudeFace(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,
	Dart d,
	const typename PFP::VEC3& N)
{
	// triangule
	Dart c = Surface::Modelisation::trianguleFace<PFP>(the_map, d);

	Dart cc = c;
	// cut edges
	do
	{
		the_map.cutEdge(cc);
		cc = the_map.phi2_1(cc);
	}while (cc != c);

	// cut faces
	do
	{
		Dart d1 = the_map.phi1(cc);
		Dart d2 = the_map.phi_1(cc);
		the_map.splitFace(d1,d2);
		cc = the_map.phi2_1(cc);
	}while (cc != c);

	// delete the central vertex
	the_map.deleteVertex(c) ;

	// embedding of new vertices
	Dart dd = d;
	do
	{
		const typename PFP::VEC3& P = positions[dd];
		positions[the_map.phi_1(dd)] = P+N;
		dd = the_map.phi1(the_map.phi2(the_map.phi1(dd)));
	} while (dd != d);

	// return a dart of the extruded face 
	return the_map.phi2(the_map.phi1(the_map.phi1(d)));
}

template <typename PFP>
Dart extrudeFace(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	Dart d,
	float dist)
{
	//compute normal
	typename PFP::VEC3 normal = Surface::Geometry::faceNormal<PFP>(the_map, d, position);
	normal *= dist;
	return extrudeFace<PFP>(the_map, position, d, normal);
}

template<typename PFP>
Dart extrudeRegion(
	typename PFP::MAP& the_map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	Dart d,
	const CellMarker<typename PFP::MAP, FACE>& cm)
{
   typedef typename PFP::MAP MAP;

   /*----------- find a dart of the border of the region -----------*/
   std::vector<Dart> borderDarts;
   std::vector<Dart> pile;
   pile.push_back(d);
   CellMarker<MAP, EDGE> em(the_map);

   while(!pile.empty())
   {
	   Traversor2FE<MAP> trav2FE(the_map , pile[pile.size()-1]) ;
	   pile.pop_back();
	   for (Dart it = trav2FE.begin(); it != trav2FE.end (); it = trav2FE.next())
	   {
		   if (!cm.isMarked(it))
		   {
			   borderDarts.push_back(the_map.phi2(it));
			   break;
		   }
		   else if (!em.isMarked(the_map.phi2(it)))
					   pile.push_back(the_map.phi2(it));
		   em.mark(it);
	   }
   }

   /*----------- find borders of the region -----------*/
   std::vector<std::vector<Dart> > contours; // border
   for (unsigned int i = 0; i < borderDarts.size(); i++)
   {
	   bool used = false;
	   for (unsigned int j = 0; j < contours.size(); j++)
	   {
		   for (unsigned int k = 0; k < contours[j].size(); k++)
		   {
			   if (borderDarts[i] == contours[j][k])
			   {
				   used = true;
				   break;
			   }
		   }
		   if(used) break;
	   }

	   if (!used)
	   {
		   std::vector<Dart> contour;
		   Dart contourDart = borderDarts[i];
		   Dart it = contourDart;
		   do
		   {
			   contour.push_back(it);
			   it = the_map.phi2(it);
			   while(!cm.isMarked(it))
				   it = the_map.phi2(the_map.phi_1(it));

		   } while(it != contourDart);
		   contours.push_back(contour);
	   }
   }

   /*----------- calculate the height of the extrusion -----------*/
   typename PFP::REAL dist = 0;
   for(unsigned int i = 0; i < (contours.size()); i++)
   {
	   for(unsigned int j = 0; j < contours[i].size()-1; j++)
	   {
		   dist = dist + (position[contours[i][j]] - position[contours[i][j+1]]).norm();
	   }
	   dist = dist + (position[contours[i][contours[i].size()-1]] - position[contours[i][0]]).norm();
   }
   int size = 0;
   for (unsigned int i = 0; i < contours.size(); i++)
   {
	   size = size + contours[i].size();
   }
   dist = dist / size; // height

   /*----------- calculate the normal -----------*/
   pile.clear();
   pile.push_back(contours[0][0]);
   typename PFP::VEC3 normal;
   CellMarker<MAP, FACE> fm(the_map);
   int faceNb = 0;
   while(!pile.empty())
   {
	   Traversor2FE<MAP> trav2FE2(the_map , pile[pile.size()-1]) ;
	   pile.pop_back();
	   for (Dart it = trav2FE2.begin(); it != trav2FE2.end (); it = trav2FE2.next())
	   {
		   if((cm.isMarked(the_map.phi2(it))) && (!fm.isMarked(the_map.phi2(it))))
			   pile.push_back(the_map.phi2(it));
		   if(!fm.isMarked(it))
		   {
			   normal = normal + Surface::Geometry::faceNormal<PFP>(the_map, it, position);
			   faceNb++;
			   fm.mark(it);
		   }
	   }
   }
   normal = normal / faceNb * dist;

   /*----------- add new faces -----------*/
   for (unsigned int i = 0; i < contours.size(); i++)
   {
	   Dart prev;
	   for(unsigned int j = 0; j < contours[i].size(); j++)
	   {
		   d = the_map.newFace(4);
		   Dart e = the_map.phi2(contours[i][j]);
		   the_map.unsewFaces(contours[i][j]);
		   the_map.sewFaces(contours[i][j], d);
		   the_map.sewFaces(e, the_map.phi1(the_map.phi1(d)));
		   if (j > 0)
			   the_map.sewFaces(prev, the_map.phi1(d));
		   prev = the_map.phi_1(d);
		   position[the_map.phi1(the_map.phi1(d))] = position[contours[i][j]];
	   }
	   the_map.sewFaces(prev, the_map.phi1(the_map.phi2(contours[i][0])));
   }

   /*----------- fix positions of vertices of the region -----------*/
   pile.clear();
   pile.push_back(contours[0][0]);
   CellMarker<MAP, VERTEX> vm(the_map);

   while(!pile.empty())
   {
	   Traversor2VF<MAP> trav2VF(the_map , pile[pile.size()-1]);
	   pile.pop_back();
	   for (Dart it = trav2VF.begin(); it != trav2VF.end (); it = trav2VF.next())
	   {
		   if(!vm.isMarked(it))
		   {
			   position[it] = position[it]+normal;
			   vm.mark(it);
		   }
		   if (cm.isMarked(it) && (!vm.isMarked(the_map.phi2(it))))
			   pile.push_back(the_map.phi2(it));
	   }
   }

   return contours[0][0];
}

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
