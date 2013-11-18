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

#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/autoAttributeHandler.h"
#include "Container/fakeAttribute.h"
#include "Algo/Modelisation/polyhedron.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{

template <typename PFP>
bool importMesh(typename PFP::MAP& map, MeshTablesSurface<PFP>& mts)
{
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	unsigned nbf = mts.getNbFaces();
	int index = 0;
	// buffer for tempo faces (used to remove degenerated edges)
	std::vector<unsigned int> edgesBuffer;
	edgesBuffer.reserve(16);

	DartMarkerNoUnmark m(map) ;

	FunctorInitEmb<typename PFP::MAP, VERTEX> fsetemb(map);

	// for each face of table
	for(unsigned int i = 0; i < nbf; ++i)
	{
		// store face in buffer, removing degenerated edges
		unsigned int nbe = mts.getNbEdgesFace(i);
		edgesBuffer.clear();
		unsigned int prec = EMBNULL;
		for (unsigned int j = 0; j < nbe; ++j)
		{
			unsigned int em = mts.getEmbIdx(index++);
			if (em != prec)
			{
				prec = em;
				edgesBuffer.push_back(em);
			}
		}
		// check first/last vertices
		if (edgesBuffer.front() == edgesBuffer.back())
			edgesBuffer.pop_back();

		// create only non degenerated faces
		nbe = edgesBuffer.size();
		if (nbe > 2)
		{
			Dart d = map.newFace(nbe, false);
			for (unsigned int j = 0; j < nbe; ++j)
			{
				unsigned int em = edgesBuffer[j];		// get embedding
				fsetemb.changeEmb(em) ;
				map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

				m.mark(d) ;								// mark on the fly to unmark on second loop
				vecDartsPerVertex[em].push_back(d);		// store incident darts for fast adjacency reconstruction
				d = map.phi1(d);
			}
		}
	}

	// reconstruct neighbourhood
	unsigned int nbBoundaryEdges = 0;
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (m.isMarked(d))
		{
			// darts incident to end vertex of edge
			std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];

			unsigned int embd = map.template getEmbedding<VERTEX>(d);
			Dart good_dart = NIL;
			for (typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
			{
				if (map.template getEmbedding<VERTEX>(map.phi1(*it)) == embd)
					good_dart = *it;
			}

			if (good_dart != NIL)
			{
				if (good_dart == map.phi2(good_dart))
				{
					map.sewFaces(d, good_dart, false);
					m.unmarkOrbit<EDGE>(d);
				}
				else
				{
					++nbBoundaryEdges;
				}
			}
			else
			{
				m.unmark(d);
				++nbBoundaryEdges;
			}
		}
	}

	if (nbBoundaryEdges > 0)
	{
		unsigned int nbH = map.closeMap();
		CGoGNout << "Map closed (" << nbBoundaryEdges << " boundary edges / " << nbH << " holes)" << CGoGNendl;
		// ensure bijection between topo and embedding
		map.template bijectiveOrbitEmbedding<VERTEX>();
	}

	return true ;
}


template <typename PFP>
bool importMesh(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, bool mergeCloseVertices)
{
	MeshTablesSurface<PFP> mts(map);

	if(!mts.importMesh(filename, attrNames))
		return false;

	if (mergeCloseVertices)
		mts.mergeCloseVertices();

	return importMesh<PFP>(map, mts);
}

template <typename PFP>
bool importVoxellisation(typename PFP::MAP& map, Algo::Surface::Modelisation::Voxellisation& voxellisation, std::vector<std::string>& attrNames, bool mergeCloseVertices)
{
    MeshTablesSurface<PFP> mts(map);

    if(!mts.importVoxellisation(voxellisation, attrNames))
        return false;

    if (mergeCloseVertices)
        mts.mergeCloseVertices();

    return importMesh<PFP>(map, mts);
}


template <typename PFP>
bool importMeshSAsV(typename PFP::MAP& map, MeshTablesSurface<PFP>& mts)
{
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");

	unsigned nbf = mts.getNbFaces();
	int index = 0;
	// buffer for tempo faces (used to remove degenerated edges)
	std::vector<unsigned int> edgesBuffer;
	edgesBuffer.reserve(16);

	DartMarkerNoUnmark m(map) ;

	// for each face of table
	for(unsigned int i = 0; i < nbf; ++i)
	{
		// store face in buffer, removing degenerated edges
		unsigned int nbe = mts.getNbEdgesFace(i);
		edgesBuffer.clear();
		unsigned int prec = EMBNULL;
		for (unsigned int j = 0; j < nbe; ++j)
		{
			unsigned int em = mts.getEmbIdx(index++);
			if (em != prec)
			{
				prec = em;
				edgesBuffer.push_back(em);
			}
		}
		// check first/last vertices
		if (edgesBuffer.front() == edgesBuffer.back())
			edgesBuffer.pop_back();

		// create only non degenerated faces
		nbe = edgesBuffer.size();
		if (nbe > 2)
		{
			Dart d = map.newFace(nbe, false);
			for (unsigned int j = 0; j < nbe; ++j)
			{
				unsigned int em = edgesBuffer[j];		// get embedding

				FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, em);
//				foreach_dart_of_orbit_in_parent<typename PFP::MAP>(&map, VERTEX, d, fsetemb) ;
				map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT2>(d, fsetemb);

				m.mark(d) ;								// mark on the fly to unmark on second loop
				vecDartsPerVertex[em].push_back(d);		// store incident darts for fast adjacency reconstruction
				d = map.phi1(d);
			}
		}
	}

	// reconstruct neighbourhood
	unsigned int nbBoundaryEdges = 0;
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (m.isMarked(d))
		{
			// darts incident to end vertex of edge
			std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];

			unsigned int embd = map.template getEmbedding<VERTEX>(d);
			Dart good_dart = NIL;
			for (typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
			{
				if (map.template getEmbedding<VERTEX>(map.phi1(*it)) == embd)
					good_dart = *it;
			}

			if (good_dart != NIL)
			{
				map.sewFaces(d, good_dart, false);
				m.unmarkOrbit<EDGE>(d);
			}
			else
			{
				m.unmark(d);
				++nbBoundaryEdges;
			}
		}
	}

	unsigned int nbH = map.closeMap();
	CGoGNout << "Map closed (" << map.template getNbOrbits<FACE>() << " boundary faces / " << nbH << " holes)" << CGoGNendl;
	std::cout << "nb darts : " << map.getNbDarts() << std::endl ;
	// ensure bijection between topo and embedding
	//map.template bijectiveOrbitEmbedding<VERTEX>();

	return true ;
}

template <typename PFP>
bool importMeshSAsV(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames)
{
	MeshTablesSurface<PFP> mts(map);

	if(!mts.importMesh(filename, attrNames))
		return false;

	return importMeshSAsV<PFP>(map, mts);
}


}
}


namespace Volume
{

namespace Import
{

template <typename PFP>
bool importMeshSToV(typename PFP::MAP& map, Surface::Import::MeshTablesSurface<PFP>& mts, float dist)
{
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");
	unsigned nbf = mts.getNbFaces();
	int index = 0;
	// buffer for tempo faces (used to remove degenerated edges)
	std::vector<unsigned int> edgesBuffer;
	edgesBuffer.reserve(16);

	DartMarkerNoUnmark m(map) ;

	VertexAttribute<typename PFP::VEC3> position = map.template getAttribute<typename PFP::VEC3, VERTEX>("position");
	std::vector<unsigned int > backEdgesBuffer(mts.getNbVertices(), EMBNULL);

	// for each face of table -> create a prism
	for(unsigned int i = 0; i < nbf; ++i)
	{
		// store face in buffer, removing degenerated edges
		unsigned int nbe = mts.getNbEdgesFace(i);
		edgesBuffer.clear();
		unsigned int prec = EMBNULL;
		for (unsigned int j = 0; j < nbe; ++j)
		{
			unsigned int em = mts.getEmbIdx(index++);
			if (em != prec)
			{
				prec = em;
				edgesBuffer.push_back(em);
			}
		}
		// check first/last vertices
		if (edgesBuffer.front() == edgesBuffer.back())
			edgesBuffer.pop_back();

		// create only non degenerated faces
		nbe = edgesBuffer.size();
		if (nbe > 2)
		{
			Dart d = Surface::Modelisation::createPrism<PFP>(map, nbe,false);

			//Embed the base faces
			for (unsigned int j = 0; j < nbe; ++j)
			{
				unsigned int em = edgesBuffer[j];		// get embedding

				if(backEdgesBuffer[em] == EMBNULL)
				{
					unsigned int emn = map.template newCell<VERTEX>();
					map.template copyCell<VERTEX>(emn, em);
					backEdgesBuffer[em] = emn;
					position[emn] += typename PFP::VEC3(0,0,dist);
				}

				FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, em);
				//foreach_dart_of_orbit_in_parent<typename PFP::MAP>(&map, VERTEX, d, fsetemb) ;
				map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);

				//Embed the other base face
				Dart d2 = map.phi1(map.phi1(map.phi2(d)));
				unsigned int em2 = backEdgesBuffer[em];
				FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb2(map, em2);
				//foreach_dart_of_orbit_in_parent<typename PFP::MAP>(&map, VERTEX, d2, fsetemb2) ;
				map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d2, fsetemb2);

				m.mark(d) ;								// mark on the fly to unmark on second loop
				vecDartsPerVertex[em].push_back(d);		// store incident darts for fast adjacency reconstruction
				d = map.phi_1(d);
			}

		}
	}

	// reconstruct neighbourhood
	unsigned int nbBoundaryEdges = 0;
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (m.isMarked(d))
		{
			// darts incident to end vertex of edge
			std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];

			unsigned int embd = map.template getEmbedding<VERTEX>(d);
			Dart good_dart = NIL;
			for (typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
			{
				if (map.template getEmbedding<VERTEX>(map.phi1(*it)) == embd)
					good_dart = *it;
			}

			if (good_dart != NIL)
			{
				map.sewVolumes(map.phi2(d), map.phi2(good_dart), false);
				m.unmarkOrbit<EDGE>(d);
			}
			else
			{
				m.unmark(d);
				++nbBoundaryEdges;
			}
		}
	}

	return true ;
}

template <typename PFP>
bool importMeshSurfToVol(typename PFP::MAP& map, Surface::Import::MeshTablesSurface<PFP>& mts, float scale, unsigned int nbStage)
{
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map);
	unsigned nbf = mts.getNbFaces();
	int index = 0;
	// buffer for tempo faces (used to remove degenerated edges)
	std::vector<unsigned int> edgesBuffer;
	edgesBuffer.reserve(16);

	DartMarkerNoUnmark m(map) ;

	unsigned int nbVertices = mts.getNbVertices();

	VertexAttribute<typename PFP::VEC3> position = map.template getAttribute<typename PFP::VEC3, VERTEX>("position");
	std::vector<unsigned int > backEdgesBuffer(nbVertices*nbStage, EMBNULL);

	// for each face of table -> create a prism
	for(unsigned int i = 0; i < nbf; ++i)
	{
		// store face in buffer, removing degenerated edges
		unsigned int nbe = mts.getNbEdgesFace(i);
		edgesBuffer.clear();
		unsigned int prec = EMBNULL;
		for (unsigned int j = 0; j < nbe; ++j)
		{
			unsigned int em = mts.getEmbIdx(index++);
			if (em != prec)
			{
				prec = em;
				edgesBuffer.push_back(em);
			}
		}
		// check first/last vertices
		if (edgesBuffer.front() == edgesBuffer.back())
			edgesBuffer.pop_back();

		// create only non degenerated faces
		nbe = edgesBuffer.size();
		if (nbe > 2)
		{
			Dart dprev = NIL;

			for(unsigned int k = 0 ; k < nbStage ; ++k)
			{
				Dart d = Surface::Modelisation::createPrism<PFP>(map, nbe,false);

				//Embed the base faces
				for (unsigned int j = 0; j < nbe; ++j)
				{
					unsigned int em = edgesBuffer[j];		// get embedding
					Dart d2 = map.phi1(map.phi1(map.phi2(d)));

					if(k==0)
					{
						FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, em);
						map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
						vecDartsPerVertex[em].push_back(d);		// store incident darts for fast adjacency reconstruction
						m.mark(d) ;								// mark on the fly to unmark on second loop
					}
					else
					{
						unsigned int emn = backEdgesBuffer[((k-1)*nbVertices) + em];
						FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, emn);
						map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
						vecDartsPerVertex[emn].push_back(d);		// store incident darts for fast adjacency reconstruction
						m.mark(d) ;								// mark on the fly to unmark on second loop
					}

					if(backEdgesBuffer[(k*nbVertices) + em] == EMBNULL)
					{
						unsigned int emn = map.template newCell<VERTEX>();
						map.template copyCell<VERTEX>(emn, em);
						backEdgesBuffer[(k*nbVertices) + em] = emn;
						position[emn] += typename PFP::VEC3(0,0, (k+1) * scale);
					}

					unsigned int em2 = backEdgesBuffer[(k*nbVertices) + em];
					FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(map, em2);
					map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d2, fsetemb);

					d = map.phi_1(d);
				}


				if(dprev != NIL)
					map.sewVolumes(d, map.phi2(map.phi1(map.phi1(map.phi2(dprev)))), false);

				dprev = d;
			}
		}
	}

	// reconstruct neighbourhood
	unsigned int nbBoundaryEdges = 0;
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (m.isMarked(d))
		{
			// darts incident to end vertex of edge
			std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];

			unsigned int embd = map.template getEmbedding<VERTEX>(d);
			Dart good_dart = NIL;
			for (typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
			{
				if (map.template getEmbedding<VERTEX>(map.phi1(*it)) == embd)
					good_dart = *it;
			}

			if (good_dart != NIL)
			{
				map.sewVolumes(map.phi2(d), map.phi2(good_dart), false);
				m.unmarkOrbit<EDGE>(d);
			}
			else
			{
				m.unmark(d);
				++nbBoundaryEdges;
			}
		}
	}

	map.closeMap();

	return true ;
}


template <typename PFP>
bool importMesh(typename PFP::MAP& map, MeshTablesVolume<PFP>& mtv)
{
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> > > vecDartsPerVertex(map, "incidents");
	return false;
}


template <typename PFP>
bool importMesh(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, bool /*mergeCloseVertices*/)
{

	ImportType kind = getFileType(filename);

	switch (kind)
	{
		case TET:
			return importTet<PFP>(map, filename, attrNames, 1.0f);
			break;
		case MSH:
			return importMSH<PFP>(map, filename, attrNames, 1.0f);
			break;
		case VTU:
		return importVTU<PFP>(map, filename, attrNames, 1.0f);
		break;

	case NAS:
		return importNAS<PFP>(map, filename, attrNames, 1.0f);
		break;

	case VBGZ:
		return importVBGZ<PFP>(map, filename, attrNames, 1.0f);
		break;

	case TETMESH:
		return importTetmesh<PFP>(map, filename, attrNames, 1.0f);
		break;

//	case OVM:
//		return importOVM<PFP>(map, filename, attrNames, 1.0f);
//		break;


		case OFF:
		{
			size_t pos = filename.rfind(".");
			std::string fileEle = filename;
			fileEle.erase(pos);
			fileEle.append(".ele");
			return importOFFWithELERegions<PFP>(map, filename, fileEle, attrNames);
			break;
		}
		case NODE:
		{
			size_t pos = filename.rfind(".");
			std::string fileEle = filename;
			fileEle.erase(pos);
			fileEle.append(".ele");
			return importNodeWithELERegions<PFP>(map, filename, fileEle, attrNames);
			break;
		}
		case Volume::Import::TS:
			return importTs<PFP>(map, filename, attrNames, 1.0f);
			break;
		default:
			CGoGNerr << "Not yet supported" << CGoGNendl;
			break;
	}
	return false;
}

template <typename PFP>
bool importMeshToExtrude(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, float scale, unsigned int nbStage)
{
	Surface::Import::MeshTablesSurface<PFP> mts(map);

	if(!mts.importMesh(filename, attrNames))
		return false;

	return importMeshSurfToVol<PFP>(map, mts, scale, nbStage);
}



} // namespace Import
}


} // namespace Algo

} // namespace CGoGN
