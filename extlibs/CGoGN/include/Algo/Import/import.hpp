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
#include "Algo/Topo/basic.h"
#include <utils.h>
namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{



template <typename PFP>
bool importMesh(typename PFP::MAP& map, CGoGN::Algo::Volume::Import::MeshTablesVolume<PFP>& mtv ) {
    return false;
}

template <typename PFP>
bool importMesh(typename PFP::MAP& map, MeshTablesSurface<PFP>& mts)
{
    typedef typename PFP::MAP MAP;

        VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> >, MAP> vecDartsPerVertex(map, "incidents");

        unsigned nbf = mts.getNbFaces();
        int index = 0;
        // buffer for tempo faces (used to remove degenerated edges)
        std::vector<unsigned int> edgesBuffer;
        edgesBuffer.reserve(16);

        DartMarkerNoUnmark<MAP> m(map) ;

    //	unsigned int vemb = EMBNULL;
    //	auto fsetemb = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb); };

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
                    unsigned int vemb = edgesBuffer[j];	// get embedding
//                    map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { map.template initDartEmbedding<VERTEX>(dd, vemb); });
                    map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb))) );

                    m.mark(d) ;								// mark on the fly to unmark on second loop
                    vecDartsPerVertex[vemb].push_back(d);	// store incident darts for fast adjacency reconstruction
                    d = map.phi1(d);
                }
            }
        }

        bool needBijectiveCheck = false;

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
                bool firstOK = true;
                for (typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
                {
                    if (map.template getEmbedding<VERTEX>(map.phi1(*it)) == embd)
                    {
                        good_dart = *it;
                        if (good_dart == map.phi2(good_dart))
                        {
                            map.sewFaces(d, good_dart, false);
                            m.template unmarkOrbit<EDGE>(d);
                        }
                        else
                        {
                            good_dart = NIL;
                            firstOK = false;
                        }
                    }
                }

                if (!firstOK)
                    needBijectiveCheck = true;

                if (good_dart == NIL)
                {
                    m.template unmarkOrbit<EDGE>(d);
                    ++nbBoundaryEdges;
                }
            }
        }

        if (nbBoundaryEdges > 0)
        {
            unsigned int nbH = map.closeMap();
            CGoGNout << "Map closed (" << nbBoundaryEdges << " boundary edges / " << nbH << " holes)" << CGoGNendl;
        }

        if (needBijectiveCheck)
        {
            // ensure bijection between topo and embedding
            Algo::Topo::bijectiveOrbitEmbedding<VERTEX>(map);
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
    typedef typename PFP::MAP MAP;

    VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> >, MAP> vecDartsPerVertex(map, "incidents");

    unsigned nbf = mts.getNbFaces();
    int index = 0;
    // buffer for tempo faces (used to remove degenerated edges)
    std::vector<unsigned int> edgesBuffer;
    edgesBuffer.reserve(16);

    DartMarkerNoUnmark<MAP> m(map) ;

    unsigned int vemb = EMBNULL;
//    auto fsetemb = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb); };
    boost::function<unsigned (Dart)> fsetemb= bl::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb));

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
                vemb = edgesBuffer[j];		// get embedding
                map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT2>(d, fsetemb);

                m.mark(d) ;								// mark on the fly to unmark on second loop
                vecDartsPerVertex[vemb].push_back(d);	// store incident darts for fast adjacency reconstruction
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


} // namespace Import

} // namespace Surface



namespace Volume
{

namespace Import
{

template <typename PFP>
bool importMeshSToV(typename PFP::MAP& map, Surface::Import::MeshTablesSurface<PFP>& mts, float dist)
{
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;

    VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> >, MAP> vecDartsPerVertex(map, "incidents");
    unsigned nbf = mts.getNbFaces();
    int index = 0;
    // buffer for tempo faces (used to remove degenerated edges)
    std::vector<unsigned int> edgesBuffer;
    edgesBuffer.reserve(16);

    DartMarkerNoUnmark<MAP> m(map) ;

    unsigned int vemb1 = EMBNULL;
//    auto fsetemb1 = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb1); };
    boost::function<unsigned (Dart)> fsetemb1= boost::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb1));
    unsigned int vemb2 = EMBNULL;
//    auto fsetemb2 = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb2); };
    boost::function<unsigned (Dart)> fsetemb2 = boost::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb2));

    VertexAttribute<VEC3, MAP> position = map.template getAttribute<VEC3, VERTEX>("position");
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
            Dart d = Surface::Modelisation::createPrism<PFP>(map, nbe, false);

            //Embed the base faces
            for (unsigned int j = 0; j < nbe; ++j)
            {
                vemb1 = edgesBuffer[j];		// get embedding

                if(backEdgesBuffer[vemb1] == EMBNULL)
                {
                    unsigned int emn = map.template newCell<VERTEX>();
                    map.template copyCell<VERTEX>(emn, vemb1);
                    backEdgesBuffer[vemb1] = emn;
                    position[emn] += typename PFP::VEC3(0,0,dist);
                }

                map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb1);

                //Embed the other base face
                Dart d2 = map.phi1(map.phi1(map.phi2(d)));
                vemb2 = backEdgesBuffer[vemb1];

                map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d2, fsetemb2);

                m.mark(d) ;								// mark on the fly to unmark on second loop
                vecDartsPerVertex[vemb1].push_back(d);	// store incident darts for fast adjacency reconstruction
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
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;

    VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> >, MAP> vecDartsPerVertex(map);
    unsigned nbf = mts.getNbFaces();
    int index = 0;
    // buffer for tempo faces (used to remove degenerated edges)
    std::vector<unsigned int> edgesBuffer;
    edgesBuffer.reserve(16);

    DartMarkerNoUnmark<MAP> m(map) ;

    unsigned int vemb1 = EMBNULL;
//	auto fsetemb1 = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb1); };
    boost::function<unsigned (Dart)> fsetemb1 = boost::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb1));
    unsigned int vemb2 = EMBNULL;
//	auto fsetemb2 = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb2); };
    boost::function<unsigned (Dart)> fsetemb2 = boost::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb2));

    unsigned int nbVertices = mts.getNbVertices();

    VertexAttribute<VEC3, MAP> position = map.template getAttribute<VEC3, VERTEX, MAP>("position");
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
                    vemb1 = edgesBuffer[j];		// get embedding
                    Dart d2 = map.phi1(map.phi1(map.phi2(d)));

                    if(k==0)
                    {
//                        map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { map.template initDartEmbedding<VERTEX>(dd, vemb1); });
                        map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb1))) );
                        vecDartsPerVertex[vemb1].push_back(d);	// store incident darts for fast adjacency reconstruction
                        m.mark(d) ;								// mark on the fly to unmark on second loop
                    }
                    else
                    {
                        //						unsigned int emn = backEdgesBuffer[((k-1)*nbVertices) + em];
                        vemb2 = backEdgesBuffer[((k-1)*nbVertices) + vemb1];
//                        map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { map.template initDartEmbedding<VERTEX>(dd, vemb2); });
                        map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb2))) );
                        vecDartsPerVertex[vemb2].push_back(d);	// store incident darts for fast adjacency reconstruction
                        m.mark(d) ;								// mark on the fly to unmark on second loop
                    }

                    if(backEdgesBuffer[(k*nbVertices) + vemb1] == EMBNULL)
                    {
                        vemb2 = map.template newCell<VERTEX>();
                        map.template copyCell<VERTEX>(vemb2, vemb1);
                        backEdgesBuffer[(k*nbVertices) + vemb1] = vemb2;
                        position[vemb2] += typename PFP::VEC3(0,0, (k+1) * scale);
                    }

                    vemb2 = backEdgesBuffer[(k*nbVertices) + vemb1];
//                    map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d2, [&] (Dart dd) { map.template initDartEmbedding<VERTEX>(dd, vemb2); });
                    map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d2,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb2))) );

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
                m.template unmarkOrbit<EDGE>(d);
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
bool importMesh(typename PFP::MAP& map, MeshTablesVolume<PFP>& mtv) {
    typedef typename MeshTablesVolume<PFP>::VOLUME_TYPE VOLUME_TYPE;
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;
    // store incident darts to a Vertex for every incident volume to this vertex
    VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> >, typename PFP::MAP > vecDartsPerVertex(map, "incidents");

    const unsigned int nbv = mtv.getNbVolumes();
    unsigned int index = 0u;
    // buffer for tempo faces (used to remove degenerated edges)
    std::vector<unsigned int> vertexEmbeddingsBuffer;
    vertexEmbeddingsBuffer.reserve(16);

    DartMarkerNoUnmark<typename PFP::MAP> m(map) ;
//    FunctorInitEmb<typename PFP::MAP, VERTEX> fsetemb(map);

    //for each volume of table
    for(unsigned int i = 0 ; i < nbv ; ++i) {
        VOLUME_TYPE VT = mtv.getVolumeType(i);
        // store volume in buffer, removing degenated faces
        const unsigned int nbVerticesVolume  = mtv.getNbVerticesOfVolume(VT);


        vertexEmbeddingsBuffer.clear();
        unsigned int prec = EMBNULL;
        for (unsigned int j = 0; j < nbVerticesVolume; ++j)
        {
            unsigned int em = mtv.getEmbIdx(index++);
            if (em != prec) {
                prec = em;
                vertexEmbeddingsBuffer.push_back(em);
            }
        }

        if(VT == MeshTablesVolume<PFP>::TETRAHEDRON) //tetrahedral case
        {
            Dart d = Surface::Modelisation::createTetrahedron<PFP>(map,false);

            // Embed three "base" vertices
            for(unsigned int j = 0 ; j < 3 ; ++j)
            {
                unsigned int em = vertexEmbeddingsBuffer[j];		// get embedding
//                map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { map.template initDartEmbedding<VERTEX>(dd, em); });
                map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
                //store darts per vertices to optimize reconstruction
                Dart dd = d;
                do
                {
                    m.mark(dd) ;
                    vecDartsPerVertex[em].push_back(dd);
                    dd = map.phi1(map.phi2(dd));
                } while(dd != d);

                d = map.phi1(d);
            }

            //Embed the last "top" vertex
            d = map.phi_1(map.phi2(d));

            unsigned int em = vertexEmbeddingsBuffer[3];		// get embedding
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );

            //store darts per vertices to optimize reconstruction
            Dart dd = d;
            do
            {
                m.mark(dd) ;
                vecDartsPerVertex[em].push_back(dd);
                dd = map.phi1(map.phi2(dd));
            } while(dd != d);

        }
        else if(VT == MeshTablesVolume<PFP>::HEXAHEDRON) //hexahedral case
        {
            Dart d = Surface::Modelisation::createHexahedron<PFP>(map,false);

            // 1.
            unsigned int em = vertexEmbeddingsBuffer[0];		// get embedding
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            Dart dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 2.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[1];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 3.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[2];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 4.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[3];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 5.
            d = map.template phi<2112>(d);
            em = vertexEmbeddingsBuffer[4];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 6.
            d = map.phi_1(d);
            em = vertexEmbeddingsBuffer[5];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 7.
            d = map.phi_1(d);
            em = vertexEmbeddingsBuffer[6];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 8.
            d = map.phi_1(d);
            em = vertexEmbeddingsBuffer[7];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

        } //end of hexa
        else  if (VT == MeshTablesVolume<PFP>::SQUARE_PYRAMID) {
            Dart d = Surface::Modelisation::createQuadrangularPyramid<PFP>(map,false);

            // 1.
            unsigned int em = vertexEmbeddingsBuffer[0];		// get embedding
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            Dart dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 2.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[1];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 3.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[2];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 4.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[3];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 5.
            d = map.phi_1(map.phi2((d)));
            em = vertexEmbeddingsBuffer[4];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

        } else if (VT == MeshTablesVolume<PFP>::TRIANGULAR_PRISM) {
            Dart d = Surface::Modelisation::createTriangularPrism<PFP>(map,false);
            const Dart e = map.phi2(d);
            const Dart f = map.phi1(map.phi2(map.phi1(e)));
            const Dart g = map.phi1(map.phi2(map.phi1(f)));

            VertexAttribute<typename PFP::VEC3, MAP> position =  map.template getAttribute<typename PFP::VEC3, VERTEX, MAP>("position") ;

            // 1.
            unsigned int em = vertexEmbeddingsBuffer[0];		// get embedding
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            Dart dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 2.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[1];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 3.
            d = map.phi1(d);
            em = vertexEmbeddingsBuffer[2];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 4.
            d = map.phi2(map.phi1(map.phi1(map.phi2(d))));//template phi<2112>(d);
            em = vertexEmbeddingsBuffer[3];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 5.
            d = map.phi_1(d);
            em = vertexEmbeddingsBuffer[4];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            // 6.
            d = map.phi_1(d);
            em = vertexEmbeddingsBuffer[5];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(em))) );
            dd = d;
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[em].push_back(dd); m.mark(dd);

            //debugging
            const VEC3 & P =position[e];
            const VEC3 & A = position[map.phi1(e)];
            const VEC3 & B = position[map.phi1(map.phi1(e))];
            const VEC3 & C = position[map.phi1(map.phi1(map.phi1(e)))];
            if (Geom::testOrientation3D<VEC3>(P, A, B, C) == Geom::ON) {
                d = map.phi2(e);
                SHOW(position[d]);
                d = map.phi1(d);
                SHOW(position[d]);
                d = map.phi1(d);
                SHOW(position[d]);
                d = map.phi2(map.phi1(map.phi1(map.phi2(d))));
                SHOW(position[d]);
                d = map.phi_1(d);
                SHOW(position[d]);
                d = map.phi_1(d);
                SHOW(position[d]);
                DEBUG;
            }

        }
    }


//    //reconstruct neighbourhood
//    unsigned int nbBoundaryFaces = 0 ;
//    //    Dart d1 = Dart(24);
//    //    Dart d2 = Dart(141558);
//    //    SHOW(map.template getEmbedding<VERTEX>(map.phi1(d1).index));
//    //    SHOW(map.template getEmbedding<VERTEX>(map.phi1(d2).index));
//    //    SHOW(map.template getEmbedding<VERTEX>(map.phi_1(d1).index));
//    //    SHOW(map.template getEmbedding<VERTEX>(map.phi_1(d2).index));
//    //    SHOW(map.faceDegree(d1));
//    //    SHOW(map.faceDegree(d2));
//    //    SHOW(map.template getEmbedding<VERTEX>(d1));
//    //    SHOW(map.template getEmbedding<VERTEX>(d2));
//    for (Dart d = map.begin(); d != map.end(); map.next(d))
//    {

//        if (m.isMarked(d))
//        {
//            std::vector<Dart>& vec = vecIncidentFacesToVertex[map.phi1(d)];

//            Dart good_dart = NIL;
//            for(typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
//            {
//                if (map.template getEmbedding<VERTEX>(map.phi1(*it)) == map.template getEmbedding<VERTEX>(d) &&
//                        map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.phi1(map.phi1(d))) &&
//                        map.template getEmbedding<VERTEX>(*it) == map.template getEmbedding<VERTEX>(map.phi1(d))  /*Always true by construction */) {
//                    good_dart = *it ;
//                }
//            }

//            if (good_dart != NIL) { // not at the boundary
//                unsigned int F1_degree = map.faceDegree(d);
//                unsigned int F2_degree = map.faceDegree(good_dart);
//                if ( F1_degree  < 3 || F2_degree < 3 || F1_degree > 4  || F2_degree > 4 || d.index == 334425 || good_dart.index == 334425) {
//                    SHOW(F1_degree);
//                    SHOW(F2_degree);
//                }
//                //                SHOW(d);
//                //                SHOW(map.phi3(d));
//                //                SHOW(good_dart);
//                //                SHOW(map.phi3(good_dart));
//                if (F1_degree != F2_degree) {
//                    if (F1_degree > F2_degree) { // F1D = 4 ; F2D = 3
//                        const Dart dt = map.phi1(map.phi1(d));
//                        map.CGoGN::Map2::splitFace(d, dt);
//                        map.template setDartEmbedding<VERTEX>(map.phi_1(d), map.template getEmbedding<VERTEX>(dt));
//                        map.template setDartEmbedding<VERTEX>(map.phi_1(dt), map.template getEmbedding<VERTEX>(d));
//                    } else { // F1D = 3 ; F2D = 4
//                        const Dart gdt = map.phi1(good_dart);
//                        map.CGoGN::Map2::splitFace(map.phi_1(good_dart), gdt);
//                        map.template setDartEmbedding<VERTEX>(map.phi1(good_dart), map.template getEmbedding<VERTEX>(gdt));
//                        map.template setDartEmbedding<VERTEX>(map.phi_1(gdt), map.template getEmbedding<VERTEX>(map.phi_1(good_dart)));
//                    }
//                } else {
//                    if (F1_degree == 4u) {
//                        VertexAttribute<typename PFP::VEC3> position =  map.template getAttribute<typename PFP::VEC3, VERTEX>("position") ;
//                        assert(position.isValid());
//                        if (map.template getEmbedding<VERTEX>(map.phi_1(d)) != map.template getEmbedding<VERTEX>(map.phi1(map.phi1(good_dart)))) {
//                            //                            SHOW(position[d]);
//                            //                            SHOW(position[map.phi1(d)]);
//                            //                            SHOW(position[map.phi1(map.phi1(d))]);
//                            //                            SHOW(position[map.phi_1(d)]);
//                            //                            DEBUG;
//                            //                            SHOW(position[good_dart]);
//                            //                            SHOW(position[map.phi1(good_dart)]);
//                            //                            SHOW(position[map.phi1(map.phi1(good_dart))]);
//                            //                            SHOW(position[map.phi_1(good_dart)]);
//                            //                            std::cerr << std::endl;
//                        }
//                    }
//                }
//                if ( map.phi3(d) != d || map.phi3(good_dart) != good_dart ) {
//                    //debugging
//                    //                    SHOW(d);
//                    //                    SHOW(map.phi3(d));
//                    //                    SHOW(good_dart);
//                    //                    SHOW(map.phi3(good_dart));


//                }
//                map.sewVolumes(d, good_dart, false);
//                m.unmarkOrbit<FACE>(d);
//            } else { // good dart = NIL
//                m.unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(d);
//                ++nbBoundaryFaces;
//            }
//        }
//    }

    //reconstruct neighbourhood
//        unsigned int nbBoundaryFaces = 0 ;
//        //unsigned int nbFakeElements = 0;
//        std::vector<Dart> vFake;
//        vFake.reserve(1024);
//        for (Dart d = map.begin(); d != map.end(); map.next(d))
//        {
//            if (m.isMarked(d))
//            {
//                std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];

//                Dart good_dart = NIL;
//                for(typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
//                {
//                    if(map.template getEmbedding<VERTEX>(map.phi1(*it)) == map.template getEmbedding<VERTEX>(d)
//                      // needed because several tetrahedra can have 2 identical vertices
//                      && ( map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.phi1(map.phi1(d)))
//                        || map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.phi_1(d))))
//                    {
//                        good_dart = *it ;
//                    }
//                }

//                if (good_dart != NIL)
//                {
//                    unsigned int degD = map.faceDegree(d);
//                    unsigned int degGD = map.faceDegree(good_dart);

//                    if(degD == degGD)
//                    {
//                        map.sewVolumes(d, good_dart, false);
//                        m.template unmarkOrbit<FACE>(d);
//                    }
//                    else
//                    {
//                        // face of d is quad
//                        if(degD > degGD)
//                        {
//                            Dart another_d = map.phi1(map.phi1(d));
//                            std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(another_d)];

//                            Dart another_good_dart = NIL;
//                            for(typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && another_good_dart == NIL; ++it)
//                            {
//                                if(map.template getEmbedding<VERTEX>(map.phi1(*it)) == map.template getEmbedding<VERTEX>(another_d) &&
//                                   map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.phi1(map.phi1(another_d))))
//                                {
//                                    another_good_dart = *it ;
//                                }
//                            }

//                            //std::cout << "is nil ? " << (another_good_dart == NIL) << std::endl;

//                            if(another_good_dart != NIL)
//                            {
//                                Dart d1 = map.newFace(4, false);
//                                Dart d2 = map.newFace(3, false);
//                                Dart d3 = map.newFace(3, false);

//                                map.sewFaces(d1, d2, false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, d1, map.template getEmbedding<VERTEX>(another_d)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, d2, map.template getEmbedding<VERTEX>(map.phi1(d))) ;

//                                map.sewFaces(map.phi1(d1), map.phi_1(d2), false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, map.phi1(d1), map.template getEmbedding<VERTEX>(map.phi1(d))) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, map.phi_1(d2), map.template getEmbedding<VERTEX>(d)) ;

//                                map.sewFaces(map.phi1(d2), d3, false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, map.phi1(d2), map.template getEmbedding<VERTEX>(another_d)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, d3, map.template getEmbedding<VERTEX>(d)) ;

//                                map.sewFaces(map.phi_1(d1), map.phi1(d3), false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, map.phi_1(d1), map.template getEmbedding<VERTEX>(another_good_dart)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map, map.phi1(d3), map.template getEmbedding<VERTEX>(another_d)) ;

//                                map.sewFaces(map.phi1(map.phi1(d1)), map.phi_1(d3), false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi1(map.phi1(d1)), map.template getEmbedding<VERTEX>(d)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi_1(d3), map.template getEmbedding<VERTEX>(another_good_dart)) ;

//                                map.sewVolumes(map.phi1(d), d1, false);
//                                map.sewVolumes(good_dart, d2, false);
//                                map.sewVolumes(another_good_dart, map.phi_1(d3), false);

//                                m.template unmarkOrbit<FACE>(d);
//                                m.template unmarkOrbit<FACE>(good_dart);
//                                m.template unmarkOrbit<FACE>(another_good_dart);
//                            }
//                        }
//                        else
//                        {
//                            // face of d is tri

//                            Dart another_good_dart = map.phi1(map.phi1(d));
//                            std::vector<Dart>& vec = vecDartsPerVertex[another_good_dart];

//                            Dart another_d = NIL;
//                            for(typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && another_d == NIL; ++it)
//                            {
//                                if(map.template getEmbedding<VERTEX>(map.phi1(*it)) == map.template getEmbedding<VERTEX>(map.phi_1(another_good_dart)) &&
//                                   map.template getEmbedding<VERTEX>(map.phi1(map.phi1(*it))) == map.template getEmbedding<VERTEX>(map.phi1(another_good_dart)))
//                                {
//                                    another_d = *it ;
//                                }
//                            }

//                            //std::cout << "is nil ? " << (another_good_dart == NIL) << std::endl;

//                            if(another_d != NIL)
//                            {
//                                Dart d1 = map.newFace(4, false);
//                                Dart d2 = map.newFace(3, false);
//                                Dart d3 = map.newFace(3, false);

//                                map.sewFaces(d1, d2, false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,d1, map.template getEmbedding<VERTEX>(another_d)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,d2, map.template getEmbedding<VERTEX>(map.phi1(d))) ;

//                                map.sewFaces(map.phi1(d1), map.phi_1(d2), false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi1(d1), map.template getEmbedding<VERTEX>(map.phi1(d))) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi_1(d2), map.template getEmbedding<VERTEX>(d)) ;

//                                map.sewFaces(map.phi1(d2), d3, false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi1(d2), map.template getEmbedding<VERTEX>(another_d)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,d3, map.template getEmbedding<VERTEX>(d)) ;

//                                map.sewFaces(map.phi_1(d1), map.phi1(d3), false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi_1(d1), map.template getEmbedding<VERTEX>(another_good_dart)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi1(d3), map.template getEmbedding<VERTEX>(another_d)) ;

//                                map.sewFaces(map.phi1(map.phi1(d1)), map.phi_1(d3), false);
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi1(map.phi1(d1)), map.template getEmbedding<VERTEX>(d)) ;
//                                Algo::Topo::setOrbitEmbedding<VERTEX, MAP>(map,map.phi_1(d3), map.template getEmbedding<VERTEX>(another_good_dart)) ;

//                                map.sewVolumes(d, d2, false);
//                                map.sewVolumes(another_d, d3, false);
//                                map.sewVolumes(good_dart, d1, false);

//                                m.template unmarkOrbit<FACE>(d);
//                                m.template unmarkOrbit<FACE>(another_d);
//                                m.template unmarkOrbit<FACE>(good_dart);
//                            }
//                        }
//                    }
//                }
//                else
//                {
//                    m.template unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(d);
//                    ++nbBoundaryFaces;
//                }
//            }
//        }
    //        std::exit(20);

    //reconstruct neighbourhood
    unsigned int nbBoundaryFaces = 0 ;
    for (Dart d = map.begin(); d != map.end(); map.next(d))
    {
        if (m.isMarked(d))
        {
            std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)];

            Dart good_dart = NIL;
            for(typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
            {
                if(map.template getEmbedding<VERTEX>(map.phi1(*it)) == map.template getEmbedding<VERTEX>(d) &&
                   map.template getEmbedding<VERTEX>(map.phi_1(*it)) == map.template getEmbedding<VERTEX>(map.phi1(map.phi1(d))))
                {
                    good_dart = *it ;
                }
            }

            if (good_dart != NIL)
            {
                unsigned int degD = map.faceDegree(d);
                unsigned int degGD = map.faceDegree(good_dart);

                //				std::cout << "degD = " << degD << std::endl;
                //				std::cout << "degGD = " << degGD << std::endl << std::endl;

                if(degD < degGD)
                {
                    Dart dt = map.phi1(good_dart);
                    map.PFP::MAP::ParentMap::splitFace(dt,map.phi_1(good_dart));

                    map.template initDartEmbedding<VERTEX>(map.phi1(good_dart), map.template getEmbedding<VERTEX>(dt)) ;
                    map.template initDartEmbedding<VERTEX>(map.phi_1(dt), map.template getEmbedding<VERTEX>(map.phi_1(good_dart))) ;

                    m.mark(map.phi1(good_dart));
                    m.mark(map.phi2(map.phi1(good_dart)));

                    unsigned int emb2 = map.template getEmbedding<VERTEX>(map.phi1(map.phi1(d)));
                    vecDartsPerVertex[emb2].push_back(map.phi2(map.phi1(good_dart)));

                    unsigned int emb1 = map.template getEmbedding<VERTEX>(d);
                    vecDartsPerVertex[emb1].push_back(map.phi1(good_dart));

                    //m.unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(d);
                    //m.unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(good_dart);
                }
                else if(degD > degGD)
                {
                    Dart dt = map.phi1(map.phi1(d));
                    map.PFP::MAP::ParentMap::splitFace(d,dt);

                    map.template initDartEmbedding<VERTEX>(map.phi_1(dt), map.template getEmbedding<VERTEX>(d)) ;
                    map.template initDartEmbedding<VERTEX>(map.phi_1(d), map.template getEmbedding<VERTEX>(dt)) ;

                    m.mark(map.phi_1(d));
                    m.mark(map.phi2(map.phi_1(d)));

                    //ne change rien sur l'exemple test
                    unsigned int emb1 = map.template getEmbedding<VERTEX>(map.phi1(map.phi1(good_dart)));
                    vecDartsPerVertex[emb1].push_back(map.phi2(map.phi_1(d)));

                    unsigned int emb2 = map.template getEmbedding<VERTEX>(good_dart);
                    vecDartsPerVertex[emb2].push_back(map.phi_1(d));

                    //m.unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(d);
                    //m.unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(good_dart);
                }
                else if(degD == degGD)
                {
                    map.sewVolumes(d, good_dart, false);
                    m.template unmarkOrbit<FACE>(d);
                }
                //				else if(degD > 3 && degGD > 3)
                //				{
                //					if(map.template getEmbedding<VERTEX>(map.phi1(map.phi1(good_dart))) != map.template getEmbedding<VERTEX>(map.phi_1(d)))
                //					{
                //						std::cout << "2 faces quad" << std::endl;
                //						Dart dtgd = map.phi1(good_dart);
                //						map.PFP::MAP::ParentMap::splitFace(dtgd,map.phi_1(good_dart));

                //						map.template initDartEmbedding<VERTEX>(map.phi1(good_dart), map.template getEmbedding<VERTEX>(dtgd)) ;
                //						map.template initDartEmbedding<VERTEX>(map.phi_1(dtgd), map.template getEmbedding<VERTEX>(map.phi_1(good_dart))) ;


                //						Dart dt = map.phi1(map.phi1(d));
                //						map.PFP::MAP::ParentMap::splitFace(d,dt);

                //						map.template initDartEmbedding<VERTEX>(map.phi_1(dt), map.template getEmbedding<VERTEX>(d)) ;
                //						map.template initDartEmbedding<VERTEX>(map.phi_1(d), map.template getEmbedding<VERTEX>(dt)) ;
                //					}
                //				}

                //				map.sewVolumes(d, good_dart, false);
                //				m.unmarkOrbit<FACE>(d);
            }
            else
            {
                m.template unmarkOrbit<PFP::MAP::FACE_OF_PARENT>(d);
                ++nbBoundaryFaces;
            }
        }
    }

//        map.check();
    if (nbBoundaryFaces > 0) {
        unsigned int nbH =  map.closeMap();
        CGoGNout << "Map closed (" << nbBoundaryFaces << " boundary faces / " << nbH << " holes)" << CGoGNendl;
    }

    return true;
}


template <typename PFP>
bool importMesh(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, bool mergeCloseVertices)
{
    MeshTablesVolume<PFP> mtv(map);

    if(!mtv.importMesh(filename, attrNames))
        return false;

    //sif(mergeCloseVertices)
    //mtv.mergeCloseVertices();

    return importMesh<PFP>(map, mtv);
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

} // namespace Volume


} // namespace Algo

} // namespace CGoGN
