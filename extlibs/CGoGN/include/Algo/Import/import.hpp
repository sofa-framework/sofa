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
#include <boost/function.hpp>

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
    boost::function<unsigned (Dart)> fsetemb1= bl::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb1));
    unsigned int vemb2 = EMBNULL;
    //    auto fsetemb2 = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb2); };
    boost::function<unsigned (Dart)> fsetemb2 = bl::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb2));

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
    boost::function<unsigned (Dart)> fsetemb1 = bl::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb1));
    unsigned int vemb2 = EMBNULL;
    //	auto fsetemb2 = [&] (Dart d) { map.template initDartEmbedding<VERTEX>(d, vemb2); };
    boost::function<unsigned (Dart)> fsetemb2 = bl::bind(&MAP::template initDartEmbedding<VERTEX>, bl::_1, boost::ref(vemb2));

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
Dart createStampVolume(typename PFP::MAP& map)
{
    Dart dquad = map.newFace(4, false);
    Dart dtri1 = map.newFace(3, false);
    Dart dtri2 = map.newFace(3, false);

    map.sewFaces(dtri1, dtri2, false);
    map.sewFaces(map.phi1(dtri1), map.phi_1(dquad), false);
    map.sewFaces(map.phi_1(dtri2), dquad, false);
    map.sewFaces(map.phi1(dtri2), map.phi1(dquad), false);
    map.sewFaces(map.phi_1(dtri1), map.phi1(map.phi1(dquad)), false);

    return dquad;
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

    DartMarker<typename PFP::MAP> m(map) ;
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
        if (VT == MeshTablesVolume<PFP>::CONNECTOR) {
            Dart d = createStampVolume<PFP>(map);
            //Embed four vertices
            unsigned int vemb = EMBNULL;
            // 1.
            vemb = vertexEmbeddingsBuffer[0];		// get embedding
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb))) );
            Dart dd = d;
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd);

            // 2.
            d = map.phi1(d);
            vemb = vertexEmbeddingsBuffer[1];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb))) );
            dd = d;
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd);

            // 3.
            d = map.phi1(d);
            vemb = vertexEmbeddingsBuffer[2];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb))) );
            dd = d;
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd);

            // 4.
            d = map.phi1(d);
            vemb = vertexEmbeddingsBuffer[3];
            map.template foreach_dart_of_orbit<MAP::VERTEX_OF_PARENT>(d,  ( bl::bind(&MAP::template initDartEmbedding<VERTEX>,boost::ref(map), bl::_1, boost::ref(vemb))) );
            dd = d;
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd); dd = map.phi1(map.phi2(dd));
            vecDartsPerVertex[vemb].push_back(dd); m.mark(dd);



        } else if (VT == MeshTablesVolume<PFP>::TETRAHEDRON) //tetrahedral case
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

        }
    }

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

                //std::cout << "degD = " << degD << " et degGD = " << degGD << std::endl;
                if(degD == degGD)
                {
                    map.sewVolumes(d, good_dart, false);
                    m.template unmarkOrbit<FACE>(d);
                }
//                else
//                    std::cout << "erreur : degD != degGD" << std::endl;
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
