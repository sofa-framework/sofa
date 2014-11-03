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
#include <cmath>

#include "Geometry/vector_gen.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"
#include "Algo/Geometry/centroid.h"
#include "Topology/generic/autoAttributeHandler.h"
#include "Algo/Geometry/normal.h"
#include "Algo/Geometry/basic.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

inline ExplodeVolumeRender::ExplodeVolumeRender(bool withColorPerFace, bool withExplodeFace, bool withSmoothFaces):
    m_cpf(withColorPerFace),m_ef(withExplodeFace),m_smooth(withSmoothFaces),
    m_nbTris(0), m_nbLines(0), m_globalColor(0.9f,0.5f,0.0f)//m_globalColor(0.7f,0.7f,0.7f)
{
    m_vboPos = new Utils::VBO();
    m_vboPos->setDataSize(3);

    m_vboColors = new Utils::VBO();
    m_vboColors->setDataSize(3);

    m_vboPosLine = new Utils::VBO();
    m_vboPosLine->setDataSize(3);

    if (m_smooth)
    {
        m_shaderS = new Utils::ShaderExplodeSmoothVolumes(withColorPerFace,withExplodeFace);
        m_shader = NULL;
        m_vboNormals = new Utils::VBO();
        m_vboNormals->setDataSize(3);
    }
    else
    {
        m_shader = new Utils::ShaderExplodeVolumes(withColorPerFace,withExplodeFace);
        m_shaderS = NULL;
        m_vboNormals = NULL;
    }

    m_shaderL = new Utils::ShaderExplodeVolumesLines();
    m_shaderL->setColor(Geom::Vec4f(1.0f,1.0f,1.0f,0.0f));

}

inline ExplodeVolumeRender::~ExplodeVolumeRender()
{
    delete m_vboPos;
    delete m_vboColors;
    delete m_vboPosLine;

    if (m_vboNormals != NULL)
        delete m_vboNormals;
    if (m_shader != NULL)
        delete m_shader;
    if (m_shaderS != NULL)
        delete m_shaderS;
    delete m_shaderL;
}


//template<typename PFP>
//void ExplodeVolumeRender::computeFace(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& positions,
//                                     const typename PFP::VEC3& centerFace, const typename PFP::VEC3& centerNormalFace,
//									 std::vector<typename PFP::VEC3>& vertices, std::vector<typename PFP::VEC3>& normals)
//{
//    computeFaceGen<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map,d,positions,centerFace, centerNormalFace, vertices, normals);
//}


template<typename PFP, typename EMBV>
void ExplodeVolumeRender::computeFace(typename PFP::MAP& map, Dart d, const EMBV& positions,
                                      const typename PFP::VEC3& centerFace, const typename PFP::VEC3& /*centerNormalFace*/,
                                      std::vector<typename PFP::VEC3>& vertices, std::vector<typename PFP::VEC3>& normals)
{
    //typedef typename PFP::VEC3 VEC3;
    typedef typename EMBV::DATA_TYPE VEC3;
    typedef typename PFP::REAL REAL;

    normals.clear();
    vertices.clear();
    Dart a = d;
    do
    {
        VEC3 v1 = positions[a] - centerFace;
        v1.normalize();
        Dart e = map.phi1(a);
        VEC3 v2 = positions[e] - centerFace;
        v2.normalize();
        VEC3 N = v1.cross(v2);
        normals.push_back(N);
        vertices.push_back(positions[a]);
        a = e;
    } while (a != d);

    unsigned int nb = normals.size();
    VEC3 Ntemp = normals[0];
    normals[0] += normals[nb-1];
    normals[0].normalize();
    for (unsigned int i=1; i!=nb ; ++i)
    {
        VEC3 Ntemp2 = normals[i];
        normals[i] += Ntemp;
        normals[i].normalize();
        Ntemp = Ntemp2;
    }
}

//template<typename PFP>
//void ExplodeVolumeRender::updateSmooth(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions, const VolumeAttribute<typename PFP::VEC3>& colorPerXXX)
//{
//    updateSmoothGen<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map,positions,colorPerXXX);
//}

template<typename PFP, typename V_ATT, typename W_ATT>
void ExplodeVolumeRender::updateSmooth(typename PFP::MAP& map, const V_ATT& positions, const W_ATT& colorPerXXX)
{
    typedef typename V_ATT::DATA_TYPE VEC3;
    typedef typename W_ATT::DATA_TYPE COL3;
    typedef typename PFP::MAP MAP;
    typedef typename PFP::REAL REAL;
    typedef Geom::Vec3f VEC3F;

    VolumeAutoAttribute<VEC3, MAP> centerVolumes(map, "centerVolumes");
    Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(map, positions, centerVolumes);

    std::vector<VEC3F> buffer;
    buffer.reserve(16384);

    std::vector<VEC3F> bufferColors;
    bufferColors.reserve(16384);
    std::vector<VEC3F> bufferNormals;
    bufferNormals.reserve(16384);

    std::vector<VEC3> normals;
    normals.reserve(20);
    std::vector<VEC3> vertices;
    vertices.reserve(20);

    TraversorCell<MAP, MAP::FACE_OF_PARENT> traFace(map);
    for (Dart d = traFace.begin(), end = traFace.end() ; d != end ; d = traFace.next() )
        //	foreach_cell<MAP::FACE_OF_PARENT>(map, [&] (Cell<MAP::FACE_OF_PARENT> d)
    {
        // compute normals
        VEC3 centerFace = Algo::Surface::Geometry::faceCentroidELW<PFP>(map, d, positions);
        VEC3 centerNormalFace = Algo::Surface::Geometry::newellNormal<PFP>(map, d, positions);

        computeFace<PFP>(map,d,positions,centerFace,centerNormalFace,vertices,normals);

        VEC3F volCol = PFP::toVec3f(colorPerXXX[d]);

        unsigned int nbs = vertices.size();
        // just to have more easy algo further
        vertices.push_back(vertices.front());
        normals.push_back(normals.front());

        typename std::vector<VEC3>::iterator iv  = vertices.begin();
        typename std::vector<VEC3>::iterator in  = normals.begin();

        if (nbs == 3)
        {
            buffer.push_back(PFP::toVec3f(centerVolumes[d]));
            bufferColors.push_back(PFP::toVec3f(centerFace));
            bufferNormals.push_back(PFP::toVec3f(centerNormalFace)); // unsused just for fill

            buffer.push_back(PFP::toVec3f(*iv++));
            bufferNormals.push_back(PFP::toVec3f(*in++));
            bufferColors.push_back(volCol);

            buffer.push_back(PFP::toVec3f(*iv++));
            bufferNormals.push_back(PFP::toVec3f(*in++));
            bufferColors.push_back(volCol);

            buffer.push_back(PFP::toVec3f(*iv++));
            bufferNormals.push_back(PFP::toVec3f(*in++));
            bufferColors.push_back(volCol);
        }
        else
        {
            for (unsigned int i=0; i<nbs; ++i)
            {
                buffer.push_back(PFP::toVec3f(centerVolumes[d]));
                bufferColors.push_back(PFP::toVec3f(centerFace));
                bufferNormals.push_back(PFP::toVec3f(centerNormalFace)); // unsused just for fill

                buffer.push_back(PFP::toVec3f(centerFace));
                bufferColors.push_back(volCol);
                bufferNormals.push_back(PFP::toVec3f(centerNormalFace));

                buffer.push_back(PFP::toVec3f(*iv++));
                bufferNormals.push_back(PFP::toVec3f(*in++));
                bufferColors.push_back(volCol);

                buffer.push_back(PFP::toVec3f(*iv));
                bufferNormals.push_back(PFP::toVec3f(*in));
                bufferColors.push_back(volCol);
            }
        }
    }
    //	,false,thread); ????

    m_nbTris = buffer.size()/4;

    m_vboPos->allocate(buffer.size());
    VEC3F* ptrPos = reinterpret_cast<VEC3F*>(m_vboPos->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));
    m_vboPos->releasePtr();
    m_shaderS->setAttributePosition(m_vboPos);

    m_vboColors->allocate(bufferColors.size());
    VEC3F* ptrCol = reinterpret_cast<VEC3F*>(m_vboColors->lockPtr());
    memcpy(ptrCol,&bufferColors[0],bufferColors.size()*sizeof(VEC3F));
    m_vboColors->releasePtr();
    m_shaderS->setAttributeColor(m_vboColors);

    m_vboNormals->allocate(bufferNormals.size());
    VEC3F* ptrNorm = reinterpret_cast<VEC3F*>(m_vboNormals->lockPtr());
    memcpy(ptrNorm,&bufferNormals[0],bufferNormals.size()*sizeof(VEC3F));
    m_vboNormals->releasePtr();
    m_shaderS->setAttributeNormal(m_vboNormals);

    buffer.clear();

    TraversorCell<typename PFP::MAP, PFP::MAP::EDGE_OF_PARENT> traEdge(map);
    for (Dart d = traEdge.begin(), end = traEdge.end() ; d != end ; d = traEdge.next() )
        //	foreach_cell<PFP::MAP::EDGE_OF_PARENT>(map, [&] (Cell<PFP::MAP::EDGE_OF_PARENT> c)
    {
        buffer.push_back(PFP::toVec3f(centerVolumes[d]));
        buffer.push_back(PFP::toVec3f(positions[d]));
        buffer.push_back(PFP::toVec3f(positions[map.phi1(d)]));
    }
    //	,false,thread); ????

    m_nbLines = buffer.size()/3;

    m_vboPosLine->allocate(buffer.size());

    ptrPos = reinterpret_cast<VEC3F*>(m_vboPosLine->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));

    m_vboPosLine->releasePtr();
    m_shaderL->setAttributePosition(m_vboPosLine);
}


//template<typename PFP>
//void ExplodeVolumeRender::updateSmooth(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions)
//{
//    updateSmoothGen<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map, positions);
//}

template<typename PFP, typename EMBV>
void ExplodeVolumeRender::updateSmooth(typename PFP::MAP& map, const EMBV& positions)
{
    typedef typename EMBV::DATA_TYPE VEC3;
    typedef typename PFP::MAP MAP;
    typedef typename PFP::REAL REAL;
    typedef typename Geom::Vec3f VEC3F;

    VolumeAutoAttribute<VEC3, MAP> centerVolumes(map, "centerVolumes");
    Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(map, positions, centerVolumes);

    std::vector<VEC3F> buffer;
    buffer.reserve(16384);

    std::vector<VEC3F> bufferColors;
    bufferColors.reserve(16384);
    std::vector<VEC3F> bufferNormals;
    bufferNormals.reserve(16384);

    std::vector<VEC3> normals;
    bufferNormals.reserve(20);
    std::vector<VEC3> vertices;
    bufferNormals.reserve(20);

    TraversorCell<MAP, MAP::FACE_OF_PARENT> traFace(map);
    for (Dart d = traFace.begin(), end = traFace.end() ; d != end ; d = traFace.next() )
        //	foreach_cell<MAP::FACE_OF_PARENT>(map, [&] (Cell<MAP::FACE_OF_PARENT> d)
    {
        // compute normals
        VEC3 centerFace = Algo::Surface::Geometry::faceCentroidELW<PFP>(map, d, positions);
        VEC3 centerNormalFace = Algo::Surface::Geometry::newellNormal<PFP>(map, d, positions);

        computeFace<PFP>(map,d,positions,centerFace,centerNormalFace,vertices,normals);

        unsigned int nbs = vertices.size();
        // just to have more easy algo further
        vertices.push_back(vertices.front());
        normals.push_back(normals.front());

        typename std::vector<VEC3>::iterator iv  = vertices.begin();
        typename std::vector<VEC3>::iterator in  = normals.begin();

        if (nbs == 3)
        {
            buffer.push_back(PFP::toVec3f(centerVolumes[d]));
            bufferColors.push_back(PFP::toVec3f(centerFace));
            bufferNormals.push_back(PFP::toVec3f(centerNormalFace)); // unsused just for fill

            buffer.push_back(PFP::toVec3f(*iv++));
            bufferNormals.push_back(PFP::toVec3f(*in++));
            bufferColors.push_back(m_globalColor);

            buffer.push_back(PFP::toVec3f(*iv++));
            bufferNormals.push_back(PFP::toVec3f(*in++));
            bufferColors.push_back(m_globalColor);

            buffer.push_back(PFP::toVec3f(*iv++));
            bufferNormals.push_back(PFP::toVec3f(*in++));
            bufferColors.push_back(m_globalColor);
        }
        else
        {
            for (unsigned int i=0; i<nbs; ++i)
            {
                buffer.push_back(PFP::toVec3f(centerVolumes[d]));
                bufferColors.push_back(PFP::toVec3f(centerFace));
                bufferNormals.push_back(PFP::toVec3f(centerNormalFace)); // unsused just for fill

                buffer.push_back(PFP::toVec3f(centerFace));
                bufferColors.push_back(m_globalColor);
                bufferNormals.push_back(PFP::toVec3f(centerNormalFace));

                buffer.push_back(PFP::toVec3f(*iv++));
                bufferNormals.push_back(PFP::toVec3f(*in++));
                bufferColors.push_back(m_globalColor);

                buffer.push_back(PFP::toVec3f(*iv));
                bufferNormals.push_back(PFP::toVec3f(*in));
                bufferColors.push_back(m_globalColor);
            }
        }
    } // false,thread) ???

    m_nbTris = buffer.size()/4;

    m_vboPos->allocate(buffer.size());
    VEC3F* ptrPos = reinterpret_cast<VEC3F*>(m_vboPos->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));
    m_vboPos->releasePtr();
    m_shaderS->setAttributePosition(m_vboPos);

    m_vboColors->allocate(bufferColors.size());
    VEC3F* ptrCol = reinterpret_cast<VEC3F*>(m_vboColors->lockPtr());
    memcpy(ptrCol,&bufferColors[0],bufferColors.size()*sizeof(VEC3F));
    m_vboColors->releasePtr();
    m_shaderS->setAttributeColor(m_vboColors);

    m_vboNormals->allocate(bufferNormals.size());
    VEC3F* ptrNorm = reinterpret_cast<VEC3F*>(m_vboNormals->lockPtr());
    memcpy(ptrNorm,&bufferNormals[0],bufferNormals.size()*sizeof(VEC3F));
    m_vboNormals->releasePtr();
    m_shaderS->setAttributeNormal(m_vboNormals);

    buffer.clear();

    TraversorCell<typename PFP::MAP, PFP::MAP::EDGE_OF_PARENT> traEdge(map);
    for (Dart d = traEdge.begin(), end = traEdge.end() ; d != end ; d = traEdge.next() )
        //	foreach_cell<MAP::EDGE_OF_PARENT>(map, [&] (Cell<MAP::EDGE_OF_PARENT> c)
    {
        buffer.push_back(PFP::toVec3f(centerVolumes[d]));
        buffer.push_back(PFP::toVec3f(positions[d]));
        buffer.push_back(PFP::toVec3f(positions[map.phi1(d)]));
    }

    m_nbLines = buffer.size()/3;

    m_vboPosLine->allocate(buffer.size());

    ptrPos = reinterpret_cast<VEC3F*>(m_vboPosLine->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));

    m_vboPosLine->releasePtr();
    m_shaderL->setAttributePosition(m_vboPosLine);
}



//template<typename PFP>
//void ExplodeVolumeRender::updateData(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions, const VolumeAttribute<typename PFP::VEC3>& colorPerXXX)
//{
//    updateDataGen<PFP,  const VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map, positions, colorPerXXX);
//}

template<typename PFP, typename V_ATT, typename W_ATT>
void ExplodeVolumeRender::updateData(typename PFP::MAP& map, const V_ATT& positions, const W_ATT& colorPerXXX)
{
    if (!m_cpf)
    {
        CGoGNerr<< "ExplodeVolumeRender: problem wrong update fonction use the other (without VolumeAttribute parameter)" << CGoGNendl;
        return;
    }

    if (m_smooth)
    {
        updateSmooth<PFP>(map,positions,colorPerXXX);
        return;
    }

    //typedef typename PFP::VEC3 VEC3;
    typedef typename V_ATT::DATA_TYPE VEC3;
    typedef typename PFP::MAP MAP;
    typedef typename PFP::REAL REAL;
    typedef Geom::Vec3f VEC3F;

    VolumeAutoAttribute<VEC3, MAP> centerVolumes(map, "centerVolumes");
    Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(map, positions, centerVolumes);

    std::vector<VEC3F> buffer;
    buffer.reserve(16384);

    std::vector<VEC3F> bufferColors;

    bufferColors.reserve(16384);

    TraversorCell<MAP, MAP::FACE_OF_PARENT> traFace(map);
    for (Dart d = traFace.begin(), end = traFace.end() ; d != end ; d = traFace.next() )
        ////	foreach_cell<MAP::FACE_OF_PARENT>(map, [&] (Cell<MAP::FACE_OF_PARENT> d)
    {
        VEC3F centerFace = PFP::toVec3f(Algo::Surface::Geometry::faceCentroidELW<PFP>(map, d, positions));
        VEC3F volColor = PFP::toVec3f(colorPerXXX[d]);

        Dart b = d;
        Dart c = map.phi1(b);
        Dart a = map.phi1(c);

        if (map.phi1(a) == d)
        {
            buffer.push_back(PFP::toVec3f(centerVolumes[d]));
            bufferColors.push_back(centerFace);

            buffer.push_back(PFP::toVec3f(positions[b]));
            bufferColors.push_back(volColor);
            buffer.push_back(PFP::toVec3f(positions[c]));
            bufferColors.push_back(volColor);
            c = map.phi1(c);
            buffer.push_back(PFP::toVec3f(positions[c]));
            bufferColors.push_back(volColor);
        }
        else
        {

            // loop to cut a polygon in triangle on the fly (ceter point method)
            do
            {
                buffer.push_back(PFP::toVec3f(centerVolumes[d]));
                bufferColors.push_back(centerFace);

                buffer.push_back(centerFace);
                bufferColors.push_back(volColor);

                buffer.push_back(PFP::toVec3f(positions[b]));
                bufferColors.push_back(volColor);
                buffer.push_back(PFP::toVec3f(positions[c]));
                bufferColors.push_back(volColor);
                b = c;
                c = map.phi1(b);
            } while (b != d);
        }
    }

    m_nbTris = buffer.size()/4;

    m_vboPos->allocate(buffer.size());
    VEC3F* ptrPos = reinterpret_cast<VEC3F*>(m_vboPos->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));
    m_vboPos->releasePtr();
    m_shader->setAttributePosition(m_vboPos);

    m_vboColors->allocate(bufferColors.size());
    VEC3F* ptrCol = reinterpret_cast<VEC3F*>(m_vboColors->lockPtr());
    memcpy(ptrCol,&bufferColors[0],bufferColors.size()*sizeof(VEC3F));
    m_vboColors->releasePtr();
    m_shader->setAttributeColor(m_vboColors);

    buffer.clear();

    ////	foreach_cell<MAP::EDGE_OF_PARENT>(map, [&] (Cell<MAP::EDGE_OF_PARENT> c)
    TraversorCell<typename PFP::MAP, PFP::MAP::EDGE_OF_PARENT> traEdge(map);
    for (Dart d = traEdge.begin(), end = traEdge.end() ; d != end ; d = traEdge.next() )
    {
        buffer.push_back(PFP::toVec3f(centerVolumes[d]));
        buffer.push_back(PFP::toVec3f(positions[d]));
        buffer.push_back(PFP::toVec3f(positions[map.phi1(d)]));
    }
    m_nbLines = buffer.size()/3;

    m_vboPosLine->allocate(buffer.size());

    ptrPos = reinterpret_cast<VEC3F*>(m_vboPosLine->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));

    m_vboPosLine->releasePtr();
    m_shaderL->setAttributePosition(m_vboPosLine);
}

//template<typename PFP>
//void ExplodeVolumeRender::updateData(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions)
//{
//    updateDataGen<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map, positions);
//}

template<typename PFP, typename EMBV>
void ExplodeVolumeRender::updateData(typename PFP::MAP& map, const EMBV& positions)
{
    if (m_smooth)
    {
        updateSmooth<PFP, EMBV>(map,positions);
        return;
    }

    typedef typename EMBV::DATA_TYPE VEC3;
    typedef typename PFP::MAP MAP;
    typedef typename PFP::REAL REAL;
    typedef VEC3 VEC3F;

    VolumeAutoAttribute<VEC3, MAP> centerVolumes(map, "centerVolumes");
    Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(map, positions, centerVolumes);

    std::vector<VEC3F> buffer;
    buffer.reserve(16384);

    std::vector<VEC3F> bufferColors;
    bufferColors.reserve(16384);

    TraversorCell<MAP, MAP::FACE_OF_PARENT> traFace(map);
    for (Dart d = traFace.begin(), end = traFace.end() ; d != end ; d = traFace.next() )
    {
        VEC3F centerFace = PFP::toVec3f(Algo::Surface::Geometry::faceCentroidELW<PFP>(map, d, positions));

        Dart b = d;
        Dart c = map.phi1(b);
        Dart a = map.phi1(c);

        if (map.phi1(a) == d)
        {
            buffer.push_back(PFP::toVec3f(centerVolumes[d]));
            bufferColors.push_back(centerFace);

            buffer.push_back(PFP::toVec3f(positions[b]));
            bufferColors.push_back(m_globalColor);
            buffer.push_back(PFP::toVec3f(positions[c]));
            bufferColors.push_back(m_globalColor);
            c = map.phi1(c);
            buffer.push_back(PFP::toVec3f(positions[c]));
            bufferColors.push_back(m_globalColor);
        }
        else
        {

            // loop to cut a polygon in triangle on the fly (ceter point method)
            do
            {
                buffer.push_back(PFP::toVec3f(centerVolumes[d]));
                bufferColors.push_back(centerFace);

                buffer.push_back(centerFace);
                bufferColors.push_back(m_globalColor);

                buffer.push_back(PFP::toVec3f(positions[b]));
                bufferColors.push_back(m_globalColor);
                buffer.push_back(PFP::toVec3f(positions[c]));
                bufferColors.push_back(m_globalColor);
                b = c;
                c = map.phi1(b);
            } while (b != d);
        }
    }

    m_nbTris = buffer.size()/4;

    m_vboPos->allocate(buffer.size());
    VEC3F* ptrPos = reinterpret_cast<VEC3F*>(m_vboPos->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));
    m_vboPos->releasePtr();
    m_shader->setAttributePosition(m_vboPos);

    m_vboColors->allocate(bufferColors.size());
    VEC3F* ptrCol = reinterpret_cast<VEC3F*>(m_vboColors->lockPtr());
    memcpy(ptrCol,&bufferColors[0],bufferColors.size()*sizeof(VEC3F));
    m_vboColors->releasePtr();
    m_shader->setAttributeColor(m_vboColors);

    buffer.clear();

    TraversorCell<typename PFP::MAP, PFP::MAP::EDGE_OF_PARENT> traEdge(map);
    for (Dart d = traEdge.begin(), end = traEdge.end() ; d != end ; d = traEdge.next() )
        //	foreach_cell<MAP::EDGE_OF_PARENT>(map, [&] (Cell<MAP::EDGE_OF_PARENT> c)
    {
        buffer.push_back(PFP::toVec3f(centerVolumes[d]));
        buffer.push_back(PFP::toVec3f(positions[d]));
        buffer.push_back(PFP::toVec3f(positions[map.phi1(d)]));
    }

    m_nbLines = buffer.size()/3;

    m_vboPosLine->allocate(buffer.size());

    ptrPos = reinterpret_cast<VEC3F*>(m_vboPosLine->lockPtr());
    memcpy(ptrPos,&buffer[0],buffer.size()*sizeof(VEC3F));

    m_vboPosLine->releasePtr();
    m_shaderL->setAttributePosition(m_vboPosLine);
}



inline void ExplodeVolumeRender::drawFaces()
{
    if (m_smooth)
    {
        m_shaderS->enableVertexAttribs();
        glDrawArrays(GL_LINES_ADJACENCY_EXT , 0 , m_nbTris*4 );
        m_shaderS->disableVertexAttribs();
    }
    else
    {
        m_shader->enableVertexAttribs();
        glDrawArrays(GL_LINES_ADJACENCY_EXT , 0 , m_nbTris*4 );
        m_shader->disableVertexAttribs();
    }
}

inline void ExplodeVolumeRender::drawEdges()
{

    m_shaderL->enableVertexAttribs();
    glDrawArrays(GL_TRIANGLES , 0 , m_nbLines*3 );
    m_shaderL->disableVertexAttribs();
}

inline void ExplodeVolumeRender::setExplodeVolumes(float explode)
{
    m_explodeV = explode;
    if (m_smooth)
        m_shaderS->setExplodeVolumes(explode);
    else
        m_shader->setExplodeVolumes(explode);
    m_shaderL->setExplodeVolumes(explode);
}

inline void ExplodeVolumeRender::setExplodeFaces(float explode)
{
    if (m_smooth)
        m_shaderS->setExplodeFaces(explode);
    else
        m_shader->setExplodeFaces(explode);
}

inline void ExplodeVolumeRender::setClippingPlane(const Geom::Vec4f& p)
{
    m_clipPlane = p;
    if (m_smooth)
        m_shaderS->setClippingPlane(p);
    else
        m_shader->setClippingPlane(p);
    m_shaderL->setClippingPlane(p);
}

inline void ExplodeVolumeRender::setNoClippingPlane()
{
    m_clipPlane = Geom::Vec4f(1.0f,1.0f,1.0f,-99999999.9f);
    if (m_smooth)
        m_shaderS->setClippingPlane(m_clipPlane);
    else
        m_shader->setClippingPlane(m_clipPlane);
    m_shaderL->setClippingPlane(m_clipPlane);
}

inline void ExplodeVolumeRender::setAmbiant(const Geom::Vec4f& ambiant)
{
    if (m_smooth)
        m_shaderS->setAmbiant(ambiant);
    else
        m_shader->setAmbiant(ambiant);
}

inline void ExplodeVolumeRender::setBackColor(const Geom::Vec4f& color)
{
    if (!m_smooth)
        m_shader->setBackColor(color);
}

inline void ExplodeVolumeRender::setLightPosition(const Geom::Vec3f& lp)
{
    if (m_smooth)
        m_shaderS->setLightPosition(lp);
    else
        m_shader->setLightPosition(lp);
}

inline void ExplodeVolumeRender::setColorLine(const Geom::Vec4f& col)
{
    m_shaderL->setColor(col);
}

inline Utils::GLSLShader* ExplodeVolumeRender::shaderFaces()
{
    if (m_smooth)
        return m_shaderS;
    return m_shader;
}

inline Utils::GLSLShader* ExplodeVolumeRender::shaderLines()
{
    return m_shaderL;
}

inline void ExplodeVolumeRender::svgoutEdges(const std::string& filename, const glm::mat4& model, const glm::mat4& proj,float af)
{
    Utils::SVG::SVGOut svg(filename,model,proj);
    svg.setAttenuationFactor(af);
    toSVG(svg);
    svg.write();
}

inline void ExplodeVolumeRender::toSVG(Utils::SVG::SVGOut& svg)
{

    Utils::SVG::SvgGroup* svg2 = new Utils::SVG::SvgGroup("alpha2", svg.m_model, svg.m_proj);

    Geom::Vec3f* ptr = reinterpret_cast<Geom::Vec3f*>(m_vboPosLine->lockPtr());
    svg2->setWidth(1.0f);
    svg2->beginLines();

    const Geom::Vec4f& col4 = m_shaderL->getColor();
    Geom::Vec3f col3(col4[0],col4[1],col4[2]);

    float XexplV = (1.0f-m_explodeV);
    for (unsigned int i=0; i<m_nbLines; ++i)
    {
        Geom::Vec3f C = ptr[3*i];
        Geom::Vec4f C4 = Geom::Vec4f(C[0],C[1],C[2],1.0f);
        if (m_clipPlane*C4 <=0.0f)
        {
            Geom::Vec3f P = XexplV*C + m_explodeV*ptr[3*i+1];
            Geom::Vec3f Q = XexplV*C + m_explodeV*ptr[3*i+2];
            svg2->addLine(P, Q, col3);
        }
    }
    svg2->endLines();
    m_vboPosLine->releasePtr();
    svg.addGroup(svg2);

}


}//end namespace VBO

}//end namespace Algo

}//end namespace Render

}//end namespace CGoGN
