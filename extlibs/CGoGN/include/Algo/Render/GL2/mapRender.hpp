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
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorCell.h"

#include "Geometry/intersection.h"
#include "Algo/Geometry/normal.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

inline bool MapRender::cmpVP(VertexPoly* lhs, VertexPoly* rhs)
{
    //	 return lhs->value < rhs->value;

    if (fabs(lhs->value - rhs->value)<0.2f)
        return lhs->length < rhs->length;
    return lhs->value < rhs->value;
}

template<typename VEC3>
bool MapRender::inTriangle(const VEC3& P, const VEC3& normal, const VEC3& Ta,  const VEC3& Tb, const VEC3& Tc)
{
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
void MapRender::recompute2Ears(const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexPoly* vp, const typename PFP::VEC3& normalPoly, VPMS& ears, bool convex)
{
    typedef typename PFP::VEC3 VEC3;

    VertexPoly* vprev = vp->prev;
    VertexPoly* vp2 = vp->next;
    VertexPoly* vnext = vp2->next;
    const VEC3& Ta = position[vp->id];
    const VEC3& Tb = position[vp2->id];
    const VEC3& Tc = position[vprev->id];
    const VEC3& Td = position[vnext->id];

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
        for (VPMS::reverse_iterator it = ears.rbegin(); (!finished)&&(it != ears.rend())&&((*it)->value > 5.0f); ++it)
        {
            int id = (*it)->id;
            const VEC3& P = position[id];

            if ((dotpr1 < 5.0f) && (id !=vprev->id))
                if (inTriangle<VEC3>(P, normalPoly,Tb,Tc,Ta))
                    dotpr1 = 5.0f;// not an ears !

            if ((dotpr2 < 5.0f) && (id !=vnext->id) )
                if (inTriangle<VEC3>(P, normalPoly,Td,Ta,Tb))
                    dotpr2 = 5.0f;// not an ears !

            finished = ((dotpr1 >= 5.0f)&&(dotpr2 >= 5.0f));
        }
    }

    vp->value  = dotpr1;
    vp->length = (Tb-Tc).norm2();
    vp->ear = ears.insert(vp);
    vp2->value = dotpr2;
    vp->length = (Td-Ta).norm2();
    vp2->ear = ears.insert(vp2);
}

template<typename PFP>
float MapRender::computeEarAngle(const typename PFP::VEC3& P1, const typename PFP::VEC3& P2,  const typename PFP::VEC3& P3, const typename PFP::VEC3& normalPoly)
{
    typedef typename PFP::VEC3 VEC3;

    VEC3 v1 = P1-P2;
    VEC3 v2 = P3-P2;
    v1.normalize();
    v2.normalize();

    //	float dotpr = 1.0f - (v1*v2);
    float dotpr = acos(v1*v2) / (M_PI/2.0f);

    VEC3 vn = v1^v2;
    if (vn*normalPoly > 0.0f)
        dotpr = 10.0f - dotpr; 		// not an ears  (concave, store at the end for optimized use for intersections)

    return dotpr;
}

template<typename PFP>
bool MapRender::computeEarIntersection(const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexPoly* vp, const typename PFP::VEC3& normalPoly)
{
    typedef typename PFP::VEC3 VEC3;

    VertexPoly* endV = vp->prev;
    VertexPoly* curr = vp->next;
    const VEC3& Ta = position[vp->id];
    const VEC3& Tb = position[curr->id];
    const VEC3& Tc = position[endV->id];
    curr = curr->next;

    while (curr != endV)
    {
        if (inTriangle<VEC3>(position[curr->id], normalPoly,Tb,Tc,Ta))
        {
            vp->value = 5.0f;// not an ears !
            return false;
        }
        curr = curr->next;
    }

    return true;
}

template<typename PFP>
inline void MapRender::addEarTri(typename PFP::MAP& map, Face f, std::vector<GLuint>& tableIndices, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* pos)
{
    typedef typename PFP::VEC3 VEC3;

    bool(*fn_pt1)(VertexPoly*,VertexPoly*) = &(MapRender::cmpVP);
    VPMS ears(fn_pt1);

    const VertexAttribute<VEC3, typename PFP::MAP>& position = *pos ;

    // compute normal to polygon
    VEC3 normalPoly = Algo::Surface::Geometry::newellNormal<PFP>(map, f, position);

    // first pass create polygon in chained list with angle computation
    VertexPoly* vpp = NULL;
    VertexPoly* prem = NULL;
    unsigned int nbv = 0;
    unsigned int nbe = 0;
    Vertex a = Dart::create(f.index());
    Vertex b = map.phi1(a);
    Vertex c = map.phi1(b);
    do
    {
        VEC3 P1 = position[map.getEmbedding(a)];
        VEC3 P2 = position[map.getEmbedding(b)];
        VEC3 P3 = position[map.getEmbedding(c)];

        float val = computeEarAngle<PFP>(P1, P2, P3, normalPoly);
        VertexPoly* vp = new VertexPoly(map.getEmbedding(b), val, (P3-P1).norm2(), vpp);

        if (vp->value < 5.0f)
            nbe++;
        if (vpp == NULL)
            prem = vp;
        vpp = vp;
        a = b;
        b = c;
        c = map.phi1(c);
        nbv++;
    } while (Dart(a) != f);

    VertexPoly::close(prem, vpp);

    bool convex = nbe == nbv;
    if (convex)
    {
        // second pass with no test of intersections with polygons
        vpp = prem;
        for (unsigned int i = 0; i < nbv; ++i)
        {
            vpp->ear = ears.insert(vpp);
            vpp = vpp->next;
        }
    }
    else
    {
        // second pass test intersections with polygons
        vpp = prem;
        for (unsigned int i = 0; i < nbv; ++i)
        {
            if (vpp->value < 5.0f)
                computeEarIntersection<PFP>(position, vpp, normalPoly);
            vpp->ear = ears.insert(vpp);
            vpp = vpp->next;
        }
    }

    // NOW WE HAVE THE POLYGON AND EARS
    // LET'S REMOVE THEM
    while (nbv > 3)
    {
        // take best (and valid!) ear
        VPMS::iterator be_it = ears.begin(); // best ear
        VertexPoly* be = *be_it;

        tableIndices.push_back(be->id);
        tableIndices.push_back(be->next->id);
        tableIndices.push_back(be->prev->id);
        nbv--;

        if (nbv > 3)	// do not recompute if only one triangle left
        {
            //remove ears and two sided ears
            ears.erase(be_it);					// from map of ears
            ears.erase(be->next->ear);
            ears.erase(be->prev->ear);
            be = VertexPoly::erase(be); 	// and remove ear vertex from polygon
            recompute2Ears<PFP>(position, be, normalPoly, ears, convex);
            convex = (*(ears.rbegin()))->value < 5.0f;
        }
        else		// finish (no need to update ears)
        {
            // remove ear from polygon
            be = VertexPoly::erase(be);
            // last triangle
            tableIndices.push_back(be->id);
            tableIndices.push_back(be->next->id);
            tableIndices.push_back(be->prev->id);
            // release memory of last triangle in polygon
            delete be->next;
            delete be->prev;
            delete be;
        }
    }
}

template<typename PFP>
inline void MapRender::addTri(typename PFP::MAP& map, Face f, std::vector<GLuint>& tableIndices)
{
    Vertex a = Dart::create(f.index());
    Vertex b = map.phi1(a);
    Vertex c = map.phi1(b);

    // loop to cut a polygon in triangle on the fly (works only with convex faces)
    do
    {
        tableIndices.push_back(map.getEmbedding(a));
        tableIndices.push_back(map.getEmbedding(b));
        tableIndices.push_back(map.getEmbedding(c));
        b = c;
        c = map.phi1(b);
    } while (c != a);
}

template<typename PFP>
void MapRender::initTriangles(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, unsigned int thread)
{
    typedef typename PFP::MAP MAP;
    tableIndices.reserve(4 * map.getNbDarts() / 3);

    if(position == NULL)
    {
        foreach_cell<FACE>(map,
                           (
                               bl::bind(&addTri<PFP>, boost::ref(map), bl::_1, boost::ref(tableIndices))
                               ), AUTO, thread);
    }
    else
    {
        foreach_cell<FACE>(map,
                           (
                               bl::if_(bl::bind(&MAP::faceDegree, boost::ref(map), bl::_1) == 3)[
                               bl::bind(&addTri<PFP>, boost::ref(map), bl::_1, boost::ref(tableIndices))
                           ].else_[
                           bl::bind(&addEarTri<PFP>, boost::ref(map), bl::_1, boost::ref(tableIndices), boost::ref(position)) ]
                ), AUTO, thread);
    }
}

template<typename PFP>
void MapRender::initTrianglesOptimized(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, unsigned int thread)
{
#define LIST_SIZE 20
    DartMarker<typename PFP::MAP> m(map, thread);
    // reserve memory for triangles ( nb indices == nb darts )
    // and a little bit more
    // if lots of polygonal faces, realloc is done by vector
    tableIndices.reserve(4 * map.getNbDarts() / 3);

    for (Dart dd = map.begin(); dd != map.end(); map.next(dd))
    {
        if (!m.isMarked(dd))
        {
            std::list<Dart> bound;

            if (!map.template isBoundaryMarked<PFP::MAP::DIMENSION>(dd))
            {
                if(position == NULL)
                    addTri<PFP>(map, dd, tableIndices);
                else
                {
                    if(map.faceDegree(dd) == 3)
                        addTri<PFP>(map, dd, tableIndices);
                    else
                        addEarTri<PFP>(map, dd, tableIndices, position);
                }
            }
            m.template markOrbit<FACE>(dd);
            bound.push_back(dd);
            int nb = 1;
            do
            {
                Dart e = bound.back();
                Dart ee = e;
                do
                {
                    Dart f = ee;
                    do
                    {
                        if (!m.isMarked(f))
                        {
                            if ( !map.template isBoundaryMarked<PFP::MAP::DIMENSION>(f))
                            {
                                if(position == NULL)
                                    addTri<PFP>(map, f, tableIndices);
                                else
                                {
                                    if(map.faceDegree(f) == 3)
                                        addTri<PFP>(map, f, tableIndices);
                                    else
                                        addEarTri<PFP>(map, f, tableIndices, position);
                                }
                            }
                            m.template markOrbit<FACE>(f);
                            bound.push_back(map.phi1(f));
                            ++nb;
                            if (nb > LIST_SIZE)
                            {
                                bound.pop_front();
                                --nb;
                            }
                        }
                        f = map.phi1(map.phi2(f));
                    } while (f != ee);
                    ee = map.phi1(ee);
                } while (ee != e);

                bound.pop_back();
                --nb;
            } while (!bound.empty());
        }
    }
#undef LIST_SIZE
}

template<typename PFP>
void MapRender::initLines(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
    typedef typename PFP::MAP MAP;
    tableIndices.reserve(map.getNbDarts());

    //	TraversorE<typename PFP::MAP> trav(map, thread);
    foreach_cell<EDGE>(map,
                       (
                           bl::bind(&std::vector<GLuint>::push_back,boost::ref(tableIndices),  bl::bind(&MAP::template getEmbedding<VERTEX>, boost::ref(map), bl::_1)),
                           bl::bind(&std::vector<GLuint>::push_back,boost::ref(tableIndices),  bl::bind(&MAP::template getEmbedding<VERTEX>, boost::ref(map), bl::bind(&MAP::phi1, boost::ref(map), bl::_1)))
                           )
                       ,AUTO,thread);
}

template<typename PFP>
void MapRender::initBoundaries(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
    typedef typename PFP::MAP MAP;
    tableIndices.reserve(map.getNbDarts()); //TODO optimisation ?

    //	TraversorE<typename PFP::MAP> trav(map, thread);
    //	for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
    foreach_cell<EDGE>(map,
                       bl::if_(bl::bind(&MAP::isBoundaryEdge, boost::ref(map), bl::_1))
                       [(
                bl::bind(&std::vector<GLuint>::push_back,boost::ref(tableIndices),  bl::bind(&MAP::template getEmbedding<VERTEX>, boost::ref(map), bl::_1)),
                bl::bind(&std::vector<GLuint>::push_back,boost::ref(tableIndices),  bl::bind(&MAP::template getEmbedding<VERTEX>, boost::ref(map), bl::bind(&MAP::phi1, boost::ref(map), bl::_1)))
                )]
            ,AUTO,thread);
}

template<typename PFP>
void MapRender::initLinesOptimized(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
#define LIST_SIZE 20
    DartMarker<typename PFP::MAP> m(map, thread);

    // reserve memory for edges indices ( nb indices == nb darts)
    tableIndices.reserve(map.getNbDarts());

    for (Dart dd = map.begin(); dd != map.end(); map.next(dd))
    {
        if (!m.isMarked(dd))
        {
            std::list<Dart> bound;
            bound.push_back(dd);
            int nb = 1;
            do
            {
                Dart e = bound.back();
                Dart ee = e;
                do
                {
                    Dart f = map.phi2(ee);
                    if (!m.isMarked(ee))
                    {
                        tableIndices.push_back(map.template getEmbedding<VERTEX>(ee));
                        tableIndices.push_back(map.template getEmbedding<VERTEX>(map.phi1(ee)));
                        m.template markOrbit<EDGE>(f);

                        bound.push_back(f);
                        ++nb;
                        if (nb > LIST_SIZE)
                        {
                            bound.pop_front();
                            --nb;
                        }
                    }
                    ee = map.phi1(f);
                } while (ee != e);
                bound.pop_back();
                --nb;
            } while (!bound.empty());
        }
    }
#undef LIST_SIZE
}

template<typename PFP>
void MapRender::initPoints(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread)
{
    typedef typename PFP::MAP MAP;
    tableIndices.reserve(map.getNbDarts() / 5);

    //	TraversorV<typename PFP::MAP> trav(map, thread);
    foreach_cell<VERTEX>(map,
    (
        bl::bind(&std::vector<GLuint>::push_back, boost::ref(tableIndices), bl::bind(&MAP::getEmbedding, boost::ref(map), bl::_1))
    )
    ,FORCE_CELL_MARKING,thread); //
}

template<typename PFP>
void MapRender::initPrimitives(typename PFP::MAP& map, int prim, bool optimized, unsigned int thread)
{
    initPrimitives<PFP>(map, prim, NULL, optimized, thread) ;
}

template <typename PFP>
void MapRender::initPrimitives(typename PFP::MAP& map, int prim, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, bool optimized, unsigned int thread)
{
    std::vector<GLuint> tableIndices;

    switch(prim)
    {
    case POINTS:

        initPoints<PFP>(map, tableIndices, thread);
        break;
    case LINES:
        if(optimized)
            initLinesOptimized<PFP>(map, tableIndices, thread);
        else
            initLines<PFP>(map, tableIndices, thread) ;
        break;
    case TRIANGLES:
        if(optimized)
            initTrianglesOptimized<PFP>(map, tableIndices, position, thread);
        else
            initTriangles<PFP>(map, tableIndices, position, thread) ;
        break;
    case FLAT_TRIANGLES:
        break;
    case BOUNDARY:
        initBoundaries<PFP>(map, tableIndices, thread) ;
        break;
    default:
        CGoGNerr << "problem initializing VBO indices" << CGoGNendl;
        break;
    }

    m_nbIndices[prim] = tableIndices.size();
    m_indexBufferUpToDate[prim] = true;

    // setup du buffer d'indices
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffers[prim]);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, m_nbIndices[prim] * sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);
}

template <typename PFP>
void MapRender::addPrimitives(typename PFP::MAP& map, int prim, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, bool optimized, unsigned int thread)
{
    std::vector<GLuint> tableIndices;

    switch(prim)
    {
    case POINTS:

        initPoints<PFP>(map, tableIndices, thread);
        break;
    case LINES:
        if(optimized)
            initLinesOptimized<PFP>(map, tableIndices, thread);
        else
            initLines<PFP>(map, tableIndices, thread) ;
        break;
    case TRIANGLES:
        if(optimized)
            initTrianglesOptimized<PFP>(map, tableIndices, position, thread);
        else
            initTriangles<PFP>(map, tableIndices, position, thread) ;
        break;
    case FLAT_TRIANGLES:
        break;
    case BOUNDARY:
        initBoundaries<PFP>(map, tableIndices, thread) ;
        break;
    default:
        CGoGNerr << "problem initializing VBO indices" << CGoGNendl;
        break;
    }

    m_indexBufferUpToDate[prim] = true;

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffers[prim]);
    GLint sz=0;
    glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &sz);
    GLuint* oldIndices =  reinterpret_cast<GLuint*>(glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_READ_WRITE));

    // allocate new buffer
    GLuint newBuffer;
    glGenBuffers(1,&newBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, newBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sz + tableIndices.size() * sizeof(GLuint),NULL, GL_STREAM_DRAW);

    //copy old indices
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sz, oldIndices);
    //and new ones
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, sz, m_nbIndices[prim] * sizeof(GLuint), &(tableIndices[0]) );

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffers[prim]);
    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);

    glDeleteBuffers(1,&(m_indexBuffers[prim]));
    m_indexBuffers[prim] = newBuffer;

    m_nbIndices[prim] += tableIndices.size();
}

} // namespace GL2

} // namespace Render

} // namespace Algo

} // namespace CGoGN
