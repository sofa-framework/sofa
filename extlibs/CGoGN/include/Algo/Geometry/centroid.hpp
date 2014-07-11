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

#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor/traversor2.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor/traversor3.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE volumeCentroid(typename PFP::MAP& map, Vol d, const V_ATT& attributs, unsigned int thread)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    EMB center;
    center.fill(0);
    unsigned int count = 0 ;

    //	foreach_incident3<VERTEX>(map,d, [&] (Vertex v)
    //	{
    //		center += attributs[v];
    //		++count;
    //	}
    //	,false,thread);

    foreach_incident3<VERTEX, VOLUME>(map,d, (bl::bind<void>(static_cast<void (EMB::*)(const EMB&)>(&EMB::operator+=), boost::ref(center), bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs),   bl::_1)), ++bl::var(count))
                                      ,false,thread);

    center /= double(count) ;
    return center ;
}

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE volumeCentroidELW(typename PFP::MAP& map, Vol d, const V_ATT& attributs, unsigned int thread)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    typedef typename PFP::MAP Map;
    EMB center;
    center.fill(0);

    double count=0.0;
    using bl::var;
    //    //    foreach_incident3<EDGE>(map,d, [&] (Edge it)
    //    //    {
    //    //        EMB e1 = attributs[it.dart];
    //    //        EMB e2 = attributs[map.phi1(it)];
    //    //        double l = (e2-e1).norm();
    //    //        center += (e1+e2)*l;
    //    //        count += 2.0*l ;
    //    //    },false,thread);
    EMB e1, e2;
    double l;
    foreach_incident3<EDGE, VOLUME>(map,d,
                                    (var(e1) = bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs), bl::bind<Vertex>(&Vertex::template convertCell<EDGE>, bl::_1)),
                                     var(e2) = bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs), bl::bind<Dart>(&Map::phi1, boost::cref(map), bl::_1)),
                                     var(l) = bl::bind<typename PFP::REAL>(&EMB::norm, bl::bind<EMB>(static_cast<EMB (EMB::*)(const EMB&) const>(&EMB::operator-),boost::cref(e2),boost::cref(e1))),
                                     bl::bind(static_cast<void (EMB::*)(const EMB&)>(&EMB::operator+=), boost::ref(center), bl::bind<EMB>(static_cast<EMB (EMB::*)(double) const>(&EMB::operator*), bl::bind<EMB>(static_cast<EMB (EMB::*)(const EMB&) const>(&EMB::operator+),boost::cref(e1),boost::cref(e2)), boost::cref(l)) ),
                                     var(count) += 2.0*var(l)), false, thread);

    center /= double(count);
    return center ;
}

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE faceCentroid(typename PFP::MAP& map, Face f, const V_ATT& attributs)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    EMB center;
    center.fill(0);
    unsigned int count = 0 ;
    using bl::var;
    //    foreach_incident2<PFP::MAP, VERTEX>(map, f, [&](Vertex it)
    //    {
    //        center += attributs[it];
    //        ++count ;
    //    });
    foreach_incident2<VERTEX, FACE>(map, f, (bl::bind<void>(static_cast<void (EMB::*)(const EMB&)>(&EMB::operator+=), boost::ref(center), bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs),bl::_1)),
                                             var(count)++));

    center /= double(count);
    return center ;
}

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE faceCentroidELW(typename PFP::MAP& map, Face f, const V_ATT& attributs)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    typedef typename PFP::MAP Map;
    using bl::var;
    EMB center;
    center.fill(0);
    double count=0.0;

    //    foreach_incident2<EDGE>(map, f, [&](Edge it)
    //    {
    //        EMB e1 = attributs[it.dart];
    //        EMB e2 = attributs[map.phi1(it)];
    //        double l = (e2-e1).norm();
    //        center += (e1+e2)*l;
    //        count += 2.0*l ;
    //    });

    EMB e1, e2;
    double l;
    foreach_incident2<EDGE, FACE>(map,f,
                                  (var(e1) = bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs), bl::bind<Vertex>(&Vertex::template convertCell<EDGE>, bl::_1)),
                                   var(e2) = bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs), bl::bind<Dart>(&Map::phi1, boost::cref(map), bl::_1)),
                                   var(l) = bl::bind<typename PFP::REAL>(&EMB::norm, bl::bind<EMB>(static_cast<EMB (EMB::*)(const EMB&) const>(&EMB::operator-),boost::cref(e2),boost::cref(e1))),
                                   bl::bind(static_cast<void (EMB::*)(const EMB&)>(&EMB::operator+=), boost::ref(center), bl::bind<EMB>(static_cast<EMB (EMB::*)(double) const>(&EMB::operator*), bl::bind<EMB>(static_cast<EMB (EMB::*)(const EMB&) const>(&EMB::operator+),boost::cref(e1),boost::cref(e2)), boost::cref(l)) ),
                                   var(count) += 2.0*var(l)));
    center /= double(count);
    return center ;
}

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexNeighborhoodCentroid(typename PFP::MAP& map, Vertex v, const V_ATT& attributs)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    EMB center;
    center.fill(0);
    using bl::var;
    unsigned int count = 0 ;
    //	Traversor2VVaE<typename PFP::MAP> t(map, d) ;
    //    foreach_adjacent2<EDGE>(map, v, [&](Vertex it)
    //    {
    //        center += attributs[it];
    //        ++count ;
    //    });
    foreach_adjacent2<EDGE, VERTEX>(map, v, (bl::bind<void>(static_cast<void (EMB::*)(const EMB&)>(&EMB::operator+=), boost::ref(center), bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs),  bl::_1)),
                                             var(count)++));
    center /= count ;
    return center ;
}

template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid, unsigned int thread)
{
    typedef typename F_ATT::DATA_TYPE EMB;
    using bl::var;
    if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread==0))
    {
        Parallel::computeCentroidFaces<PFP,V_ATT,F_ATT>(map,position,face_centroid);
        return;
    }

    //    foreach_cell<FACE>(map, [&] (Face f)
    //    {
    //        face_centroid[f] = faceCentroid<PFP,V_ATT>(map, f, position) ;
    //    }
    //    ,AUTO,thread);
    foreach_cell<FACE>(map,
                       (
                           bl::bind<EMB& >(static_cast<EMB& (F_ATT::*)(Face)>(&F_ATT::operator[]),boost::ref(face_centroid), bl::_1)  = bl::bind<typename V_ATT::DATA_TYPE>(&faceCentroid<PFP,V_ATT>, boost::ref(map), bl::_1, boost::ref(position))
            )
                       ,AUTO,thread);
}

template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidELWFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid, unsigned int thread)
{
    typedef typename F_ATT::DATA_TYPE EMB;
    if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread==0))
    {
        Parallel::computeCentroidELWFaces<PFP,V_ATT,F_ATT>(map,position,face_centroid);
        return;
    }

    //    foreach_cell<FACE>(map, [&] (Face f)
    //    {
    //        face_centroid[f] = faceCentroidELW<PFP,V_ATT>(map, f, position) ;
    //    }
    //    ,AUTO,thread);
    foreach_cell<FACE>(map,
                       (
                           bl::bind<EMB& >(static_cast<EMB& (F_ATT::*)(Face)>(&F_ATT::operator[]),boost::ref(face_centroid), bl::_1) = bl::bind<typename V_ATT::DATA_TYPE>(&faceCentroidELW<PFP,V_ATT>, boost::ref(map), bl::_1, boost::ref(position))
            )
                       ,AUTO,thread);
}

template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid, unsigned int thread)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread == 0))
    {
        Parallel::computeNeighborhoodCentroidVertices<PFP,V_ATT>(map,position,vertex_centroid);
        return;
    }

    //    foreach_cell<VERTEX>(map, [&] (Vertex v)
    //    {
    //        vertex_centroid[v] = vertexNeighborhoodCentroid<PFP,V_ATT>(map, v, position) ;
    //    }, AUTO, thread);
    foreach_cell<VERTEX>(map,
                         (
                             bl::bind<EMB& >(static_cast<EMB& (V_ATT::*)(Vertex)>(&V_ATT::operator[]),boost::ref(vertex_centroid), bl::_1) = bl::bind<EMB>(&vertexNeighborhoodCentroid<PFP,V_ATT>, boost::ref(map), bl::_1, boost::cref(position))
            )
                         ,AUTO,thread);
}


namespace Parallel
{

template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid)
{
    //    CGoGN::Parallel::foreach_cell<FACE>(map,[&](Face f, unsigned int /*thr*/)
    //    {
    //        face_centroid[f] = faceCentroid<PFP>(map, f, position) ;
    //    });
    CGoGN::Parallel::foreach_cell<FACE>(map,
                                        (
                                            bl::bind<typename F_ATT::DATA_TYPE&>(static_cast<typename F_ATT::DATA_TYPE& (F_ATT::*)(Face)>(&F_ATT::operator[]),boost::ref(face_centroid), bl::_1) = bl::bind<typename V_ATT::DATA_TYPE>(&faceCentroid<PFP>, boost::ref(map), bl::_1, boost::ref(position))
            ));

}

template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidELWFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid)
{
    //    CGoGN::Parallel::foreach_cell<FACE>(map,[&](Face f, unsigned int /*thr*/)
    //    {
    //        face_centroid[f] = faceCentroidELW<PFP>(map, f, position) ;
    //    });
    CGoGN::Parallel::foreach_cell<FACE>(map,
                                        (
                                            bl::bind<typename F_ATT::DATA_TYPE&>(static_cast<typename F_ATT::DATA_TYPE& (F_ATT::*)(Face)>(&F_ATT::operator[]),boost::ref(face_centroid), bl::_1) = bl::bind<typename V_ATT::DATA_TYPE>(&faceCentroidELW<PFP>, boost::ref(map), bl::_1, boost::ref(position))
            ));
}

template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map,
                                         const V_ATT& position, V_ATT& vertex_centroid)
{
    //    CGoGN::Parallel::foreach_cell<VERTEX>(map,[&](Vertex v, unsigned int /*thr*/)
    //    {
    //        vertex_centroid[v] = vertexNeighborhoodCentroid<PFP>(map, v, position) ;
    //    }, FORCE_CELL_MARKING);
    CGoGN::Parallel::foreach_cell<VERTEX>(map,
                                          (
                                              bl::bind<typename V_ATT::DATA_TYPE&>(static_cast<typename V_ATT::DATA_TYPE& (V_ATT::*)(Vertex)>(&V_ATT::operator[]),boost::ref(vertex_centroid), bl::_1) = bl::bind<typename V_ATT::DATA_TYPE>(&vertexNeighborhoodCentroid<PFP>, boost::ref(map), bl::_1, boost::ref(position))
            ), FORCE_CELL_MARKING);
}

} // namespace Parallel



} // namespace Geometry

} // namespace Surface

namespace Volume
{

namespace Geometry
{

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexNeighborhoodCentroid(typename PFP::MAP& map, Vertex v, const V_ATT& attributs, unsigned int thread)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    EMB center;
    center.fill(0);
    using bl::var;
    unsigned int count = 0 ;
    //    foreach_adjacent3<EDGE>(map, v, [&] (Vertex it)
    //    {
    //        center += attributs[it];
    //        ++count ;
    //    }, false, thread);
    foreach_adjacent3<EDGE>(map, v,
                            (
                                var(center) += bl::bind<const EMB&>(static_cast<const EMB& (V_ATT::*)(Vertex) const>(&V_ATT::operator[]),boost::cref(attributs), bl::_1),
                                var(count)++
                                ), false, thread);
    center /= count ;
    return center ;
}

template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid, unsigned int thread)
{
    using bl::var;
    if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread == 0))
    {
        Parallel::computeCentroidVolumes<PFP,V_ATT,W_ATT>(map,position,vol_centroid);
        return;
    }

    //    foreach_cell<VOLUME>(map, [&] (Vol v)
    //    {
    //        vol_centroid[v] = Surface::Geometry::volumeCentroid<PFP,V_ATT>(map, v, position,thread) ;
    //    }, AUTO, thread);
    foreach_cell<VOLUME>(map,
                         (
                             bl::bind<typename W_ATT::DATA_TYPE&>(static_cast<typename W_ATT::DATA_TYPE& (W_ATT::*)(Vol)>(&W_ATT::operator[]),boost::ref(vol_centroid), bl::_1) = bl::bind<typename V_ATT::DATA_TYPE>(&Surface::Geometry::volumeCentroid<PFP,V_ATT>, boost::ref(map), bl::_1, boost::cref(position),thread)
            ), AUTO, thread);
}

template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidELWVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid, unsigned int thread)
{
    typedef typename V_ATT::DATA_TYPE EMBV;
    typedef typename W_ATT::DATA_TYPE EMBW;
    if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread == 0))
    {
        Parallel::computeCentroidELWVolumes<PFP,V_ATT,W_ATT>(map,position,vol_centroid);
        return;
    }

    //    foreach_cell<VOLUME>(map, [&] (Vol v)
    //    {
    //        vol_centroid[v] = Surface::Geometry::volumeCentroidELW<PFP,V_ATT>(map, v, position,thread) ;
    //    }, AUTO, thread);
    foreach_cell<VOLUME>(map,
                         (
                             bl::bind<EMBW&>(static_cast<EMBW& (EMBW::*)(const EMBW&)>(&EMBW::operator=), bl::bind<EMBW&>(static_cast<EMBW& (W_ATT::*)(Vol)>(&W_ATT::operator[]),boost::ref(vol_centroid), bl::_1), bl::bind<EMBV>(&Surface::Geometry::volumeCentroidELW<PFP,V_ATT>, boost::ref(map), bl::_1, boost::cref(position),bl::var(thread)))
                             ), AUTO, thread);
}

template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid, unsigned int thread)
{
    typedef typename V_ATT::DATA_TYPE EMB;
    if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread == 0))
    {
        Parallel::computeNeighborhoodCentroidVertices<PFP,V_ATT>(map,position,vertex_centroid);
        return;
    }

    //    foreach_cell<VERTEX>(map, [&] (Vertex v)
    //    {
    //        vertex_centroid[v] = Volume::Geometry::vertexNeighborhoodCentroid<PFP,V_ATT>(map, v, position) ;
    //    }, AUTO, thread);
    foreach_cell<VOLUME>(map,
                         (
                             bl::bind<EMB&>(static_cast<EMB& (V_ATT::*)(Vertex)>(&V_ATT::operator[]),boost::ref(vertex_centroid), bl::_1) = bl::bind<EMB&>(&Surface::Geometry::vertexNeighborhoodCentroid<PFP,V_ATT>, boost::ref(map), bl::_1, boost::cref(position))
            ), AUTO, thread);
}


namespace Parallel
{

template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid)
{
    //    CGoGN::Parallel::foreach_cell<VOLUME>(map, [&] (Vol v, unsigned int thr)
    //    {
    //        vol_centroid[v] = Surface::Geometry::volumeCentroid<PFP,V_ATT>(map, v, position, thr) ;
    //    });
    CGoGN::Parallel::foreach_cell<VOLUME>(map,
                                          (
                                              bl::bind<typename W_ATT::DATA_TYPE&>(static_cast<typename W_ATT::DATA_TYPE& (W_ATT::*)(Vol)>(&W_ATT::operator[]),boost::ref(vol_centroid), bl::_1) = bl::bind<typename V_ATT::DATA_TYPE>(&Surface::Geometry::volumeCentroid<PFP,V_ATT>,boost::ref(map), bl::_1, boost::cref(position), bl::_2)
            )
                                          );
}

template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidELWVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid)
{
    typedef typename V_ATT::DATA_TYPE EMBV;
    typedef typename W_ATT::DATA_TYPE EMBW;
    //    CGoGN::Parallel::foreach_cell<VOLUME>(map, [&] (Vol v, unsigned int thr)
    //    {
    //        vol_centroid[v] = Surface::Geometry::volumeCentroidELW<PFP,V_ATT>(map, v, position, thr) ;
    //    });

    CGoGN::Parallel::foreach_cell<VOLUME, typename PFP::MAP>(map,
                                                             (
                                                                 bl::bind<EMBW&>(static_cast<EMBW& (EMBW::*)(const EMBW&)>(&EMBW::operator=), bl::bind<EMBW&>(static_cast<EMBW& (W_ATT::*)(Vol)>(&W_ATT::operator[]),boost::ref(vol_centroid),bl::_1),  bl::bind<EMBV>(&Surface::Geometry::volumeCentroidELW<PFP,V_ATT>,boost::ref(map), bl::_1, boost::cref(position), bl::_2))
                                                                 )
                                                             );
}

template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid)
{
    //    CGoGN::Parallel::foreach_cell<VERTEX>(map, [&] (Vertex v, unsigned int thr)
    //    {
    //        vertex_centroid[v] = Volume::Geometry::vertexNeighborhoodCentroid<PFP,V_ATT>(map, v, position,thr) ;
    //    }, FORCE_CELL_MARKING);
    CGoGN::Parallel::foreach_cell<VERTEX>(map,
                                          (
                                              bl::bind<typename V_ATT::DATA_TYPE&>(static_cast<typename V_ATT::DATA_TYPE& (V_ATT::*)(Vertex)>(&V_ATT::operator[]),boost::ref(vertex_centroid), bl::_1) = bl::bind<typename V_ATT::DATA_TYPE>(&Volume::Geometry::vertexNeighborhoodCentroid<PFP,V_ATT>,boost::ref(map), bl::_1, boost::cref(position), bl::_2)
            )
                                          );
}

} // namespace Parallel


} // namespace Geometry

} // namespace Volume

} // namespace Algo

} // namespace CGoGN
