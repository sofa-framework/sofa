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

#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor2.h"
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor3.h"
#include "Algo/Parallel/parallel_foreach.h"


namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE volumeCentroid(typename PFP::MAP& map, Dart d, const V_ATT& attributs, unsigned int thread)
{
	typename V_ATT::DATA_TYPE center(0.0);
	unsigned int count = 0 ;
	Traversor3WV<typename PFP::MAP> tra(map,d,false,thread);
	for (Dart d = tra.begin(); d != tra.end(); d = tra.next())
	{
		center += attributs[d];
		++count;
	}

	center /= double(count) ;
	return center ;
}


template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE volumeCentroidELW(typename PFP::MAP& map, Dart d, const V_ATT& attributs, unsigned int thread)
{
	typedef typename V_ATT::DATA_TYPE EMB;
	EMB center(0.0);

	double count=0.0;
	Traversor3WE<typename PFP::MAP> t(map, d,false,thread) ;
	for(Dart it = t.begin(); it != t.end();it = t.next())
	{
		EMB e1 = attributs[it];
		EMB e2 = attributs[map.phi1(it)];
		double l = (e2-e1).norm();
		center += (e1+e2)*l;
		count += 2.0*l ;
	}
	center /= double(count);	
	return center ;
}


template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE faceCentroid(typename PFP::MAP& map, Dart d, const V_ATT& attributs)
{
	typename V_ATT::DATA_TYPE center(0.0);
	unsigned int count = 0 ;
	Traversor2FV<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		center += attributs[it];
		++count ;
	}
	center /= double(count);
	return center ;
}



template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE faceCentroidELW(typename PFP::MAP& map, Dart d, const V_ATT& attributs)
{
	typedef typename V_ATT::DATA_TYPE EMB;

	EMB center(0.0);
	double count=0.0;
	Traversor2FE<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		EMB e1 = attributs[it];
		EMB e2 = attributs[map.phi1(it)];
		double l = (e2-e1).norm();
		center += (e1+e2)*l;
		count += 2.0*l ;
	}
	center /= double(count);
	return center ;
}


template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexNeighborhoodCentroid(typename PFP::MAP& map, Dart d, const V_ATT& attributs)
{
	typename V_ATT::DATA_TYPE center(0.0);

	unsigned int count = 0 ;
	Traversor2VVaE<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		center += attributs[it];
		++count ;
	}
	center /= count ;
	return center ;
}





template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid, unsigned int thread)
{
	TraversorF<typename PFP::MAP> t(map,thread) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		face_centroid[d] = faceCentroid<PFP,V_ATT>(map, d, position) ;
}



template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidELWFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid, unsigned int thread)
{
	TraversorF<typename PFP::MAP> t(map,thread) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		face_centroid[d] = faceCentroidELW<PFP,V_ATT>(map, d, position) ;
}



template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid, unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map, thread) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vertex_centroid[d] = vertexNeighborhoodCentroid<PFP,V_ATT>(map, d, position) ;
}


namespace Parallel
{

template <typename PFP, typename V_ATT, typename F_ATT>
class FunctorComputeCentroidFaces: public FunctorMapThreaded<typename PFP::MAP >
{
	 const V_ATT& m_position;
	 F_ATT& m_fcentroid;
public:
	 FunctorComputeCentroidFaces<PFP,V_ATT,F_ATT>( typename PFP::MAP& map, const V_ATT& position, F_ATT& fcentroid):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_fcentroid(fcentroid)
	 { }

	void run(Dart d, unsigned int /*threadID*/)
	{
		m_fcentroid[d] = faceCentroid<PFP>(this->m_map, d, m_position) ;
	}
};

template <typename PFP, typename V_ATT, typename F_ATT>
class FunctorComputeCentroidELWFaces: public FunctorMapThreaded<typename PFP::MAP >
{
	const V_ATT& m_position;
	F_ATT& m_fcentroid;
public:
	 FunctorComputeCentroidELWFaces<PFP,V_ATT,F_ATT>( typename PFP::MAP& map, const V_ATT& position, F_ATT& fcentroid):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_fcentroid(fcentroid)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		m_fcentroid[d] = faceCentroidELW<PFP>(this->m_map, d, m_position) ;
	}
};

template <typename PFP, typename V_ATT>
class FunctorComputeNeighborhoodCentroidVertices: public FunctorMapThreaded<typename PFP::MAP >
{
	 const V_ATT& m_position;
	 V_ATT& m_vcentroid;
public:
	 FunctorComputeNeighborhoodCentroidVertices<PFP,V_ATT>( typename PFP::MAP& map, const V_ATT& position, V_ATT& vcentroid):
		 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_vcentroid(vcentroid)
	 { }

	void run(Dart d, unsigned int /*threadID*/)
	{
		m_vcentroid[d] = vertexNeighborhoodCentroid<PFP>(this->m_map, d, m_position) ;
	}
};


template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid,
		unsigned int nbth, unsigned int current_thread)
{
	FunctorComputeCentroidFaces<PFP,V_ATT,F_ATT> funct(map,position,face_centroid);
	Algo::Parallel::foreach_cell<typename PFP::MAP,FACE>(map, funct, nbth, false, current_thread);
}

template <typename PFP, typename V_ATT, typename F_ATT>
void computeCentroidELWFaces(typename PFP::MAP& map, const V_ATT& position, F_ATT& face_centroid,
		unsigned int nbth, unsigned int current_thread)
{
	FunctorComputeCentroidELWFaces<PFP,V_ATT,F_ATT> funct(map,position,face_centroid);
	Algo::Parallel::foreach_cell<typename PFP::MAP,FACE>(map, funct, nbth, false, current_thread);
}



template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map,
		const V_ATT& position, V_ATT& vertex_centroid,
		unsigned int nbth, unsigned int current_thread)
{
	FunctorComputeNeighborhoodCentroidVertices<PFP,V_ATT> funct(map,position,vertex_centroid);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VERTEX>(map, funct, nbth, false);
}

}
} // namespace Geometry
} // namespace Surface

namespace Volume
{
namespace Geometry
{

template <typename PFP, typename V_ATT>
typename V_ATT::DATA_TYPE vertexNeighborhoodCentroid(typename PFP::MAP& map, Dart d, const V_ATT& attributs)
{
	typename V_ATT::DATA_TYPE  center(0.0);
	unsigned int count = 0 ;
	Traversor3VVaE<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		center += attributs[it];
		++count ;
	}
	center /= count ;
	return center ;
}

template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid, unsigned int thread)
{
	TraversorW<typename PFP::MAP> t(map, thread) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vol_centroid[d] = Surface::Geometry::volumeCentroid<PFP,V_ATT>(map, d, position,thread) ;
}



template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidELWVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid, unsigned int thread)
{
	TraversorW<typename PFP::MAP> t(map,thread) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vol_centroid[d] = Surface::Geometry::volumeCentroidELW<PFP,V_ATT>(map, d, position,thread) ;
}



template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid, unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map, thread) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vertex_centroid[d] = Volume::Geometry::vertexNeighborhoodCentroid<PFP,V_ATT>(map, d, position) ;
}



namespace Parallel
{
template <typename PFP, typename V_ATT, typename W_ATT>
class FunctorComputeCentroidVolumes: public FunctorMapThreaded<typename PFP::MAP >
{
	 const V_ATT& m_position;
	 W_ATT& m_vol_centroid;
public:
	 FunctorComputeCentroidVolumes<PFP,V_ATT,W_ATT>( typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_vol_centroid(vol_centroid)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		m_vol_centroid[d] = Surface::Geometry::volumeCentroid<PFP,V_ATT,W_ATT>(this->m_map, d, m_position,threadID) ;
	}
};


template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidVolumes(typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid,	unsigned int nbth)
{
	FunctorComputeCentroidVolumes<PFP,V_ATT,W_ATT> funct(map,position,vol_centroid);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VOLUME>(map, funct, nbth, true);
}






template <typename PFP, typename V_ATT, typename W_ATT>
class FunctorComputeCentroidELWVolumes: public FunctorMapThreaded<typename PFP::MAP >
{
	 const V_ATT& m_position;
	 W_ATT& m_vol_centroid;
//	 VolumeAttribute<typename PFP::VEC3>& m_vol_centroid;
public:
	 FunctorComputeCentroidELWVolumes<PFP,V_ATT,W_ATT>( typename PFP::MAP& map, const V_ATT& position, W_ATT& vol_centroid):
		 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_vol_centroid(vol_centroid)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		m_vol_centroid[d] = Surface::Geometry::volumeCentroidELW<PFP,V_ATT>(this->m_map, d, m_position, threadID) ;
	}
};



template <typename PFP, typename V_ATT, typename W_ATT>
void computeCentroidELWVolumes(typename PFP::MAP& map,
		const V_ATT& position, W_ATT& vol_centroid,
		unsigned int nbth)
{
	FunctorComputeCentroidELWVolumes<PFP,V_ATT,W_ATT> funct(map,position,vol_centroid);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VOLUME>(map, funct, nbth, true);
}




template <typename PFP, typename V_ATT>
class FunctorComputeNeighborhoodCentroidVertices: public FunctorMapThreaded<typename PFP::MAP >
{
	 const V_ATT& m_position;
	 V_ATT& m_vcentroid;
public:
	 FunctorComputeNeighborhoodCentroidVertices<PFP,V_ATT>( typename PFP::MAP& map, const V_ATT& position, V_ATT& vcentroid):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_vcentroid(vcentroid)
	 { }

	void run(Dart d, unsigned int /*threadID*/)
	{
		m_vcentroid[d] = vertexNeighborhoodCentroid<PFP,V_ATT>(this->m_map, d, m_position) ;
	}
};


template <typename PFP, typename V_ATT>
void computeNeighborhoodCentroidVertices(typename PFP::MAP& map, const V_ATT& position, V_ATT& vertex_centroid,
		unsigned int nbth)
{
	FunctorComputeNeighborhoodCentroidVertices<PFP,V_ATT> funct(map,position,vertex_centroid);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VERTEX>(map, funct, nbth, false);
}


} // namespace Parallel
} // namespace Geometry

} // namespace Volume



} // namespace Algo

} // namespace CGoGN
