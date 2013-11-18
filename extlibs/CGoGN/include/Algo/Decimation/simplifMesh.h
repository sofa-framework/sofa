/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
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

#include <map>
#include "Container/fakeAttribute.h"
#include "Algo/Geometry/basic.h"
#include "Topology/generic/autoAttributeHandler.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

/**
 * Example of Edge_Critera class parameter
 * Will be used as edge embedding 
 * 
 * Necessary methods are:
 *  - constructor from dart 
 *  - getDart
 * - getKey -> float
 *  - removingAllowed
 *  - newPosition
 */
template <typename PFP>
class EdgeCrit
{
public:
	typedef typename PFP::MAP MAP;


protected:
	Dart m_dart;
	bool m_dirty;

public:
	/**
	 * Constructor (compute length on the fly)
	 * @param d a dart of the edge
	 */
	EdgeCrit(): m_dirty(false) {}

	virtual float computeKey(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& posi) const = 0;

	/**
	 * MUST BE IMPLEMENTED
	 * return the dart
	 */
	virtual Dart getDart() const  = 0;

	/**
	 * MUST BE IMPLEMENTED
	 * criteria test to allow removing
	 */
	virtual bool removingAllowed() const  = 0;

	/**
	 * MUST BE IMPLEMENTED
	 * compute new vertex position: here middle of edge
	 */
	virtual typename PFP::VEC3 newPosition(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& posi) const = 0;


	/**
	 * MUST BE IMPLEMENTED
	 * test if criteria is taged as dirty
	 */
	virtual bool isDirty() const = 0;

	/**
	 * MUST BE IMPLEMENTED
	 * tag the criteria as dirty (no more usable)
	 */
	virtual void tagDirty() = 0;

	/**
	 * Create
	 */
	virtual EdgeCrit* create(Dart d) const = 0;
};


/**
 * Example of Edge_Critera class parameter
 * Will be used as edge embedding
 *
 * Necessary methods are:
 *  - constructor from dart
 *  - getDart
 *  - getKey -> float
 *  - removingAllowed
 *  - newPosition
 */
template <typename PFP>
class EdgeCrit_LengthMiddle : public EdgeCrit<PFP>
{
public:
	typedef typename PFP::MAP MAP;

public:
	/**
	 * Constructor (compute length on the fly)
	 * @param d a dart of the edge
	 */
	EdgeCrit_LengthMiddle(): EdgeCrit<PFP>()
	{}

	EdgeCrit_LengthMiddle(Dart d) :	EdgeCrit<PFP>()
	{
		this->m_dart = d;
	}

	float computeKey(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& posi) const
	{
		typename PFP::VEC3 PQ = Algo::Geometry::vectorOutOfDart<PFP>(map, this->m_dart, posi);
		return PQ.norm2();
	}
	
	/**
	 * MUST BE IMPLEMENTED
	 * return the dart
	 */ 
	Dart getDart() const { return this->m_dart; }
	
	/**
	 * MUST BE IMPLEMENTED
	 * criteria test to allow removing
	 */ 
	bool removingAllowed() const { return true; }
	
	/**
	 * MUST BE IMPLEMENTED
	 * compute new vertex position: here middle of edge
	 */
	typename PFP::VEC3 newPosition(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& posi) const
	{
		const typename PFP::VEC3& p1 = posi[this->m_dart];
		const typename PFP::VEC3& p2 = posi[map.phi1(this->m_dart)];
		return typename PFP::REAL(0.5) * (p1 + p2);
	}
	
	/**
	 * MUST BE IMPLEMENTED
	 * test if criteria is taged as dirty
	 */
	bool isDirty() const
	{
		return this->m_dirty;
	}

	/**
	 * MUST BE IMPLEMENTED
	 * tag the criteria as dirty (no more usable)
	 */
	void tagDirty() 
	{
		this->m_dirty = true;
	}

	/**
	 * Create
	 */
	EdgeCrit<PFP>* create(Dart d) const
	{
		return new EdgeCrit_LengthMiddle(d);
	}
};

/**
 * Example of Edge_Critera class parameter
 * Will be used as edge embedding
 *
 * Necessary methods are:
 *  - constructor from dart
 *  - getDart
 *  - getKey -> float
 *  - removingAllowed
 *  - newPosition
 */
template <typename PFP>
class EdgeCrit_LengthFirst : public EdgeCrit<PFP>
{
public:
	typedef typename PFP::MAP MAP;
	
public:
	/**
	 * Constructor (compute length on the fly)
	 * @param d a dart of the edge
	 */
	EdgeCrit_LengthFirst(): EdgeCrit<PFP>() {}

	EdgeCrit_LengthFirst(Dart d) :
		EdgeCrit<PFP>() {this->m_dart = d;}

	float computeKey(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& posi) const
	{
		typename PFP::VEC3 PQ = Algo::Geometry::vectorOutOfDart<PFP>(map, this->m_dart, posi);
		return PQ.norm2();
	}

	/**
	 * MUST BE IMPLEMENTED
	 * return the dart
	 */
	Dart getDart() const { return this->m_dart; }

	/**
	 * MUST BE IMPLEMENTED
	 * criteria test to allow removing
	 */
	bool removingAllowed() const { return true; }

	/**
	 * MUST BE IMPLEMENTED
	 * compute new vertex position: here middle of edge
	 */
	typename PFP::VEC3 newPosition(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& posi) const
	{
		const typename PFP::VEC3& p1 = posi[this->m_dart];
		const typename PFP::VEC3& p2 = posi[map.phi1(this->m_dart)];
		return typename PFP::REAL(0.9)*p1 + typename PFP::REAL(0.1)*p2 ;
	}

	/**
	 * MUST BE IMPLEMENTED
	 * test if criteria is taged as dirty
	 */
	bool isDirty() const
	{
		return this->m_dirty;
	}

	/**
	 * MUST BE IMPLEMENTED
	 * tag the criteria as dirty (no more usable)
	 */
	void tagDirty()
	{
		this->m_dirty = true;
	}

	/**
	 * Create
	 */
	EdgeCrit<PFP>* create(Dart d) const
	{
		return new EdgeCrit_LengthFirst(d);
	}
};


template <typename PFP>
class SimplifTrian
{
public:
	typedef typename PFP::MAP MAP;

	typedef EdgeCrit<PFP> CRIT;

private:
	MAP& m_map;

	CRIT* m_crit;

	VertexAttribute<typename PFP::VEC3> m_positions;

	// map des critères (donc trié)
	std::multimap<float, CRIT*> m_edgeCrit;

	typedef typename std::multimap<float, CRIT*>::iterator CRIT_IT;

	typedef NoTypeNameAttribute<CRIT_IT> CRIT_IT_ATT;

	// attribut d'arête
	AutoAttributeHandler<CRIT_IT_ATT> m_edgeEmb;

	// attribut de sommet (valence)
	AutoAttributeHandler<unsigned int> m_valences;

	AttributeHandler<Mark> m_edgeMarkers;

	CellMarker m_protectMarker;

	unsigned int m_nbTriangles;

	unsigned int m_nbWanted;

	unsigned int m_passe;

public:
	SimplifTrian(MAP& the_map, unsigned int idPos, CRIT* cr);

	void changeCriteria(CRIT* cr);

	void simplifUntil(unsigned int nbWantedTriangles);

	~SimplifTrian();

	void updateCriterias(Dart d);

	bool edgeIsCollapsible(Dart d);

	Dart edgeCollapse(Dart d, typename PFP::VEC3 &);

	void computeVerticesValences(bool gc);

protected:
	CRIT* getCrit(Dart d)
	{
		return m_edgeEmb[d]->second;
	}

};

} //namespace Decimation

}

} //namespace Algo

} //namespace CGoGN

#include "simplifMesh.hpp"
