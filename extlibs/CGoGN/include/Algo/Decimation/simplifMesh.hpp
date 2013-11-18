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

#include <vector>
#include <set>
#include <algorithm>

namespace CGoGN
{

namespace Algo
{


namespace Surface
{

namespace Decimation
{

template <typename PFP>
SimplifTrian<PFP>::SimplifTrian(MAP& the_map, unsigned int idPos, CRIT* cr):
	m_map(the_map),
	m_crit(cr),
	m_positions(the_map, idPos),
	m_edgeEmb(the_map, EDGE),
	m_valences(the_map, VERTEX),
	m_edgeMarkers(the_map, EDGE << 24),
	m_protectMarker(the_map, EDGE),
	m_passe(0)
{
	computeVerticesValences(false);

	// mesh of triangle, we can compute the number function of number of dart
	m_nbTriangles = the_map.getNbDarts() / the_map.getDartsPerTriangle();

	// local marker to ensure that edges only once in structure
	DartMarker m(m_map);

	for (Dart d = m_map.begin(); d != m_map.end(); m_map.next(d))
	{
		if (!m.isMarked(d))
		{
			// creation of a criteria for the edge
			CRIT* cr = m_crit->create(d);

			// store it in the map
			float key = cr->computeKey(m_map, m_positions);
			CRIT_IT it = m_edgeCrit.insert(std::make_pair(key, cr));
			m_edgeEmb[d] = it;

			// mark cell for traversal
			m.markOrbit<EDGE>(d);
		}
	}
}

template <typename PFP>
SimplifTrian<PFP>::~SimplifTrian()
{}

template <typename PFP>
void SimplifTrian<PFP>::changeCriteria(CRIT* cr)
{
	m_crit = cr;

	// local marker to ensure that edges only once in structure
	DartMarker m(m_map);

	// vide la multimap de critere
	m_edgeCrit.clear();

	// et la remplit;
	for (Dart d = m_map.begin(); d != m_map.end(); m_map.next(d))
	{
		if (!m.isMarked(d))
		{
			// creation of a criteria for the edge
			CRIT* cr = m_crit->create(d);

			// store it in the map
			float key = cr->computeKey(m_map, m_positions);
			CRIT_IT it = m_edgeCrit.insert(std::make_pair(key,cr));
			m_edgeEmb[d] = it;

			// mark cell for traversal
			m.markOrbit<EDGE>(d);
		}
	}
}

template <typename PFP>
void SimplifTrian<PFP>::simplifUntil(unsigned int nbWantedTriangles)
{
	m_nbWanted = nbWantedTriangles;

	bool notFinished = true;
	while (notFinished)
	{
		// TODO optimiser la calcul de diff
		int diff = (m_nbTriangles - m_nbWanted) / 8;
		if (diff < 128)
			diff = 128;

		int collapsed = 0;
		CRIT_IT it = m_edgeCrit.begin();
		// traverse the criteria map until diff collapse & enough triangles
		while ( (it != m_edgeCrit.end()) /*&& (collapsed < diff)*/ && (m_nbWanted < m_nbTriangles) )
		{
			// get the criteria
			CRIT* cr = it->second;
//			 if criteria invalid then remove it and step to next
			if (cr->isDirty())
			{
				CRIT_IT jt = it++;
				delete jt->second;
				m_edgeCrit.erase(jt);
			}
			else
			{
				Dart d = cr->getDart();
				if (cr->removingAllowed())
				{
					if (edgeIsCollapsible(d))
					{
						// compute new position
						typename PFP::VEC3 np = cr->newPosition(m_map, m_positions);
						// and collapse edge
						Dart dd = edgeCollapse(d, np);
						// update criterias
						updateCriterias(dd);
						m_nbTriangles -= 2;
						++collapsed;
					}
				}
				++it;
			}
		}
		// test finish condition
		if ((collapsed == 0) || (m_nbWanted >= m_nbTriangles))
			notFinished = false;

		m_protectMarker.unmarkAll();
	}
}

template <typename PFP>
void SimplifTrian<PFP>::updateCriterias(Dart d)
{
	// turn around vertex of d
	Dart dd = d;
	do  // for each incident edges
	{
		// create criteria
		CRIT* cr = m_crit->create(d);

		float key = cr->computeKey(m_map, m_positions);
		CRIT_IT it = m_edgeCrit.insert(std::make_pair(key,cr));
		// store iterator on edge
		unsigned int em = m_map.getEmbedding(d, EDGE);
		m_map.setOrbitEmbedding<EDGE>(d, em);
		m_edgeEmb[em] = it;

		m_protectMarker.mark(em) ;

		// next edge
		d = m_map.phi2_1(d);
	} while (d!=dd);
}

template <typename PFP>
bool SimplifTrian<PFP>::edgeIsCollapsible(Dart d)
{
	// Check conflict distance condition
	if (m_protectMarker.isMarked(d))
		return false;

	return edgeCanCollapse<PFP>(m_map, d, m_valences);
}

template <typename PFP>
Dart SimplifTrian<PFP>::edgeCollapse(Dart d, typename PFP::VEC3& newPos)
{
	// store some darts
	Dart dd = m_map.phi2(d);
	Dart d1 = m_map.phi2(m_map.phi1(d));
	Dart d2 = m_map.phi2(m_map.phi_1(d));
	Dart dd1 = m_map.phi2(m_map.phi1(dd));
	Dart dd2 = m_map.phi2(m_map.phi_1(dd));

	// tag as dirty the critera that are associate to edges to be removed
	// and modified BUT NOT D
	Dart xd = d;
	do
	{
		CRIT* cr = getCrit(xd);
		cr->tagDirty();
		xd = m_map.phi2_1(xd);
	} while (xd != d);

	xd = m_map.phi2_1(dd); // phi2_1 pour ne pas repasser sur l'arete d/dd
	do
	{
		CRIT* cr = getCrit(xd);
		cr->tagDirty();
		xd = m_map.phi2_1(xd);
	} while (xd != dd);

	// store old valences
	int v_d = m_valences[d];
	int v_dd = m_valences[dd];

	// collapse the edge
	m_map.deleteFace(d);
	m_map.deleteFace(dd);
	m_map.sewFaces(d1, d2);
	m_map.sewFaces(dd1, dd2);

	// embed new vertex
	unsigned int emb = m_map.getEmbedding(d2, VERTEX);
	m_map.setOrbitEmbedding<VERTEX>(d2, emb);

	m_positions[d2] = newPos;
	m_valences[d2] = v_d + v_dd - 4;

	// update the valence of two incident vertices
	m_valences[d1]--;
	m_valences[dd1]--;

	return d2;
}

template <typename PFP>
void SimplifTrian<PFP>::computeVerticesValences(bool gc)
{
	unsigned int end = m_valences.end();

	for (unsigned int it = m_valences.begin(); it != end; m_valences.next(it))
		m_valences[it] = 0;

	for (Dart d = m_map.begin(); d != m_map.end(); m_map.next(d))
		m_valences[d]++;

	if (gc)
	{
		for (unsigned int it = m_valences.begin(); it != end; m_valences.next(it))
		{
			unsigned int& nb = m_valences[it];
			if (nb%2 == 0)
				nb /= 2;
			else
				nb = nb/2 +1;
		}
	}
}

} //namespace Decimation

}

} //namespace Algo

} //namespace CGoGN
