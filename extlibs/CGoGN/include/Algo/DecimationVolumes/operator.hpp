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

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Decimation
{

/************************************************************************************
 *							Collapse Edge Operator	                                *
 ************************************************************************************/
template <typename PFP>
unsigned int CollapseEdgeOperator<PFP>::collapse(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& position)
{
	unsigned int nbCell = 0;

	m.collapseEdge(this->m_edge);
	++nbCell;

	return nbCell;
}

template <typename PFP>
bool CollapseEdgeOperator<PFP>::canCollapse(typename PFP::MAP &m ,Dart d, VertexAttribute<typename PFP::VEC3>& position)
{
	bool canCollapse = true;

//	Dart e = d;
//	do
//	{
//		//isBoundaryVolume
//		if(m.isBoundaryVolume(e))
//		{
//			canCollapse = false;
//		}
//		//l'un de ces 2 voisins existe
//		else if(m.phi3(m.phi2(m.phi1(e))) != m.phi2(m.phi1(e)) && m.phi3(m.phi2(m.phi_1(e))) != m.phi2(m.phi_1(e)))
//		{
//			//l'un de ces 2 voisins est au bord
//			if(m.isBoundaryVolume(m.phi3(m.phi2(m.phi1(e)))) || m.isBoundaryVolume(m.phi3(m.phi2(m.phi_1(e)))))
//			{
//				canCollapse = false;
//
//			}
//		}
//		else
//		{
//			//Edge Criteria Valide
//			if(m.edgeDegree(m.phi1(m.phi2(m.phi_1(e)))) < 3)
//				canCollapse = false;
//			elseframe
//			{
//				//Is inverted
//				Dart a = m.phi3(m.phi2(m.phi1(e)));
//				Dart b = m.phi1(m.phi3(m.phi2(m.phi_1(e))));
//
//				typename PFP::VEC3::DATA_TYPE v1;
//				typename PFP::VEC3::DATA_TYPE v2;
//
//				v1 = Algo::Geometry::tetrahedronSignedVolume<PFP>(m,a,position);
//
//				if (v1 < 0)
//					canCollapse = false;
//
//				v2 = Algo::Geometry::tetrahedronSignedVolume<PFP>(m,b,position);
//				if (v2 < 0)
//					canCollapse = false;
//
//				//CGoGNout << " v2 = " << v2;
//			}
//		}
//
//		e = m.alpha2(e);
//	}while ( e != d && canCollapse);

	return canCollapse;
}


template <typename PFP>
void CollapseEdgeOperator<PFP>::split(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& position)
{
//	Dart d = vs->getEdge() ;
//	Dart dd = m_map.phi2(d) ; 		// get some darts
//	Dart dd2 = vs->getRightEdge() ;
//	Dart d2 = vs->getLeftEdge() ;
//	Dart d1 = m_map.phi2(d2) ;
//	Dart dd1 = m_map.phi2(dd2) ;
//
//	unsigned int v1 = m_map.template getEmbedding<VERTEX>(d) ;				// get the embedding
//	unsigned int v2 = m_map.template getEmbedding<VERTEX>(dd) ;			// of the new vertices
//	unsigned int e1 = m_map.template getEmbedding<EDGE>(m_map.phi1(d)) ;
//	unsigned int e2 = m_map.template getEmbedding<EDGE>(m_map.phi_1(d)) ;	// and new edges
//	unsigned int e3 = m_map.template getEmbedding<EDGE>(m_map.phi1(dd)) ;
//	unsigned int e4 = m_map.template getEmbedding<EDGE>(m_map.phi_1(dd)) ;
//
//	//vertexSplit(vs) ; // split vertex
//	//map.vertexSplit()
//
//	m_map.template setOrbitEmbedding<VERTEX>(d, v1) ;		// embed the
//	m_map.template setOrbitEmbedding<VERTEX>(dd, v2) ;	// new vertices
//	m_map.template setOrbitEmbedding<EDGE>(d1, e1) ;
//	m_map.template setOrbitEmbedding<EDGE>(d2, e2) ;		// and new edges
//	m_map.template setOrbitEmbedding<EDGE>(dd1, e3) ;
//	m_map.template setOrbitEmbedding<EDGE>(dd2, e4) ;
}

/****************************************************************************************************
 *									Operator List									*
 ****************************************************************************************************/

template <typename PFP>
OperatorList<PFP>::~OperatorList()
{
//	for(typename std::list<CollapseSplitOperator<PFP>*>::iterator it= m_ops.begin() ; it != m_ops.end() ; ++it)
//	{
//		delete *it;
//	}
}

template <typename PFP>
void OperatorList<PFP>::coarsen(VertexAttribute<typename PFP::VEC3>& position)
{
	(*m_cur)->collapse(m_map, position);
	++m_cur; // ou ++ ça dépend dans quel sens c'est stocké
}

template <typename PFP>
void OperatorList<PFP>::refine(VertexAttribute<typename PFP::VEC3>& position)
{
	--m_cur; // ou -- ça dépend dans quel sens c'est stocké
	(*m_cur)->split(m_map, position);
}


} //end namespace Decimation

} //namespace Volume

} //end namespace Algo

} //end namespace CGoGN
