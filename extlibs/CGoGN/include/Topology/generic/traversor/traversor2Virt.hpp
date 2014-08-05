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

/*******************************************************************************
					VERTEX CENTERED TRAVERSALS
*******************************************************************************/

// VTraversor2VE

template <typename MAP>
VTraversor2VE<MAP>::VTraversor2VE(const MAP& map, Dart dart) : m(map), start(dart),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<VERTEX,EDGE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<VERTEX>(dart)));
	}
}

template <typename MAP>
Dart VTraversor2VE<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2VE<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2VE<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}

	if(current != NIL)
	{
//		current = m.alpha1(current) ;
		current = m.phi2(m.phi_1(current)) ;
		if(current == start)
			current = NIL ;
	}
	return current ;
}

// VTraversor2VF

template <typename MAP>
VTraversor2VF<MAP>::VTraversor2VF(const MAP& map, Dart dart) : m(map), start(dart),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<VERTEX,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<VERTEX>(dart)));
	}
	else
	{
		if(m.template isBoundaryMarked<2>(start)) // jump over a boundary face
			start = m.phi2(m.phi_1(start)) ;
	}
}

template <typename MAP>
Dart VTraversor2VF<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2VF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2VF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi2(m.phi_1(current)) ;
		if(m.template isBoundaryMarked<2>(current)) // jump over a boundary face
			current = m.phi2(m.phi_1(current)) ;
		if(current == start)
			current = NIL ;
	}
	return current ;
}

// VTraversor2VVaE

template <typename MAP>
VTraversor2VVaE<MAP>::VTraversor2VVaE(const MAP& map, Dart dart) : m(map),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<VERTEX,EDGE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<VERTEX>(dart)));
	}
	else
	{
		start = m.phi2(dart) ;
	}
}

template <typename MAP>
Dart VTraversor2VVaE<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2VVaE<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2VVaE<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi_1(m.phi2(current)) ;
		if(current == start)
			current = NIL ;
	}
	return current ;
}

// VTraversor2VVaF

template <typename MAP>
VTraversor2VVaF<MAP>::VTraversor2VVaF(const MAP& map, Dart dart) : m(map),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<VERTEX,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<VERTEX>(dart)));
	}
	else
	{
		if(m.template isBoundaryMarked<2>(dart))
			dart = m.phi2(m.phi_1(dart)) ;
		start = m.phi1(m.phi1(dart)) ;
		if(start == dart)
			start = m.phi1(dart) ;
		stop = dart ;
	}
}

template <typename MAP>
Dart VTraversor2VVaF<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2VVaF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2VVaF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi1(current) ;
		if(current == stop)
		{
			Dart d = m.phi2(m.phi_1(current)) ;
			if(m.template isBoundaryMarked<2>(d)) // jump over a boundary face
			{
				d = m.phi2(m.phi_1(d)) ;
				current = m.phi1(d);
			}
			else
				current = m.phi1(m.phi1(d)) ;
			if(current == d)
				current = m.phi1(d) ;
			stop = d ;
		}
		if(current == start)
			current = NIL ;
	}
	return current ;
}

/*******************************************************************************
					EDGE CENTERED TRAVERSALS
*******************************************************************************/

// VTraversor2EV

template <typename MAP>
VTraversor2EV<MAP>::VTraversor2EV(const MAP& map, Dart dart) : m(map), start(dart),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<EDGE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<EDGE>(dart)));
	}
}

template <typename MAP>
Dart VTraversor2EV<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2EV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2EV<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi2(current) ;
		if(current == start)
			current = NIL ;
	}
	return current ;
}

// VTraversor2EF

template <typename MAP>
VTraversor2EF<MAP>::VTraversor2EF(const MAP& map, Dart dart) : m(map), start(dart),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<EDGE,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<EDGE>(dart)));
	}
	else
	{
		if(m.template isBoundaryMarked<2>(start))
			start = m.phi2(start) ;
	}
}

template <typename MAP>
Dart VTraversor2EF<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2EF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2EF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi2(current) ;
		if(current == start || m.template isBoundaryMarked<2>(current)) // do not consider a boundary face
			current = NIL ;
	}
	return current ;
}

// VTraversor2EEaV

template <typename MAP>
VTraversor2EEaV<MAP>::VTraversor2EEaV(const MAP& map, Dart dart) : m(map),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<EDGE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<EDGE>(dart)));
	}
	else
	{
		start = m.phi2(m.phi_1(dart)) ;
		stop1 = dart ;
		stop2 = m.phi2(dart) ;
	}
}

template <typename MAP>
Dart VTraversor2EEaV<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2EEaV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2EEaV<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi2(m.phi_1(current)) ;
		if(current == stop1)
			current = m.phi2(m.phi_1(stop2)) ;
		else if(current == stop2)
			current = NIL ;
	}
	return current ;
}

// VTraversor2EEaF

template <typename MAP>
VTraversor2EEaF<MAP>::VTraversor2EEaF(const MAP& map, Dart dart) : m(map),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<EDGE,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<EDGE>(dart)));
	}
	else
	{
		if (m.template isBoundaryMarked<2>(dart))
			stop1 = m.phi2(dart);
		else
			stop1 = dart;
		stop2 = m.phi2(stop1) ;
		start = m.phi1(stop1);
	}
}

template <typename MAP>
Dart VTraversor2EEaF<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2EEaF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2EEaF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi1(current) ;
		if (current == stop1)
		{
			if (!m.template isBoundaryMarked<2>(stop2))
				current = m.phi1(stop2) ;
			else
				current=NIL;
		}
		else if (current == stop2)
			current = NIL ;
	}
	return current ;
}

/*******************************************************************************
					FACE CENTERED TRAVERSALS
*******************************************************************************/

// VTraversor2FV

template <typename MAP>
VTraversor2FV<MAP>::VTraversor2FV(const MAP& map, Dart dart) : m(map), start(dart),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<FACE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<FACE>(dart)));
	}
}

template <typename MAP>
Dart VTraversor2FV<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2FV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2FV<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi1(current) ;
		if(current == start)
			current = NIL ;
	}
	return current ;
}

// VTraversor2FFaV

template <typename MAP>
VTraversor2FFaV<MAP>::VTraversor2FFaV(const MAP& map, Dart dart) : m(map),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<FACE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<FACE>(dart)));
	}
	else
	{
		start = m.phi2(m.phi_1(m.phi2(m.phi_1(dart)))) ;
		while (start == dart)
		{
			dart = m.phi1(dart);
			start = m.phi2(m.phi_1(m.phi2(m.phi_1(dart)))) ;
		}
		current = start ;
		stop = dart ;
		if(m.template isBoundaryMarked<2>(start))
			start = next() ;
	}
}

template <typename MAP>
Dart VTraversor2FFaV<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2FFaV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2FFaV<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		current = m.phi2(m.phi_1(current)) ;
		if(current == stop)
		{
			Dart d = m.phi1(current) ;
			current = m.phi2(m.phi_1(m.phi2(m.phi_1(d)))) ;
			if(current == d)
			{
				stop = m.phi1(d);
				current = m.phi2(d);
				return next() ;
			}
			stop = d ;
			if(m.template isBoundaryMarked<2>(current))
				return next() ;
		}
		if(current == start)
			current = NIL ;
	}
	return current ;
}

// VTraversor2FFaE

template <typename MAP>
VTraversor2FFaE<MAP>::VTraversor2FFaE(const MAP& map, Dart dart) : m(map),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<FACE,EDGE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT  = &(quickTraversal->operator[](map.template getEmbedding<FACE>(dart)));
	}
	else
	{
		start = m.phi2(dart) ;
		while(start != NIL && m.template isBoundaryMarked<2>(start))
		{
			start = m.phi2(m.phi1(m.phi2(start))) ;
			if(start == m.phi2(dart))
				start = NIL ;
		}
	}
}

template <typename MAP>
Dart VTraversor2FFaE<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return *m_ItDarts++;
	}

	current = start ;
	return current ;
}

template <typename MAP>
Dart VTraversor2FFaE<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Dart VTraversor2FFaE<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
	if(current != NIL)
	{
		do
		{
			current = m.phi2(m.phi1(m.phi2(current))) ;
		} while(m.template isBoundaryMarked<2>(current)) ;
		if(current == start)
			current = NIL ;
	}
	return current ;
}


//
//template<typename MAP>
//VTraversor2<MAP>* VTraversor2<MAP>::createIncident(MAP& map, Dart dart, unsigned int orbX, unsigned int orbY)
//{
//	int code = 0x100*(orbX-VERTEX) + orbY-VERTEX;
//
//	switch(code)
//	{
//	case 0x0001:
//		return new VTraversor2VE<MAP>(map,dart);
//		break;
//	case 0x0002:
//		return new VTraversor2VF<MAP>(map,dart);
//		break;
//	case 0x0100:
//		return new VTraversor2EV<MAP>(map,dart);
//		break;
//	case 0x0102:
//		return new VTraversor2EF<MAP>(map,dart);
//		break;
//	case 0x0200:
//		return new VTraversor2FV<MAP>(map,dart);
//		break;
//	case 0x0201:
//		return new VTraversor2FE<MAP>(map,dart);
//		break;
//	default:
//		return NULL;
//		break;
//	}
//	return NULL;
//}
//
//template<typename MAP>
//VTraversor2<MAP>* VTraversor2<MAP>::createAdjacent(MAP& map, Dart dart, unsigned int orbX, unsigned int orbY)
//{
//	int code = 0x100*(orbX-VERTEX) + orbY-VERTEX;
//	switch(code)
//	{
//	case 0x0001:
//		return new VTraversor2VVaE<MAP>(map,dart);
//		break;
//	case 0x0002:
//		return new VTraversor2VVaF<MAP>(map,dart);
//		break;
//	case 0x0100:
//		return new VTraversor2EEaV<MAP>(map,dart);
//		break;
//	case 0x0102:
//		return new VTraversor2EEaF<MAP>(map,dart);
//		break;
//	case 0x0200:
//		return new VTraversor2FFaV<MAP>(map,dart);
//		break;
//	case 0x0201:
//		return new VTraversor2FFaE<MAP>(map,dart);
//		break;
//	default:
//		return NULL;
//		break;
//	}
//	return NULL;
//}

} // namespace CGoGN
