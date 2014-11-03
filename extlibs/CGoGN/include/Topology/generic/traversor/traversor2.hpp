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

// Traversor2VE

template <typename MAP>
Traversor2VE<MAP>::Traversor2VE(const MAP& map, Vertex v) : m(map), start(v),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<VERTEX,EDGE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(v)));
	}
}

template <typename MAP>
Edge Traversor2VE<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return Edge(*m_ItDarts++);
	}

	current = start ;
	return current ;
}

template <typename MAP>
Edge Traversor2VE<MAP>::end()
{
	return Edge(NIL) ;
}

template <typename MAP>
Edge Traversor2VE<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return Edge(*m_ItDarts++);
	}

    if(!current.isNil())
	{
//		current = m.alpha1(current) ;
        current = Edge(m.phi2(m.phi_1(current))) ;
        if(current == start)
			current = NIL ;
	}
	return current ;
}

// Traversor2VF

template <typename MAP>
Traversor2VF<MAP>::Traversor2VF(const MAP& map, Vertex v) : m(map), start(v),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<VERTEX,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(v)));
	}
	else
	{
		if(m.template isBoundaryMarked<2>(start)) // jump over a boundary face
			start = m.phi2(m.phi_1(start)) ;
	}
}

template <typename MAP>
Face Traversor2VF<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return Face(*m_ItDarts++);
	}

	current = start ;
	return current ;
}

template <typename MAP>
Face Traversor2VF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Face Traversor2VF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return Face(*m_ItDarts++);
	}
    if(!current.isNil())
	{
		current = m.phi2(m.phi_1(current)) ;
		if(m.template isBoundaryMarked<2>(current)) // jump over a boundary face
			current = m.phi2(m.phi_1(current)) ;
        if(current == start)
			current = NIL ;
	}
	return current ;
}

// Traversor2VVaE

template <typename MAP>
Traversor2VVaE<MAP>::Traversor2VVaE(const MAP& map, Vertex v) : m(map), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<VERTEX,EDGE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(v)));
	}
	else
	{
        start = m.phi2(v) ;
	}
}

template <typename MAP>
Vertex Traversor2VVaE<MAP>::begin()
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
Vertex Traversor2VVaE<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Vertex Traversor2VVaE<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return Vertex(*m_ItDarts++);
	}
    if(!current.isNil())
	{
		current = m.phi_1(m.phi2(current)) ;
        if(current == start)
			current = NIL ;
	}
	return current ;
}

// Traversor2VVaF

template <typename MAP>
Traversor2VVaF<MAP>::Traversor2VVaF(const MAP& map, Vertex v) : m(map), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<VERTEX,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(v)));
	}
	else
	{
        if(m.template isBoundaryMarked<2>(v))
            v = m.phi2(m.phi_1(v)) ;
        start = m.phi1(m.phi1(v)) ;
        if(start == v)
            start = m.phi1(v) ;
		stop = v ;
	}
}

template <typename MAP>
Vertex Traversor2VVaF<MAP>::begin()
{
	if(m_QLT != NULL)
	{
		m_ItDarts = m_QLT->begin();
		return Vertex(*m_ItDarts++);
	}

	current = start ;
	return current ;
}

template <typename MAP>
Vertex Traversor2VVaF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Vertex Traversor2VVaF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
    if(!current.isNil())
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

// Traversor2EV

template <typename MAP>
Traversor2EV<MAP>::Traversor2EV(const MAP& map, Edge e) : m(map), start(e), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<EDGE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(e)));
	}
}

template <typename MAP>
Vertex Traversor2EV<MAP>::begin()
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
Vertex Traversor2EV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Vertex Traversor2EV<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
    if(!current.isNil())
	{
		current = m.phi2(current) ;
        if(current == start)
			current = NIL ;
	}
	return current ;
}

// Traversor2EF

template <typename MAP>
Traversor2EF<MAP>::Traversor2EF(const MAP& map, Edge e) : m(map), start(e),m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<EDGE,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(e)));
	}
	else
	{
        if(m.template isBoundaryMarked<2>(start))
            start = m.phi2(start) ;
	}
}

template <typename MAP>
Face Traversor2EF<MAP>::begin()
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
Face Traversor2EF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Face Traversor2EF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
    if(!current.isNil())
	{
		current = m.phi2(current) ;
        if(current == start || m.template isBoundaryMarked<2>(current)) // do not consider a boundary face
			current = NIL ;
	}
	return current ;
}

// Traversor2EEaV

template <typename MAP>
Traversor2EEaV<MAP>::Traversor2EEaV(const MAP& map, Edge e) : m(map), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<EDGE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(e)));
	}
	else
	{
        start = m.phi2(m.phi_1(e)) ;
		stop1 = e ;
        stop2 = m.phi2(e) ;
	}
}

template <typename MAP>
Edge Traversor2EEaV<MAP>::begin()
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
Edge Traversor2EEaV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Edge Traversor2EEaV<MAP>::next()
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

// Traversor2EEaF

template <typename MAP>
Traversor2EEaF<MAP>::Traversor2EEaF(const MAP& map, Edge e) : m(map), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<EDGE,FACE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(e)));
	}
	else
	{
        if (m.template isBoundaryMarked<2>(e))
            stop1 = m.phi2(e);
		else
			stop1 = e;
        stop2 = m.phi2(stop1) ;
        start = m.phi1(stop1);
	}
}

template <typename MAP>
Edge Traversor2EEaF<MAP>::begin()
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
Edge Traversor2EEaF<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Edge Traversor2EEaF<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
    if(!current.isNil())
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

// Traversor2FV

template <typename MAP>
Traversor2FV<MAP>::Traversor2FV(const MAP& map, Face f) : m(map), start(Vertex::convertCell(f)), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<FACE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(f)));
	}
}

template <typename MAP>
Vertex Traversor2FV<MAP>::begin()
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
Vertex Traversor2FV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Vertex Traversor2FV<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
    if(!current.isNil())
	{
		current = m.phi1(current) ;
        if(current == start)
			current = NIL ;
	}
	return current ;
}

// Traversor2FE

template <typename MAP>
Traversor2FE<MAP>::Traversor2FE(const MAP& map, Face f) : m(map), start(Edge::convertCell(f)), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickIncidentTraversal<FACE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(f)));
	}
}

template <typename MAP>
Edge Traversor2FE<MAP>::begin()
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
Edge Traversor2FE<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Edge Traversor2FE<MAP>::next()
{
	if(m_QLT != NULL)
    {
		return *m_ItDarts++;
	}
    if(!current.isNil())
    {
		current = m.phi1(current) ;
        if(current == start)
			current = NIL ;
	}
	return current ;
}

// Traversor2FFaV

template <typename MAP>
Traversor2FFaV<MAP>::Traversor2FFaV(const MAP& map, Face f) : m(map), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<FACE,VERTEX>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(f)));
	}
	else
	{
        start = m.phi2(m.phi_1(m.phi2(m.phi_1(f)))) ;
        while (start == f)
		{
            f = m.phi1(f);
            start = m.phi2(m.phi_1(m.phi2(m.phi_1(f)))) ;
		}
		current = start ;
		stop = f ;
		if(m.template isBoundaryMarked<2>(start))
			start = next() ;
	}
}

template <typename MAP>
Face Traversor2FFaV<MAP>::begin()
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
Face Traversor2FFaV<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Face Traversor2FFaV<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
    if(!current.isNil())
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
		}

		if(m.template isBoundaryMarked<2>(current))
			return next() ;

        if(current == start)
			current = NIL ;
	}
	return current ;
}

// Traversor2FFaE

template <typename MAP>
Traversor2FFaE<MAP>::Traversor2FFaE(const MAP& map, Face f) : m(map), m_QLT(NULL)
{
	const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* quickTraversal = map.template getQuickAdjacentTraversal<FACE,EDGE>() ;
	if (quickTraversal != NULL)
	{
		m_QLT = &(quickTraversal->operator[](map.getEmbedding(f)));
	}
	else
	{
        start = m.phi2(f) ;
        while((!start.isNil()) && m.template isBoundaryMarked<2>(start))
		{
			start = m.phi2(m.phi1(m.phi2(start))) ;
            if(start == m.phi2(f))
				start = NIL ;
		}
	}
}

template <typename MAP>
Face Traversor2FFaE<MAP>::begin()
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
Face Traversor2FFaE<MAP>::end()
{
	return NIL ;
}

template <typename MAP>
Face Traversor2FFaE<MAP>::next()
{
	if(m_QLT != NULL)
	{
		return *m_ItDarts++;
	}
    if(!current.isNil())
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

//template<typename MAP>
//Traversor2<MAP>* Traversor2<MAP>::createIncident(MAP& map, Dart dart, unsigned int orbX, unsigned int orbY)
//{
//	int code = 0x100*(orbX-VERTEX) + orbY-VERTEX;
//
//	switch(code)
//	{
//	case 0x0001:
//		return new Traversor2VE<MAP>(map,dart);
//		break;
//	case 0x0002:
//		return new Traversor2VF<MAP>(map,dart);
//		break;
//	case 0x0100:
//		return new Traversor2EV<MAP>(map,dart);
//		break;
//	case 0x0102:
//		return new Traversor2EF<MAP>(map,dart);
//		break;
//	case 0x0200:
//		return new Traversor2FV<MAP>(map,dart);
//		break;
//	case 0x0201:
//		return new Traversor2FE<MAP>(map,dart);
//		break;
//	default:
//		return NULL;
//		break;
//	}
//	return NULL;
//}
//
//template<typename MAP>
//Traversor2<MAP>* Traversor2<MAP>::createAdjacent(MAP& map, Dart dart, unsigned int orbX, unsigned int orbY)
//{
//	int code = 0x100*(orbX-VERTEX) + orbY-VERTEX;
//	switch(code)
//	{
//	case 0x0001:
//		return new Traversor2VVaE<MAP>(map,dart);
//		break;
//	case 0x0002:
//		return new Traversor2VVaF<MAP>(map,dart);
//		break;
//	case 0x0100:
//		return new Traversor2EEaV<MAP>(map,dart);
//		break;
//	case 0x0102:
//		return new Traversor2EEaF<MAP>(map,dart);
//		break;
//	case 0x0200:
//		return new Traversor2FFaV<MAP>(map,dart);
//		break;
//	case 0x0201:
//		return new Traversor2FFaE<MAP>(map,dart);
//		break;
//	default:
//		return NULL;
//		break;
//	}
//	return NULL;
//}

} // namespace CGoGN
