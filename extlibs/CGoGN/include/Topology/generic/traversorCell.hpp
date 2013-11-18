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

template <typename MAP, unsigned int ORBIT>
TraversorCell<MAP, ORBIT>::TraversorCell(MAP& map, bool forceDartMarker, unsigned int thread) :
	m(map), dmark(NULL), cmark(NULL), quickTraversal(NULL), current(NIL), firstTraversal(true)
{
	if(forceDartMarker)
		dmark = new DartMarker(map, thread) ;
	else
	{
		quickTraversal = map.template getQuickTraversal<ORBIT>() ;
		if(quickTraversal != NULL)
		{
			cont = &(m.template getAttributeContainer<ORBIT>()) ;
		}
		else
		{
			if(map.template isOrbitEmbedded<ORBIT>())
				cmark = new CellMarker<ORBIT>(map, thread) ;
			else
				dmark = new DartMarker(map, thread) ;
		}
	}
}

template <typename MAP, unsigned int ORBIT>
TraversorCell<MAP, ORBIT>::~TraversorCell()
{
	if(dmark)
		delete dmark ;
	else if(cmark)
		delete cmark ;
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorCell<MAP, ORBIT>::begin()
{
	if(quickTraversal != NULL)
	{
		qCurrent = cont->begin() ;
		current = (*quickTraversal)[qCurrent] ;
	}
	else
	{
		if(!firstTraversal)
		{
			if(dmark)
				dmark->unmarkAll() ;
			else
				cmark->unmarkAll() ;
		}

		current = m.begin() ;
		while(current != m.end() && (m.template isBoundaryMarked<MAP::DIMENSION>(current) ))
			m.next(current) ;

		if(current == m.end())
			current = NIL ;
		else
		{
			if(dmark)
				dmark->markOrbit<ORBIT>(current) ;
			else
				cmark->mark(current) ;
		}

		firstTraversal = false ;
	}

	return current ;
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorCell<MAP, ORBIT>::end()
{
	return NIL ;
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorCell<MAP, ORBIT>::next()
{
	assert(current != NIL);
//	if(current != NIL)
//	{
	if(quickTraversal != NULL)
	{
		cont->next(qCurrent) ;
		if (qCurrent != cont->end())
			current = (*quickTraversal)[qCurrent] ;
		else current = NIL;
	}
	else
	{
		if(dmark)
		{
			bool ismarked = dmark->isMarked(current) ;
			while(current != NIL && (ismarked || m.template isBoundaryMarked<MAP::DIMENSION>(current)))
			{
				m.next(current) ;
				if(current == m.end())
					current = NIL ;
				else
					ismarked = dmark->isMarked(current) ;
			}
			if(current != NIL)
				dmark->markOrbit<ORBIT>(current) ;
		}
		else
		{
			bool ismarked = cmark->isMarked(current) ;
			while(current != NIL && (ismarked || m.template isBoundaryMarked<MAP::DIMENSION>(current) ))
			{
				m.next(current) ;
				if(current == m.end())
					current = NIL ;
				else
					ismarked = cmark->isMarked(current) ;
			}
			if(current != NIL)
				cmark->mark(current) ;
		}
	}
//	}
	return current ;
}

template <typename MAP, unsigned int ORBIT>
void TraversorCell<MAP, ORBIT>::skip(Dart d)
{
	if(dmark)
		dmark->markOrbit<ORBIT>(d) ;
	else
		cmark->mark(d) ;
}



//special version (partial specialization) for Genric Map
template <unsigned int ORBIT>
TraversorCell<GenericMap, ORBIT>::TraversorCell(GenericMap& map, bool forceDartMarker, unsigned int thread) :
	m(map), dmark(NULL), cmark(NULL), quickTraversal(NULL), current(NIL), firstTraversal(true)
{
	if(forceDartMarker)
		dmark = new DartMarker(map, thread) ;
	else
	{
		quickTraversal = map.template getQuickTraversal<ORBIT>() ;
		if(quickTraversal != NULL)
		{
			cont = &(m.template getAttributeContainer<ORBIT>()) ;
		}
		else
		{
			if(map.template isOrbitEmbedded<ORBIT>())
				cmark = new CellMarker<ORBIT>(map, thread) ;
			else
				dmark = new DartMarker(map, thread) ;
		}
	}
}

template <unsigned int ORBIT>
TraversorCell<GenericMap, ORBIT>::~TraversorCell()
{
	if(dmark)
		delete dmark ;
	else if(cmark)
		delete cmark ;
}

template <unsigned int ORBIT>
Dart TraversorCell<GenericMap, ORBIT>::begin()
{
	if(quickTraversal != NULL)
	{
		qCurrent = cont->begin() ;
		current = (*quickTraversal)[qCurrent] ;
	}
	else
	{
		if(!firstTraversal)
		{
			if(dmark)
				dmark->unmarkAll() ;
			else
				cmark->unmarkAll() ;
		}

		current = m.begin() ;
		while(current != m.end() && (m.isBoundaryMarkedCurrent(current) ))
			m.next(current) ;

		if(current == m.end())
			current = NIL ;
		else
		{
			if(dmark)
				dmark->markOrbit<ORBIT>(current) ;
			else
				cmark->mark(current) ;
		}

		firstTraversal = false ;
	}

	return current ;
}

template <unsigned int ORBIT>
Dart TraversorCell<GenericMap, ORBIT>::end()
{
	return NIL ;
}

template <unsigned int ORBIT>
Dart TraversorCell<GenericMap, ORBIT>::next()
{
	assert(current != NIL);
//	if(current != NIL)
//	{
	if(quickTraversal != NULL)
	{
		cont->next(qCurrent) ;
		if (qCurrent != cont->end())
			current = (*quickTraversal)[qCurrent] ;
		else current = NIL;
	}
	else
	{
		if(dmark)
		{
			bool ismarked = dmark->isMarked(current) ;
			while(current != NIL && (ismarked || m.isBoundaryMarkedCurrent(current) ))
			{
				m.next(current) ;
				if(current == m.end())
					current = NIL ;
				else
					ismarked = dmark->isMarked(current) ;
			}
			if(current != NIL)
				dmark->markOrbit<ORBIT>(current) ;
		}
		else
		{
			bool ismarked = cmark->isMarked(current) ;
			while(current != NIL && (ismarked || m.isBoundaryMarkedCurrent(current) ))
			{
				m.next(current) ;
				if(current == m.end())
					current = NIL ;
				else
					ismarked = cmark->isMarked(current) ;
			}
			if(current != NIL)
				cmark->mark(current) ;
		}
	}
//	}
	return current ;
}

template <unsigned int ORBIT>
void TraversorCell<GenericMap, ORBIT>::skip(Dart d)
{
	if(dmark)
		dmark->markOrbit<ORBIT>(d) ;
	else
		cmark->mark(d) ;
}




} // namespace CGoGN
