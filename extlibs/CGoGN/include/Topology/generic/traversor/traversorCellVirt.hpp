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
VTraversorCell<MAP, ORBIT>::VTraversorCell(const MAP& map, bool forceDartMarker, unsigned int thread) :
	m(map), dmark(NULL), cmark(NULL), quickTraversal(NULL), current(NIL), firstTraversal(true)
{
	if(forceDartMarker)
		dmark = new DartMarker<MAP>(map, thread) ;
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
				cmark = new CellMarker<MAP, ORBIT>(map, thread) ;
			else
				dmark = new DartMarker<MAP>(map, thread) ;
		}
	}
}

template <typename MAP, unsigned int ORBIT>
VTraversorCell<MAP, ORBIT>::~VTraversorCell()
{
	if(dmark)
		delete dmark ;
	else if(cmark)
		delete cmark ;
}

template <typename MAP, unsigned int ORBIT>
Dart VTraversorCell<MAP, ORBIT>::begin()
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
Dart VTraversorCell<MAP, ORBIT>::end()
{
	return NIL ;
}

template <typename MAP, unsigned int ORBIT>
Dart VTraversorCell<MAP, ORBIT>::next()
{
	if(current != NIL)
	{
		if(quickTraversal != NULL)
		{
			cont->next(qCurrent) ;
			current = (*quickTraversal)[qCurrent] ;
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
	}
	return current ;
}

template <typename MAP, unsigned int ORBIT>
void VTraversorCell<MAP, ORBIT>::skip(Dart d)
{
	if(dmark)
		dmark->markOrbit<ORBIT>(d) ;
	else
		cmark->mark(d) ;
}

} // namespace CGoGN
