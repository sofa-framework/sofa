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

template <int DIM>
ECellDart<DIM>::ECellDart(unsigned int i): m_id(i)
{
	s_cont->refLine(m_id);
}

template <int DIM>
ECellDart<DIM>::ECellDart()
{
	m_id = s_cont->insertLine(); /*CGoGNout << "NEW CELL "<< m_id<< CGoGNendl;*/
}

template <int DIM>
ECellDart<DIM>::ECellDart(const ECellDart<DIM>& ec)
{
	m_id = ec.m_id;
	s_cont->refLine(m_id);
}

template <int DIM>
ECellDart<DIM>::~ECellDart()
{
	s_cont->unrefLine(m_id);
}

template <int DIM>
void ECellDart<DIM>::setContainer(AttributeContainer& cont)
{
	s_cont = &cont;
}

template <int DIM>
void ECellDart<DIM>::setMap(GenericMap& map)
{
	s_map = &map;
}

template <int DIM>
void  ECellDart<DIM>::zero()
{
	s_cont->initLine(m_id);
}

template <int DIM>
void ECellDart<DIM>::operator =(const ECellDart<DIM>& ec)
{
	s_cont->affect(m_id,ec.m_id);
}

template <int DIM>
void ECellDart<DIM>::operator +=(const ECellDart<DIM>& ec)
{
	s_cont->add(m_id,ec.m_id);
}

template <int DIM>
void ECellDart<DIM>::operator -=(const ECellDart<DIM>& ec)
{
	s_cont->sub(m_id,ec.m_id);
}

template <int DIM>
void ECellDart<DIM>::operator *=(double a)
{
	s_cont->mult(m_id,a);
}

template <int DIM>
void ECellDart<DIM>::operator /=(double a)
{
	s_cont->div(m_id,a);
}

template <int DIM>
void ECellDart<DIM>::lerp(const ECellDart<DIM>& ec1, const ECellDart<DIM>& ec2, double a)
{
	s_cont->lerp(m_id, ec1.m_id, ec2.m_id, a);
}

template <int DIM>
ECellDart<DIM> ECellDart<DIM>::operator +(const ECellDart<DIM>& ec)
{
	ECellDart<DIM> x;
	s_cont->affect(x.m_id, m_id);
	s_cont->add(x.m_id, ec.m_id);
	return x;
}

template <int DIM>
ECellDart<DIM> ECellDart<DIM>::operator -(const ECellDart<DIM>& ec)
{
	ECellDart<DIM> x;
	s_cont->affect(x.m_id, m_id);
	s_cont->sub(x.m_id, ec.m_id);
	return x;
}

template <int DIM>
ECellDart<DIM> ECellDart<DIM>::operator *(double a)
{
	ECellDart<DIM> x;
	s_cont->affect(x.m_id, m_id);
	s_cont->mult(x.m_id,a);
	return x;
}

template <int DIM>
ECellDart<DIM> ECellDart<DIM>::operator /(double a)
{
	ECellDart<DIM> x;
	s_cont->affect(x.m_id, m_id);
	s_cont->div(x.m_id, a);
	return x;
}

template <int DIM>
ECellDart<DIM> ECellDart<DIM>::operator[](Dart d)
{
	unsigned int a = s_map->getEmbedding(d,DIM);

	if (a == EMBNULL)
		a = s_map->setOrbitEmbeddingOnNewCell(DIM, d);

	return ECellDart<DIM>(a);
}

template <int DIM>
ECellDart<DIM> ECellDart<DIM>::at(unsigned int i)
{
	return ECellDart<DIM>(i);
}

} //namespace CGoGN
