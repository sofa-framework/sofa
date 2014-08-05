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

inline void MapMono::clear(bool removeAttrib)
{
	GenericMap::clear(removeAttrib) ;
	if (removeAttrib)
	{
		m_permutation.clear();
		m_permutation_inv.clear();
		m_involution.clear();
	}
}

/****************************************
 *          DARTS MANAGEMENT            *
 ****************************************/

inline Dart MapMono::newDart()
{
	Dart d = GenericMap::newDart() ;

	for (unsigned int i = 0; i < m_permutation.size(); ++i)
		(*m_permutation[i])[d.index] = d ;
	for (unsigned int i = 0; i < m_permutation_inv.size(); ++i)
		(*m_permutation_inv[i])[d.index] = d ;
	for (unsigned int i = 0; i < m_involution.size(); ++i)
		(*m_involution[i])[d.index] = d ;

	return d ;
}

inline void MapMono::deleteDart(Dart d)
{
	deleteDartLine(d.index) ;
}

inline unsigned int MapMono::dartIndex(Dart d) const
{
	return d.index;
}

inline Dart MapMono::indexDart(unsigned int index) const
{
	return Dart(index);
}

inline unsigned int MapMono::getNbDarts() const
{
	return m_attribs[DART].size() ;
}

inline AttributeContainer& MapMono::getDartContainer()
{
	return m_attribs[DART];
}

/****************************************
 *        RELATIONS MANAGEMENT          *
 ****************************************/

inline void MapMono::addInvolution()
{
	std::stringstream sstm;
	sstm << "involution_" << m_involution.size();
	m_involution.push_back(addRelation(sstm.str()));
}

inline void MapMono::removeLastInvolutionPtr()
{
	m_involution.pop_back();
}

inline void MapMono::addPermutation()
{
	std::stringstream sstm;
	sstm << "permutation_" << m_permutation.size();
	m_permutation.push_back(addRelation(sstm.str()));
	std::stringstream sstm2;
	sstm2 << "permutation_inv_" << m_permutation_inv.size();
	m_permutation_inv.push_back(addRelation(sstm2.str()));
}

inline AttributeMultiVector<Dart>* MapMono::getInvolutionAttribute(unsigned int i)
{
	if (i < m_involution.size())
		return m_involution[i];
	else
		return NULL;
}

inline AttributeMultiVector<Dart>* MapMono::getPermutationAttribute(unsigned int i)
{
	if (i < m_permutation.size())
		return m_permutation[i];
	else
		return NULL;
}

inline AttributeMultiVector<Dart>* MapMono::getPermutationInvAttribute(unsigned int i)
{
	if (i < m_permutation_inv.size())
		return m_permutation_inv[i];
	else
		return NULL;
}

template <int I>
inline Dart MapMono::getInvolution(Dart d) const
{
	return (*m_involution[I])[d.index];
}

template <int I>
inline Dart MapMono::getPermutation(Dart d) const
{
	return (*m_permutation[I])[d.index];
}

template <int I>
inline Dart MapMono::getPermutationInv(Dart d) const
{
	return (*m_permutation_inv[I])[d.index];
}

template <int I>
inline void MapMono::involutionSew(Dart d, Dart e)
{
	assert((*m_involution[I])[d.index] == d) ;
	assert((*m_involution[I])[e.index] == e) ;
	(*m_involution[I])[d.index] = e ;
	(*m_involution[I])[e.index] = d ;
}

template <int I>
inline void MapMono::involutionUnsew(Dart d)
{
	Dart e = (*m_involution[I])[d.index] ;
	(*m_involution[I])[d.index] = d ;
	(*m_involution[I])[e.index] = e ;
}

template <int I>
inline void MapMono::permutationSew(Dart d, Dart e)
{
	Dart f = (*m_permutation[I])[d.index] ;
	Dart g = (*m_permutation[I])[e.index] ;
	(*m_permutation[I])[d.index] = g ;
	(*m_permutation[I])[e.index] = f ;
	(*m_permutation_inv[I])[g.index] = d ;
	(*m_permutation_inv[I])[f.index] = e ;
}

template <int I>
inline void MapMono::permutationUnsew(Dart d)
{
	Dart e = (*m_permutation[I])[d.index] ;
	Dart f = (*m_permutation[I])[e.index] ;
	(*m_permutation[I])[d.index] = f ;
	(*m_permutation[I])[e.index] = e ;
	(*m_permutation_inv[I])[f.index] = d ;
	(*m_permutation_inv[I])[e.index] = e ;
}

inline void MapMono::compactTopo()
{
	std::vector<unsigned int> oldnew;
	m_attribs[DART].compact(oldnew);

	for (unsigned int i = m_attribs[DART].begin(); i != m_attribs[DART].end(); m_attribs[DART].next(i))
	{
		for (unsigned int j = 0; j < m_permutation.size(); ++j)
		{
			Dart d = (*m_permutation[j])[i];
			if (d.index != oldnew[d.index])
				(*m_permutation[j])[i] = Dart(oldnew[d.index]);
		}
		for (unsigned int j = 0; j < m_permutation_inv.size(); ++j)
		{
			Dart d = (*m_permutation_inv[j])[i];
			if (d.index != oldnew[d.index])
				(*m_permutation_inv[j])[i] = Dart(oldnew[d.index]);
		}
		for (unsigned int j = 0; j < m_involution.size(); ++j)
		{
			Dart d = (*m_involution[j])[i];
			if (d.index != oldnew[d.index])
				(*m_involution[j])[i] = Dart(oldnew[d.index]);
		}
	}
}

/****************************************
 *           DARTS TRAVERSALS           *
 ****************************************/

inline Dart MapMono::begin() const
{
	return Dart::create(m_attribs[DART].begin()) ;
}

inline Dart MapMono::end() const
{
	return Dart::create(m_attribs[DART].end()) ;
}

inline void MapMono::next(Dart& d) const
{
	m_attribs[DART].next(d.index) ;
}

template <typename FUNC>
inline void MapMono::foreach_dart(FUNC f)
{
	for (Dart d = begin(); d != end(); next(d))
		f(d);
}

template <typename FUNC>
inline void MapMono::foreach_dart(FUNC& f)
{
	for (Dart d = begin(); d != end(); next(d))
		f(d);
}

} // namespace CGoGN
