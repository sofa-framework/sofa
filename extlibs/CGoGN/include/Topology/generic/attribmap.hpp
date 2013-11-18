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

template <typename T, unsigned int ORBIT>
inline AttributeHandler<T, ORBIT> AttribMap::addAttribute(const std::string& nameAttr)
{
	if(!isOrbitEmbedded<ORBIT>())
		addEmbedding<ORBIT>() ;
	AttributeMultiVector<T>* amv = m_attribs[ORBIT].addAttribute<T>(nameAttr) ;
	return AttributeHandler<T, ORBIT>(this, amv) ;
}

template <typename T, unsigned int ORBIT>
inline bool AttribMap::removeAttribute(AttributeHandler<T, ORBIT>& attr)
{
	assert(attr.isValid() || !"Invalid attribute handler") ;
	if(m_attribs[attr.getOrbit()].template removeAttribute<T>(attr.getIndex()))
	{
		AttributeMultiVectorGen* amv = attr.getDataVector();
		typedef std::multimap<AttributeMultiVectorGen*, AttributeHandlerGen*>::iterator IT ;
		std::pair<IT, IT> bounds = attributeHandlers.equal_range(amv) ;
		for(IT i = bounds.first; i != bounds.second; ++i)
			(*i).second->setInvalid() ;
		attributeHandlers.erase(bounds.first, bounds.second) ;
		return true ;
	}
	return false ;
}

template <typename T, unsigned int ORBIT>
inline AttributeHandler<T ,ORBIT> AttribMap::getAttribute(const std::string& nameAttr)
{
	AttributeMultiVector<T>* amv = m_attribs[ORBIT].getDataVector<T>(nameAttr) ;
	return AttributeHandler<T, ORBIT>(this, amv) ;
}

template <typename T, unsigned int ORBIT>
inline AttributeHandler<T ,ORBIT> AttribMap::checkAttribute(const std::string& nameAttr)
{
	AttributeHandler<T,ORBIT> att = this->getAttribute<T,ORBIT>(nameAttr);
	if (!att.isValid())
		att = this->addAttribute<T,ORBIT>(nameAttr);
	return att;
}



template <typename T, unsigned int ORBIT>
inline bool AttribMap::swapAttributes(AttributeHandler<T, ORBIT>& attr1, AttributeHandler<T, ORBIT>& attr2)
{
	assert((attr1.isValid() && attr2.isValid()) || !"Invalid attribute handler") ;
//	assert(attr1.getOrbit() == attr2.getOrbit() || !"Cannot swap attributes of different orbits") ;
//	unsigned int orbit = attr1.getOrbit() ;
	unsigned int index1 = attr1.getIndex() ;
	unsigned int index2 = attr2.getIndex() ;
	if(index1 != index2)
		return m_attribs[ORBIT].swapAttributes(index1, index2) ;
	return false ;
}

template <typename T, unsigned int ORBIT>
inline bool AttribMap::copyAttribute(AttributeHandler<T, ORBIT>& dst, AttributeHandler<T, ORBIT>& src)
{
	assert((dst.isValid() && src.isValid()) || !"Invalid attribute handler") ;
//	unsigned int orbit = dst.getOrbit() ;
//	assert(orbit == src.getOrbit() || !"Cannot copy attributes of different orbits") ;
	unsigned int index_dst = dst.getIndex() ;
	unsigned int index_src = src.getIndex() ;
	if(index_dst != index_src)
		return m_attribs[ORBIT].copyAttribute(index_dst, index_src) ;
	return false ;
}

/****************************************
 *               UTILITIES              *
 ****************************************/

template <unsigned int ORBIT>
unsigned int AttribMap::computeIndexCells(AttributeHandler<unsigned int, ORBIT>& idx)
{
	AttributeContainer& cont = m_attribs[ORBIT] ;
	unsigned int cpt = 0 ;
	for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
		idx[i] = cpt++ ;
	return cpt ;
}

template <unsigned int ORBIT>
void AttribMap::bijectiveOrbitEmbedding()
{
	assert(isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded") ;

	AttributeHandler<int, ORBIT> counter = addAttribute<int, ORBIT>("tmpCounter") ;
	counter.setAllValues(int(0)) ;

	DartMarker mark(*this) ;
	for(Dart d = begin(); d != end(); next(d))
	{
		if(!mark.isMarked(d))
		{
			mark.markOrbit<ORBIT>(d) ;
			unsigned int emb = getEmbedding<ORBIT>(d) ;
			if (emb != EMBNULL)
			{
				if (counter[d] > 0)
				{
					unsigned int newEmb = setOrbitEmbeddingOnNewCell<ORBIT>(d) ;
					copyCell<ORBIT>(newEmb, emb) ;
				}
				counter[d]++ ;
			}
		}
	}

	removeAttribute(counter) ;
}

} // namespace CGoGN
