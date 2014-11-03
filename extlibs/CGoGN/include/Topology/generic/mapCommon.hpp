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

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include "traversor/traversorFactory.h"
namespace bl = boost::lambda;
namespace CGoGN
{

/****************************************
 *           DARTS TRAVERSALS           *
 ****************************************/

template <typename MAP_IMPL>
template <unsigned int ORBIT, unsigned int INCIDENT>
unsigned int MapCommon<MAP_IMPL>::degree(Dart d) const
{
	assert(ORBIT != INCIDENT || !"degree does not manage adjacency counting") ;
	Traversor* t = TraversorFactory<MapCommon<MAP_IMPL> >::createIncident(*this, d, this->dimension(), ORBIT, INCIDENT) ;
	unsigned int cpt = 0;
    t->apply( ++boost::ref(cpt));
	delete t ;
	return cpt;
}

template <typename MAP_IMPL>
template <unsigned int ORBIT>
bool MapCommon<MAP_IMPL>::sameOrbit(Cell<ORBIT> c1, Cell<ORBIT> c2, unsigned int thread) const
{
	TraversorDartsOfOrbit<MapCommon<MAP_IMPL>, ORBIT> tradoo(*this, c1, thread);
	for (Dart x = tradoo.begin(); x != tradoo.end(); x = tradoo.next())
	{
        if (x == c2)
			return true;
	}
	return false;
}

/****************************************
 *         EMBEDDING MANAGEMENT         *
 ****************************************/

template <typename MAP_IMPL>
template <unsigned int ORBIT>
inline unsigned int MapCommon<MAP_IMPL>::getEmbedding(Cell<ORBIT> c) const
{
	assert(this->template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");

	if (ORBIT == DART)
        return this->dartIndex(c);
    return (*this->m_embeddings[ORBIT])[this->dartIndex(c)] ;
}

//template <typename MAP_IMPL>
//template<unsigned int ORBIT>
//inline void MapCommon<MAP_IMPL>::copyCell(Cell<ORBIT> dest, Cell<ORBIT> src) {
//    assert(this->template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
//    const unsigned destEmb = this->getEmbedding(dest);
//    const unsigned srcEmb =  this->getEmbedding(src);
//    assert((destEmb != EMBNULL));
//    assert((srcEmb != EMBNULL));
//    AttributeContainer& cont = this->template getAttributeContainer<ORBIT>();
//    cont.copyLine(destEmb, srcEmb);
//}

template <typename MAP_IMPL>
template <unsigned int ORBIT>
void MapCommon<MAP_IMPL>::setDartEmbedding(Dart d, unsigned int emb)
{
	assert(this->template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");

	unsigned int old = getEmbedding<ORBIT>(d);

	if (old == emb)	// if same emb
		return;		// nothing to do

	if (old != EMBNULL)	// if different
	{
		this->m_attribs[ORBIT].unrefLine(old);	// then unref the old emb
	}

	if (emb != EMBNULL)
		this->m_attribs[ORBIT].refLine(emb);	// ref the new emb

	(*this->m_embeddings[ORBIT])[this->dartIndex(d)] = emb ; // finally affect the embedding to the dart
}

template <typename MAP_IMPL>
template <unsigned int ORBIT>
void MapCommon<MAP_IMPL>::initDartEmbedding(Dart d, unsigned int emb)
{
	assert(this->template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
	assert(getEmbedding<ORBIT>(d) == EMBNULL || !"initDartEmbedding called on already embedded dart");

	if(emb != EMBNULL)
		this->m_attribs[ORBIT].refLine(emb);	// ref the new emb
	(*this->m_embeddings[ORBIT])[this->dartIndex(d)] = emb ; // affect the embedding to the dart
}

template <typename MAP_IMPL>
template <unsigned int ORBIT>
inline void MapCommon<MAP_IMPL>::copyDartEmbedding(Dart dest, Dart src)
{
	assert(this->template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");

	setDartEmbedding<ORBIT>(dest, getEmbedding<ORBIT>(src));
}

/**************************
 *  BOUNDARY MANAGEMENT   *
 **************************/

template <typename MAP_IMPL>
template <unsigned int DIM>
inline void MapCommon<MAP_IMPL>::boundaryMark(Dart d)
{
	this->m_boundaryMarkers[DIM-2]->setTrue(this->dartIndex(d));
}

template <typename MAP_IMPL>
template <unsigned int DIM>
inline void MapCommon<MAP_IMPL>::boundaryUnmark(Dart d)
{
	this->m_boundaryMarkers[DIM-2]->setFalse(this->dartIndex(d));
}

template <typename MAP_IMPL>
template <unsigned int DIM>
inline bool MapCommon<MAP_IMPL>::isBoundaryMarked(Dart d) const
{
	return this->m_boundaryMarkers[DIM-2]->operator[](this->dartIndex(d));
}

template <typename MAP_IMPL>
inline bool MapCommon<MAP_IMPL>::isBoundaryMarkedCurrent(Dart d) const
{
	return this->m_boundaryMarkers[this->dimension()-2]->operator[](this->dartIndex(d));
}

template <typename MAP_IMPL>
inline bool MapCommon<MAP_IMPL>::isBoundaryMarked(unsigned int dim, Dart d) const
{
	switch(dim)
	{
		case 2 : return isBoundaryMarked<2>(d) ; break ;
		case 3 : return isBoundaryMarked<3>(d) ; break ;
		default : return false ; break ;
	}
}

template <typename MAP_IMPL>
template <unsigned int DIM>
void MapCommon<MAP_IMPL>::boundaryUnmarkAll()
{
	this->m_boundaryMarkers[DIM-2]->allFalse();
}

/****************************************
 *        ATTRIBUTES MANAGEMENT         *
 ****************************************/

template <typename MAP_IMPL>
template <typename T, unsigned int ORBIT, typename MAP>
inline AttributeHandler<T, ORBIT, MAP> MapCommon<MAP_IMPL>::addAttribute(const std::string& nameAttr)
{
	if(!this->template isOrbitEmbedded<ORBIT>())
		this->template addEmbedding<ORBIT>() ;
	AttributeMultiVector<T>* amv = this->m_attribs[ORBIT].template addAttribute<T>(nameAttr) ;
	return AttributeHandler<T, ORBIT, MAP>(static_cast<MAP*>(this), amv) ;
}

template <typename MAP_IMPL>
template <typename T, unsigned int ORBIT, typename MAP>
inline bool MapCommon<MAP_IMPL>::removeAttribute(AttributeHandler<T, ORBIT, MAP>& attr)
{
	assert(attr.isValid() || !"Invalid attribute handler") ;
	if(this->m_attribs[attr.getOrbit()].template removeAttribute<T>(attr.getIndex()))
	{
		AttributeMultiVectorGen* amv = attr.getDataVector();
		typedef std::multimap<AttributeMultiVectorGen*, AttributeHandlerGen*>::iterator IT ;
		std::pair<IT, IT> bounds = this->attributeHandlers.equal_range(amv) ;
		for(IT i = bounds.first; i != bounds.second; ++i)
			(*i).second->setInvalid() ;
		this->attributeHandlers.erase(bounds.first, bounds.second) ;
		return true ;
	}
	return false ;
}

template <typename MAP_IMPL>
template <typename T, unsigned int ORBIT, typename MAP>
inline AttributeHandler<T ,ORBIT, MAP> MapCommon<MAP_IMPL>::getAttribute(const std::string& nameAttr)
{
	AttributeMultiVector<T>* amv = this->m_attribs[ORBIT].template getDataVector<T>(nameAttr) ;
	return AttributeHandler<T, ORBIT, MAP>(static_cast<MAP*>(this), amv) ;
}

template <typename MAP_IMPL>
template <typename T, unsigned int ORBIT, typename MAP>
inline AttributeHandler<T ,ORBIT, MAP> MapCommon<MAP_IMPL>::checkAttribute(const std::string& nameAttr)
{
    AttributeHandler<T, ORBIT, MAP> att = this->getAttribute<T,ORBIT, MAP>(nameAttr);
	if (!att.isValid())
        att = this->addAttribute<T, ORBIT, MAP>(nameAttr);
	return att;
}

template <typename MAP_IMPL>
template <typename T, unsigned int ORBIT, typename MAP>
inline bool MapCommon<MAP_IMPL>::swapAttributes(AttributeHandler<T, ORBIT, MAP>& attr1, AttributeHandler<T, ORBIT, MAP>& attr2)
{
	assert((attr1.isValid() && attr2.isValid()) || !"Invalid attribute handler") ;
//	assert(attr1.getOrbit() == attr2.getOrbit() || !"Cannot swap attributes of different orbits") ;
//	unsigned int orbit = attr1.getOrbit() ;
	unsigned int index1 = attr1.getIndex() ;
	unsigned int index2 = attr2.getIndex() ;
	if(index1 != index2)
		return this->m_attribs[ORBIT].swapAttributes(index1, index2) ;
	return false ;
}

template <typename MAP_IMPL>
template <typename T, unsigned int ORBIT, typename MAP>
inline bool MapCommon<MAP_IMPL>::copyAttribute(AttributeHandler<T, ORBIT, MAP>& dst, AttributeHandler<T, ORBIT, MAP>& src)
{
	assert((dst.isValid() && src.isValid()) || !"Invalid attribute handler") ;
//	unsigned int orbit = dst.getOrbit() ;
//	assert(orbit == src.getOrbit() || !"Cannot copy attributes of different orbits") ;
	unsigned int index_dst = dst.getIndex() ;
	unsigned int index_src = src.getIndex() ;
	if(index_dst != index_src)
		return this->m_attribs[ORBIT].copyAttribute(index_dst, index_src) ;
	return false ;
}

//template <typename MAP_IMPL>
//inline DartAttribute<Dart, MAP_IMPL> MapCommon<MAP_IMPL>::getInvolution(unsigned int i)
//{
//	return DartAttribute<Dart, MAP_IMPL>(this, this->getInvolutionAttribute(i));
//}

//template <typename MAP_IMPL>
//inline DartAttribute<Dart, MAP_IMPL> MapCommon<MAP_IMPL>::getPermutation(unsigned int i)
//{
//	return DartAttribute<Dart, MAP_IMPL>(this, this->getPermutationAttribute(i));
//}

//template <typename MAP_IMPL>
//inline DartAttribute<Dart, MAP_IMPL> MapCommon<MAP_IMPL>::getPermutationInv(unsigned int i)
//{
//	return DartAttribute<Dart, MAP_IMPL>(this, this->getPermutationInvAttribute(i));
//}

/****************************************
 *     QUICK TRAVERSAL MANAGEMENT       *
 ****************************************/

template <typename MAP_IMPL>
template <typename MAP, unsigned int ORBIT>
inline void MapCommon<MAP_IMPL>::enableQuickTraversal()
{
	if(this->m_quickTraversal[ORBIT] == NULL)
	{
		if(!this->template isOrbitEmbedded<ORBIT>())
			this->template addEmbedding<ORBIT>() ;
		this->m_quickTraversal[ORBIT] = this->m_attribs[ORBIT].template addAttribute<Dart>("quick_traversal") ;
	}
	updateQuickTraversal<MAP, ORBIT>() ;
}

template <typename MAP_IMPL>
template <typename MAP, unsigned int ORBIT>
inline void MapCommon<MAP_IMPL>::updateQuickTraversal()
{
	assert(this->m_quickTraversal[ORBIT] != NULL || !"updateQuickTraversal on a disabled orbit") ;

//	foreach_cell<ORBIT>(static_cast<MAP&>(*this), [&] (Cell<ORBIT> c) {
//        (*this->m_quickTraversal[ORBIT])[getEmbedding(c)] = c ;
//	}, FORCE_CELL_MARKING);
    foreach_cell<ORBIT>(static_cast<MAP&>(*this), bl::bind(&Dart::operator=, bl::bind(static_cast<Dart& (AttributeMultiVector<Dart>::*)(unsigned)>(&AttributeMultiVector<Dart>::operator[]),boost::ref(*(this->m_quickTraversal[ORBIT])), bl::bind(static_cast<unsigned (MapCommon<MAP_IMPL>::*)(Cell<ORBIT>) const>(&MapCommon<MAP_IMPL>::getEmbedding), boost::cref(*this), bl::_1 )  ), bl::_1), FORCE_CELL_MARKING);
}

template <typename MAP_IMPL>
template <unsigned int ORBIT>
inline const AttributeMultiVector<Dart>* MapCommon<MAP_IMPL>::getQuickTraversal() const
{
	return this->m_quickTraversal[ORBIT] ;
}

template <typename MAP_IMPL>
template <unsigned int ORBIT>
inline void MapCommon<MAP_IMPL>::disableQuickTraversal()
{
	if(this->m_quickTraversal[ORBIT] != NULL)
	{
		this->m_attribs[ORBIT].template removeAttribute<Dart>(this->m_quickTraversal[ORBIT]->getIndex()) ;
		this->m_quickTraversal[ORBIT] = NULL ;
	}
}

template <typename MAP_IMPL>
template <typename MAP, unsigned int ORBIT, unsigned int INCI>
inline void MapCommon<MAP_IMPL>::enableQuickIncidentTraversal()
{
	if(this->m_quickLocalIncidentTraversal[ORBIT][INCI] == NULL)
	{
		if(!this->template isOrbitEmbedded<ORBIT>())
			this->template addEmbedding<ORBIT>() ;
		std::stringstream ss;
		ss << "quickIncidentTraversal_" << INCI;
		this->m_quickLocalIncidentTraversal[ORBIT][INCI] = this->m_attribs[ORBIT].template addAttribute<NoTypeNameAttribute<std::vector<Dart> > >(ss.str()) ;
	}
	updateQuickIncidentTraversal<MAP, ORBIT, INCI>() ;
}

template <typename MAP_IMPL>
template <typename MAP, unsigned int ORBIT, unsigned int INCI>
inline void MapCommon<MAP_IMPL>::updateQuickIncidentTraversal()
{
	assert(this->m_quickLocalIncidentTraversal[ORBIT][INCI] != NULL || !"updateQuickTraversal on a disabled orbit") ;

	AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* ptrVD = this->m_quickLocalIncidentTraversal[ORBIT][INCI];
	this->m_quickLocalIncidentTraversal[ORBIT][INCI] = NULL;

	std::vector<Dart> buffer;
	buffer.reserve(100);

	TraversorCell<MAP, ORBIT> tra_glob(*this);
	for (Dart d = tra_glob.begin(); d != tra_glob.end(); d = tra_glob.next())
	{
		buffer.clear();
		Traversor* tra_loc = TraversorFactory<MAP>::createIncident(*this, d, this->dimension(), ORBIT, INCI);
		for (Dart e = tra_loc->begin(); e != tra_loc->end(); e = tra_loc->next())
			buffer.push_back(e);
		delete tra_loc;
		buffer.push_back(NIL);
		std::vector<Dart>& vd = (*ptrVD)[getEmbedding<ORBIT>(d)];
		vd.reserve(buffer.size());
		vd.assign(buffer.begin(), buffer.end());
	}

	this->m_quickLocalIncidentTraversal[ORBIT][INCI] = ptrVD;
}

template <typename MAP_IMPL>
template <unsigned int ORBIT, unsigned int INCI>
inline const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* MapCommon<MAP_IMPL>::getQuickIncidentTraversal() const
{
	return this->m_quickLocalIncidentTraversal[ORBIT][INCI] ;
}

template <typename MAP_IMPL>
template <unsigned int ORBIT, unsigned int INCI>
inline void MapCommon<MAP_IMPL>::disableQuickIncidentTraversal()
{
	if(this->m_quickLocalIncidentTraversal[ORBIT][INCI] != NULL)
	{
		this->m_attribs[ORBIT].template removeAttribute<Dart>(this->m_quickLocalIncidentTraversal[ORBIT][INCI]->getIndex()) ;
		this->m_quickLocalIncidentTraversal[ORBIT][INCI] = NULL ;
	}
}

template <typename MAP_IMPL>
template <typename MAP, unsigned int ORBIT, unsigned int ADJ>
inline void MapCommon<MAP_IMPL>::enableQuickAdjacentTraversal()
{
	if(this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] == NULL)
	{
		if(!this->template isOrbitEmbedded<ORBIT>())
			this->template addEmbedding<ORBIT>() ;
		std::stringstream ss;
		ss << "quickAdjacentTraversal" << ADJ;
		this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] = this->m_attribs[ORBIT].template addAttribute<NoTypeNameAttribute<std::vector<Dart> > >(ss.str()) ;
	}
	updateQuickAdjacentTraversal<MAP, ORBIT, ADJ>() ;
}

template <typename MAP_IMPL>
template <typename MAP, unsigned int ORBIT, unsigned int ADJ>
inline void MapCommon<MAP_IMPL>::updateQuickAdjacentTraversal()
{
	assert(this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] != NULL || !"updateQuickTraversal on a disabled orbit") ;

	AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* ptrVD = this->m_quickLocalAdjacentTraversal[ORBIT][ADJ];
	this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] = NULL;

	std::vector<Dart> buffer;
	buffer.reserve(100);

	TraversorCell<MAP, ORBIT> tra_glob(*this);
	for (Dart d = tra_glob.begin(); d != tra_glob.end(); d = tra_glob.next())
	{
		buffer.clear();
		Traversor* tra_loc = TraversorFactory<MAP>::createAdjacent(*this, d, this->dimension(), ORBIT, ADJ);
		for (Dart e = tra_loc->begin(); e != tra_loc->end(); e = tra_loc->next())
			buffer.push_back(e);
		buffer.push_back(NIL);
		delete tra_loc;
		std::vector<Dart>& vd = (*ptrVD)[getEmbedding<ORBIT>(d)];
		vd.reserve(buffer.size());
		vd.assign(buffer.begin(),buffer.end());
	}

	this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] = ptrVD;
}

template <typename MAP_IMPL>
template <unsigned int ORBIT, unsigned int ADJ>
inline const AttributeMultiVector<NoTypeNameAttribute<std::vector<Dart> > >* MapCommon<MAP_IMPL>::getQuickAdjacentTraversal() const
{
	return this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] ;
}

template <typename MAP_IMPL>
template <unsigned int ORBIT, unsigned int ADJ>
inline void MapCommon<MAP_IMPL>::disableQuickAdjacentTraversal()
{
	if(this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] != NULL)
	{
		this->m_attribs[ORBIT].template removeAttribute<Dart>(this->m_quickLocalAdjacentTraversal[ORBIT][ADJ]->getIndex()) ;
		this->m_quickLocalAdjacentTraversal[ORBIT][ADJ] = NULL ;
	}
}

} // namespace CGoGN
