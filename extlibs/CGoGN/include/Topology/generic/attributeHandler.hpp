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

#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <vector>
#include <Algo/Topo/embedding.h>
namespace CGoGN
{

template <typename T, unsigned int ORBIT, typename MAP>
inline void AttributeHandler<T, ORBIT, MAP>::registerInMap()
{
    boost::mutex::scoped_lock lockAH(m_map->attributeHandlersMutex);
	m_map->attributeHandlers.insert(std::pair<AttributeMultiVectorGen*, AttributeHandlerGen*>(m_attrib, this)) ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline void AttributeHandler<T, ORBIT, MAP>::unregisterFromMap()
{
	typedef std::multimap<AttributeMultiVectorGen*, AttributeHandlerGen*>::iterator IT ;

    boost::mutex::scoped_lock lockAH(m_map->attributeHandlersMutex);
    std::pair<IT, IT> bounds = m_map->attributeHandlers.equal_range(m_attrib) ;
	for(IT i = bounds.first; i != bounds.second; ++i)
	{
		if((*i).second == this)
		{
			m_map->attributeHandlers.erase(i) ;
			return ;
		}
	}
	assert(false || !"Should not get here") ;
}

// =================================================================

template <typename T, unsigned int ORBIT, typename MAP>
AttributeHandler<T, ORBIT, MAP>::AttributeHandler() :
	AttributeHandlerGen(false),
	m_map(NULL),
	m_attrib(NULL)
{}

template <typename T, unsigned int ORBIT, typename MAP>
AttributeHandler<T, ORBIT, MAP>::AttributeHandler(MAP* m, AttributeMultiVector<T>* amv) :
	AttributeHandlerGen(false),
	m_map(m),
	m_attrib(amv)
{
	if(m != NULL && amv != NULL && amv->getIndex() != AttributeContainer::UNKNOWN)
	{
		assert(ORBIT == amv->getOrbit() || !"AttributeHandler: orbit incompatibility") ;
		valid = true ;
		registerInMap() ;
	}
	else
		valid = false ;
}

template <typename T, unsigned int ORBIT, typename MAP>
AttributeHandler<T, ORBIT, MAP>::AttributeHandler(const AttributeHandler<T, ORBIT, MAP>& ta) :
	AttributeHandlerGen(ta.valid),
	m_map(ta.m_map),
	m_attrib(ta.m_attrib)
{
	if(valid)
		registerInMap() ;
}

//template <typename T, unsigned int ORBIT>
//template <unsigned int ORBIT2>
//template <typename T, unsigned int ORBIT>
//template <unsigned int ORBIT2>
//AttributeHandler<T, ORBIT>::AttributeHandler(const AttributeHandler<T, ORBIT2>& h) :
//	AttributeHandlerGen(h.m_map, h.valid)
//{
//	m_attrib = h.m_attrib;
//	if(m_attrib->getOrbit() == ORBIT2)
//	{
//		if(valid)
//			registerInMap() ;
//	}
//	else
//		valid = false;
//}

template <typename T, unsigned int ORBIT, typename MAP>
inline AttributeHandler<T, ORBIT, MAP>& AttributeHandler<T, ORBIT, MAP>::operator=(const AttributeHandler<T, ORBIT, MAP>& ta)
{
	if(valid)
		unregisterFromMap() ;
	m_map = ta.m_map ;
	m_attrib = ta.m_attrib ;
	valid = ta.valid ;
	if(valid)
		registerInMap() ;
	return *this ;
}

//template <typename T, unsigned int ORBIT>
//template <unsigned int ORBIT2>
//inline AttributeHandler<T, ORBIT>& AttributeHandler<T, ORBIT>::operator=(const AttributeHandler<T, ORBIT2>& ta)
//{
//	if(valid)
//		unregisterFromMap() ;
//	m_map = ta.map() ;
//	m_attrib = ta.getDataVector() ;
//	valid = ta.isValid() ;
//	if(valid)
//		registerInMap() ;
//	return *this ;
//}

template <typename T, unsigned int ORBIT, typename MAP>
AttributeHandler<T, ORBIT, MAP>::~AttributeHandler()
{
	if(valid)
		unregisterFromMap() ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline AttributeMultiVector<T>* AttributeHandler<T, ORBIT, MAP>::getDataVector() const
{
	return m_attrib ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline AttributeMultiVectorGen* AttributeHandler<T, ORBIT, MAP>::getDataVectorGen() const
{
	return m_attrib ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline int AttributeHandler<T, ORBIT, MAP>::getSizeOfType() const
{
	return sizeof(T) ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline unsigned int AttributeHandler<T, ORBIT, MAP>::getOrbit() const
{
	return ORBIT ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline unsigned int AttributeHandler<T, ORBIT, MAP>::getIndex() const
{
	return m_attrib->getIndex() ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline const std::string& AttributeHandler<T, ORBIT, MAP>::name() const
{
	return m_attrib->getName() ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline const std::string& AttributeHandler<T, ORBIT, MAP>::typeName() const
{
	return m_attrib->getTypeName();
}


template <typename T, unsigned int ORBIT, typename MAP>
inline unsigned int AttributeHandler<T, ORBIT, MAP>::nbElements() const
{
	return m_map->template getAttributeContainer<ORBIT>().size() ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline T& AttributeHandler<T, ORBIT, MAP>::operator[](Cell<ORBIT> c)
{
	assert(valid || !"Invalid AttributeHandler") ;
	unsigned int a = m_map->getEmbedding(c) ;

	if (a == EMBNULL)
		a = Algo::Topo::setOrbitEmbeddingOnNewCell(*m_map, c) ;
	return m_attrib->operator[](a) ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline const T& AttributeHandler<T, ORBIT, MAP>::operator[](Cell<ORBIT> c) const
{
	assert(valid || !"Invalid AttributeHandler") ;
	unsigned int a = m_map->getEmbedding(c) ;
	return m_attrib->operator[](a) ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline T& AttributeHandler<T, ORBIT, MAP>::operator[](unsigned int a)
{
	assert(valid || !"Invalid AttributeHandler") ;
	return m_attrib->operator[](a) ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline const T& AttributeHandler<T, ORBIT, MAP>::operator[](unsigned int a) const
{
	assert(valid || !"Invalid AttributeHandler") ;
	return m_attrib->operator[](a) ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline unsigned int AttributeHandler<T, ORBIT, MAP>::insert(const T& elt)
{
	assert(valid || !"Invalid AttributeHandler") ;
	unsigned int idx = m_map->template getAttributeContainer<ORBIT>().insertLine() ;
	m_attrib->operator[](idx) = elt ;
	return idx ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline unsigned int AttributeHandler<T, ORBIT, MAP>::newElt()
{
	assert(valid || !"Invalid AttributeHandler") ;
	unsigned int idx = m_map->template getAttributeContainer<ORBIT>().insertLine() ;
	return idx ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline void AttributeHandler<T, ORBIT, MAP>::setAllValues(const T& v)
{
	for(unsigned int i = begin(); i != end(); next(i))
		m_attrib->operator[](i) = v ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline unsigned int AttributeHandler<T, ORBIT, MAP>::begin() const
{
	assert(valid || !"Invalid AttributeHandler") ;
	return m_map->template getAttributeContainer<ORBIT>().begin() ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline unsigned int AttributeHandler<T, ORBIT, MAP>::end() const
{
	assert(valid || !"Invalid AttributeHandler") ;
	return m_map->template getAttributeContainer<ORBIT>().end() ;
}

template <typename T, unsigned int ORBIT, typename MAP>
inline void AttributeHandler<T, ORBIT, MAP>::next(unsigned int& iter) const
{
	assert(valid || !"Invalid AttributeHandler") ;
	m_map->template getAttributeContainer<ORBIT>().next(iter) ;
}


namespace Parallel
{

template <typename FUNC>
class ThreadFunctionAttrib
{
protected:
    std::vector<unsigned int>& m_ids;
    boost::barrier& m_sync1;
    boost::barrier& m_sync2;
    bool& m_finished;
    unsigned int m_id;
    FUNC m_lambda;
public:
    ThreadFunctionAttrib(FUNC func, std::vector<unsigned int>& vid, boost::barrier& s1, boost::barrier& s2, bool& finished, unsigned int id):
        m_ids(vid), m_sync1(s1), m_sync2(s2), m_finished(finished), m_id(id), m_lambda(func)
    {
    }

    ThreadFunctionAttrib(const ThreadFunctionAttrib& tf):
        m_ids(tf.m_ids), m_sync1(tf.m_sync1), m_sync2(tf.m_sync2), m_finished(tf.m_finished), m_id(tf.m_id), m_lambda(tf.m_lambda){}

    void operator()()
    {
        while (!m_finished)
        {
            for (std::vector<unsigned int>::const_iterator it = m_ids.begin(); it != m_ids.end(); ++it)
                m_lambda(*it,m_id);
            m_sync1.wait();
            m_sync2.wait();
        }
    }
};



template <typename ATTR, typename FUNC>
void foreach_attribute(ATTR& attribute, FUNC func, unsigned int nbthread)
{
    // thread 0 is for attribute traversal
    unsigned int nbth = nbthread -1;

    std::vector< unsigned int >* vd = new std::vector< unsigned int >[nbth];
    for (unsigned int i = 0; i < nbth; ++i)
        vd[i].reserve(SIZE_BUFFER_THREAD);

    unsigned int nb = 0;
    unsigned int attIdx = attribute.begin();
    while ((attIdx != attribute.end()) && (nb < nbth*SIZE_BUFFER_THREAD) )
    {
        vd[nb%nbth].push_back(attIdx);
        nb++;
        attribute.next(attIdx);
    }
    boost::barrier sync1(nbth+1);
    boost::barrier sync2(nbth+1);
    bool finished=false;


    boost::thread** threads = new boost::thread*[nbth];
    ThreadFunctionAttrib<FUNC>** tfs = new ThreadFunctionAttrib<FUNC>*[nbth];

    for (unsigned int i = 0; i < nbth; ++i)
    {
        tfs[i] = new ThreadFunctionAttrib<FUNC>(func, vd[i], sync1,sync2,finished,1+i);
        threads[i] = new boost::thread( boost::ref( *(tfs[i]) ) );
    }

    // and continue to traverse the map
    std::vector< unsigned int >* tempo = new std::vector< unsigned int >[nbth];
    for (unsigned int i = 0; i < nbth; ++i)
        tempo[i].reserve(SIZE_BUFFER_THREAD);

    while (attIdx != attribute.end())
    {
        for (unsigned int i = 0; i < nbth; ++i)
            tempo[i].clear();
        unsigned int nb = 0;

        while ((attIdx != attribute.end()) && (nb < nbth*SIZE_BUFFER_THREAD) )
        {
            tempo[nb%nbth].push_back(attIdx);
            nb++;
            attribute.next(attIdx);
        }
        sync1.wait();// wait for all thread to finish its vector
        for (unsigned int i = 0; i < nbth; ++i)
            vd[i].swap(tempo[i]);
        sync2.wait();// everybody refilled then go
    }

    sync1.wait();// wait for all thread to finish its vector
    finished = true;// say finsih to everyone
    sync2.wait(); // just wait for last barrier wait !

    //wait for all theads to be finished
    for (unsigned int i = 0; i < nbth; ++i)
    {
        threads[i]->join();
        delete threads[i];
        delete tfs[i];
    }
    delete[] tfs;
    delete[] threads;
    delete[] vd;
    delete[] tempo;
}

}

} //namespace CGoGN
