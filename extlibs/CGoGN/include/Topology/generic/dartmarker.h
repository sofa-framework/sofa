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

#ifndef __DART_MARKER__
#define __DART_MARKER__

#include "Topology/generic/marker.h"
#include "Topology/generic/genericmap.h"
#include "Topology/generic/functor.h"

#include "Utils/static_assert.h"
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/if.hpp>
#include <boost/bind.hpp>

namespace bl = boost::lambda;

namespace CGoGN
{

/**
 * generic class that allows the marking of darts
 * \warning no default constructor
 */
class DartMarkerGen
{
    friend class GenericMap ;
protected:
    typedef AttributeMultiVector<MarkerBool> AMV_MarkerBool;
    AttributeMultiVector<MarkerBool>* m_markVector;

public:
    /**
     * constructor
     */
    DartMarkerGen()
    {}

    virtual ~DartMarkerGen()
    {}

protected:
    // protected copy constructor to forbid its usage
    DartMarkerGen(const DartMarkerGen& /*dm*/)
    {}

} ;

template <typename MAP>
class DartMarkerTmpl : public DartMarkerGen
{
protected:
    MAP& m_map ;

public:
    /**
     * constructor
     * @param map the map on which we work
     */
    DartMarkerTmpl(MAP& map) :
        DartMarkerGen(),
        m_map(map)
    {
        m_markVector = m_map.template askMarkVector<DART>();
    }

    DartMarkerTmpl(const MAP& map) :
        DartMarkerGen(),
        m_map(const_cast<MAP&>(map))
    {
        m_markVector = m_map.template askMarkVector<DART>();
    }

    virtual ~DartMarkerTmpl()
    {
        if (GenericMap::alive(&m_map))
            m_map.template releaseMarkVector<DART>(m_markVector);

    }




protected:
    // protected copy constructor to forbid its usage
    DartMarkerTmpl(const DartMarkerTmpl<MAP>& dm) :
        m_map(dm.m_map)
    {}

public:
    /**
     * mark the dart
     */
    inline void mark(Dart d)
    {
        assert(m_markVector != NULL);
        unsigned int d_index = m_map.dartIndex(d) ;
        m_markVector->setTrue(d_index);
    }

    /**
     * unmark the dart
     */
    inline void unmark(Dart d)
    {
        assert(m_markVector != NULL);
        unsigned int d_index = m_map.dartIndex(d) ;
        m_markVector->setFalse(d_index);
    }

    /**
     * test if dart is marked
     */
    inline bool isMarked(Dart d) const
    {
        assert(m_markVector != NULL);
        unsigned int d_index = m_map.dartIndex(d) ;
        return (*m_markVector)[d_index];
    }

    /**
     * mark the darts of the given cell
     */
    template <unsigned int ORBIT>
    inline void markOrbit(Cell<ORBIT> c)
    {
        assert(m_markVector != NULL);
        m_map.foreach_dart_of_orbit(c,bl::bind(&AMV_MarkerBool::setTrue, boost::ref(*m_markVector), bl::bind(&MAP::dartIndex, boost::cref(m_map), bl::_1)));
    }

    /**
     * unmark the darts of the given cell
     */
    template <unsigned int ORBIT>
    inline void unmarkOrbit(Cell<ORBIT> c)
    {
        assert(m_markVector != NULL);
        m_map.foreach_dart_of_orbit(c,bl::bind(&AMV_MarkerBool::setFalse, boost::ref(*m_markVector), bl::bind(&MAP::dartIndex, boost::cref(m_map), bl::_1)));
    }


    /**
     * mark all darts
     */
    inline void markAll()
    {
        assert(m_markVector != NULL);
        AttributeContainer& cont = m_map.template getAttributeContainer<DART>() ;
        if (cont.hasBrowser())
            for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
                m_markVector->setTrue(i);
        else
            m_markVector->allTrue();
    }

    /**
     * unmark all darts
     */
    inline bool isAllUnmarked()
    {
        assert(m_markVector != NULL);
        AttributeContainer& cont = m_map.template getAttributeContainer<DART>() ;
        if (cont.hasBrowser())
        {
            for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
                if ((*m_markVector)[i])
                    return false ;
            return true ;
        }
        //else
        return m_markVector->isAllFalse();
    }

};

/**
 * class that allows the marking of darts
 * \warning no default constructor
 */
template <typename MAP>
class DartMarker : public DartMarkerTmpl<MAP>
{
public:
    DartMarker( MAP& map) :
        DartMarkerTmpl<MAP>(map)
    {}

    DartMarker(const MAP& map) :
        DartMarkerTmpl<MAP>(map)
    {}

    virtual ~DartMarker()
    {
        unmarkAll() ;
    }

protected:
    DartMarker(const DartMarker& dm) :
        DartMarkerTmpl<MAP>(dm)
    {}

public:
    inline void unmarkAll()
    {
        //		AttributeContainer& cont = this->m_map.template  getAttributeContainer<DART>();
        //		if (cont.hasBrowser())
        //			for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
        //				this->m_markVector->setFalse(i);
        //		else

        // always unmark all darts, it's to dangerous because of markOrbit that can mark dart out of Browser !
        this->m_markVector->allFalse();
    }
} ;

/**
 * class that allows the marking of darts
 * the marked darts are stored to optimize the unmarking task at destruction
 * \warning no default constructor
 */
template <typename MAP>
class DartMarkerStore : public DartMarkerTmpl<MAP>
{
protected:
    std::vector<Dart>* m_markedDarts ;
public:
    DartMarkerStore(MAP& map) :
        DartMarkerTmpl<MAP>(map)
    {
        m_markedDarts = this->m_map.askDartBuffer();
    }

    DartMarkerStore(const MAP& map) :
        DartMarkerTmpl<MAP>(map)
    {
        m_markedDarts =this->m_map.askDartBuffer();
    }

    virtual ~DartMarkerStore()
    {
        unmarkAll() ;
        this->m_map.releaseDartBuffer(m_markedDarts);
    }

protected:
    DartMarkerStore(const DartMarkerStore& dm) :
        DartMarkerTmpl<MAP>(dm),
        m_markedDarts(dm.m_markedDarts)
    {}

public:
    inline void mark(Dart d)
    {
        DartMarkerTmpl<MAP>::mark(d) ;
        m_markedDarts->push_back(d) ;
    }

    template <unsigned int ORBIT>
    inline void markOrbit(Cell<ORBIT> c)
    { //TODO
        //		this->m_map.foreach_dart_of_orbit(c, [&] (Dart d)
        //		{
        //			DartMarkerTmpl<MAP>::mark(d) ;
        //			m_markedDarts->push_back(d) ;
        //		}) ;
        this->m_map.foreach_dart_of_orbit(c, (
                                              bl::bind(&DartMarkerTmpl<MAP>::mark,this, bl::_1)
                                              ,bl::bind(&std::vector<Dart>::push_back, boost::ref(*m_markedDarts), bl::_1)
                                              ));
    }

    inline void unmarkAll()
    {
        for (std::vector<Dart>::iterator it = m_markedDarts->begin(); it != m_markedDarts->end(); ++it)
            this->m_markVector->setFalse(this->m_map.dartIndex(*it));
    }

    inline const std::vector<Dart>& getDartVector() const
    {
        return *m_markedDarts;
    }
} ;

/**
 * class that allows the marking of darts
 * the markers are not unmarked at destruction
 * \warning no default constructor
 */
template <typename MAP>
class DartMarkerNoUnmark : public DartMarkerTmpl<MAP>
{

public:
    DartMarkerNoUnmark(MAP& map) :
        DartMarkerTmpl<MAP>(map)

    {}

    DartMarkerNoUnmark(const MAP& map) :
        DartMarkerTmpl<MAP>(map)

    {}

    virtual ~DartMarkerNoUnmark()
    {
        unmarkAll();
    }

protected:
    DartMarkerNoUnmark(const DartMarkerNoUnmark& dm) :
        DartMarkerTmpl<MAP>(dm)

    {}

public:
    inline void unmarkAll()
    {
        //		AttributeContainer& cont = this->m_map.template  getAttributeContainer<DART>();
        //		if (cont.hasBrowser())
        //			for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
        //				this->m_markVector->setFalse(i);
        //		else

        // always unmark all darts, it's to dangerous because of markOrbit that can mark dart out of Browser !
        this->m_markVector->allFalse();
    }

} ;

// Selector and count functors testing for marker existence
/********************************************************/

template <typename MAP>
class SelectorMarked : public FunctorSelect
{
protected:
    DartMarkerTmpl<MAP>& m_marker ;

public:
    SelectorMarked(DartMarkerTmpl<MAP>& m) :
        m_marker(m)
    {}

    inline bool operator()(Dart d) const
    {
        return m_marker.isMarked(d) ;
    }

    inline FunctorSelect* copy() const
    {
        return new SelectorMarked(m_marker) ;
    }
} ;

template <typename MAP>
class SelectorUnmarked : public FunctorSelect
{
protected:
    DartMarkerTmpl<MAP>& m_marker ;

public:
    SelectorUnmarked(DartMarkerTmpl<MAP>& m) :
        m_marker(m)
    {}

    inline bool operator()(Dart d) const
    {
        return !m_marker.isMarked(d) ;
    }

    inline FunctorSelect* copy() const
    {
        return new SelectorUnmarked(m_marker) ;
    }
} ;

// Functor version (needed for use with foreach_xxx)

template <typename MAP>
class FunctorIsMarked : public FunctorType
{
protected:
    DartMarkerTmpl<MAP>& m_marker ;

public:
    FunctorIsMarked(DartMarkerTmpl<MAP>& dm) :
        m_marker(dm)
    {}

    inline bool operator()(Dart d)
    {
        return m_marker.isMarked(d) ;
    }
} ;

template <typename MAP>
class FunctorIsUnmarked : public FunctorType
{
protected:
    DartMarkerTmpl<MAP>& m_marker ;

public:
    FunctorIsUnmarked(DartMarkerTmpl<MAP>& dm) :
        m_marker(dm)
    {}

    inline bool operator()(Dart d)
    {
        return !m_marker.isMarked(d) ;
    }
} ;

} // namespace CGoGN

#endif
