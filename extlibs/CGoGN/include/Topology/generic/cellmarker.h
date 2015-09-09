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

#ifndef __CELL_MARKER__
#define __CELL_MARKER__

#include "Topology/generic/marker.h"
#include "Topology/generic/genericmap.h"
#include "Topology/generic/functor.h"
#include "Algo/Topo/embedding.h"

#include "Utils/static_assert.h"

namespace CGoGN
{
/**
 * @brief The CellMarkerGen class
 * @warning CellMarkerGen is no polymorphic version of CellMarker
 */
class CellMarkerGen
{
	friend class GenericMap ;

protected:
	unsigned int m_cell ;

public:
	CellMarkerGen(unsigned int cell) :
		m_cell(cell)
	{}

	virtual ~CellMarkerGen()
	{}

    inline unsigned int getCell() { return m_cell ; }

protected:
	// protected copy constructor to forbid its usage
	CellMarkerGen(const CellMarkerGen& /*cm*/)
	{}

};

/**
 * generic class that allows the marking of cells
 * \warning no default constructor
 */
template <typename MAP, unsigned int CELL>
class CellMarkerBase : public CellMarkerGen
{
protected:
	MAP& m_map ;
    AttributeMultiVector<MarkerBool>* m_markVector ;
public:
	/**
	 * constructor
	 * @param map the map on which we work
	 */
	CellMarkerBase(MAP& map) :
		CellMarkerGen(CELL),
		m_map(map)
	{
		if(!m_map.template isOrbitEmbedded<CELL>())
			m_map.template addEmbedding<CELL>() ;
		m_markVector = m_map.template askMarkVector<CELL>();
	}

	CellMarkerBase(const MAP& map) :
		CellMarkerGen(CELL),
		m_map(const_cast<MAP&>(map))
	{
		if(!m_map.template isOrbitEmbedded<CELL>())
			m_map.template addEmbedding<CELL>() ;
		m_markVector = m_map.template askMarkVector<CELL>();
	}

	virtual ~CellMarkerBase()
	{
		this->m_markVector->allFalse();
		if (GenericMap::alive(&m_map))
			m_map.template releaseMarkVector<CELL>(m_markVector);
	}

	/**
	 * @brief update: realloc the marker in map
	 * @warning call only after map cleaning
	 */
	void update()
	{
		if(!m_map.template isOrbitEmbedded<CELL>())
			m_map.template addEmbedding<CELL>() ;
		m_markVector = m_map.template askMarkVector<CELL>();
	}


protected:
	// protected copy constructor to forbid its usage
	CellMarkerBase(const CellMarkerBase<MAP, CELL>& cm) :
		m_map(cm.m_map),
		CellMarkerGen(CELL)
	{}

public:
	/**
	 * mark the cell of dart
	 */
	inline void mark(Cell<CELL> c)
	{
        assert(m_markVector != NULL);
        unsigned int a = m_map.getEmbedding(c) ;
		if (a == EMBNULL)
            a = Algo::Topo::template setOrbitEmbeddingOnNewCell<CELL, MAP>(m_map, c) ;

		m_markVector->setTrue(a);
	}

	/**
	 * unmark the cell of dart
	 */
	inline void unmark(Cell<CELL> c)
	{
		assert(m_markVector != NULL);
//        std::cerr << "unmarking dart " << c << " currLvl " << m_map.getCurrentLevel() << std::endl;
		unsigned int a = m_map.getEmbedding(c) ;
//        std::cerr << "a =  " << a << std::endl;

		if (a == EMBNULL)
			a = Algo::Topo::setOrbitEmbeddingOnNewCell(m_map, c) ;

		m_markVector->setFalse(a);
	}

	/**
	 * test if cell of dart is marked
	 */
	inline bool isMarked(Cell<CELL> c) const
	{
		assert(m_markVector != NULL);
		unsigned int a = m_map.getEmbedding(c) ;

		if (a == EMBNULL)
			return false ;
		return m_markVector->operator[](a);
	}

	/**
	 * mark the cell
	 */
	inline void mark(unsigned int em)
	{
		assert(m_markVector != NULL);
		m_markVector->setTrue(em);
	}

	/**
	 * unmark the cell
	 */
	inline void unmark(unsigned int em)
	{
		assert(m_markVector != NULL);
		m_markVector->setFalse(em);
	}

	/**
	 * test if cell is marked
	 */
	inline bool isMarked(unsigned int em) const
	{
		assert(m_markVector != NULL);

		if (em == EMBNULL)
			return false ;

		return m_markVector->operator[](em);
	}

	/**
	 * mark all the cells
	 */
	inline void markAll()
	{
		assert(m_markVector != NULL);

		AttributeContainer& cont = m_map.template getAttributeContainer<CELL>() ;
		if (cont.hasBrowser())
			for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
				this->m_markVector->setTrue(i);
		else
			m_markVector->allTrue();
	}

	inline bool isAllUnmarked()
	{
		assert(m_markVector != NULL);

		AttributeContainer& cont = m_map.template getAttributeContainer<CELL>() ;
		if (cont.hasBrowser())
		{
			for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
				if(m_markVector->operator[](i))
				return false ;
			return true ;
		}
		//else
		return m_markVector->isAllFalse();
	}

    inline bool isAllMarked()
    {
        assert(m_markVector != NULL);

        AttributeContainer& cont = m_map.template getAttributeContainer<CELL>() ;
        for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
            if(!this->isMarked(i))
                return false ;
        return true ;
    }
};

/**
 * class that allows the marking of cells
 * \warning no default constructor
 */

template <typename MAP, unsigned int CELL>
class CellMarker : public CellMarkerBase<MAP, CELL>
{
public:
	CellMarker(MAP& map) : CellMarkerBase<MAP, CELL>(map)
	{}

	CellMarker(const MAP& map) :
		CellMarkerBase<MAP, CELL>(map)
	{}

	virtual ~CellMarker()
	{}

protected:
	CellMarker(const CellMarker& cm) :
		CellMarkerBase<MAP, CELL>(cm)
	{}

public:
	inline void unmarkAll()
	{
		assert(this->m_markVector != NULL);

		AttributeContainer& cont = this->m_map.template getAttributeContainer<CELL>() ;
		if (cont.hasBrowser())
			for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
				this->m_markVector->setFalse(i);
		else
			this->m_markVector->allFalse();
	}
};

/**
 * class that allows the marking of cells
 * the marked cells are stored to optimize the unmarking task at destruction
 * \warning no default constructor
 */
template <typename MAP, unsigned int CELL>
class CellMarkerStore: public CellMarkerBase<MAP, CELL>
{
protected:
	std::vector<unsigned int>* m_markedCells ;

public:
	CellMarkerStore(MAP& map) :
		CellMarkerBase<MAP, CELL>(map)
	{
//		m_markedCells.reservef(128);
		m_markedCells = this->m_map.askUIntBuffer();
	}

    CellMarkerStore(const MAP& map) :
        CellMarkerBase<MAP, CELL>(map)
	{
//		m_markedCells.reserve(128);
		m_markedCells = this->m_map.askUIntBuffer();
	}

	virtual ~CellMarkerStore()
	{
//		unmarkAll() ;
		this->m_map.releaseUIntBuffer(m_markedCells);
//		assert(isAllUnmarked);
//		CGoGN_ASSERT(this->isAllUnmarked())
	}

protected:
	CellMarkerStore(const CellMarkerStore& cm) :
		CellMarkerBase<MAP, CELL>(cm)
	{}

public:
	inline void mark(Cell<CELL> d)
	{
		CellMarkerBase<MAP, CELL>::mark(d) ;
//        std::cerr << "CellMarkerStore marking " << CELL << "-cell of index " << d.index() << std::endl;
		m_markedCells->push_back(this->m_map.template getEmbedding<CELL>(d)) ;
//        std::cerr << "its emb is : " << m_markedCells->back() << std::endl;
	}

	inline void mark(unsigned int em)
	{
		CellMarkerBase<MAP, CELL>::mark(em) ;
//        std::cerr << "CellMarkerStore marking " << CELL << "-cell. embedding : " << em << std::endl;
		m_markedCells->push_back(em) ;
	}

    inline void unmark(Cell<CELL> c)
    {
        CellMarkerBase<MAP, CELL>::unmark(c);
        const unsigned cellID = this->m_map.getEmbedding(c);
        for (std::vector<unsigned int>::iterator it = m_markedCells->begin(), end = m_markedCells->end() ; it != end ; ++it)
        {
            if (*it == cellID)
            {
                std::swap(*it, m_markedCells->back());
                m_markedCells->pop_back();
                break;
            }
        }
    }

	inline void unmarkAll()
	{
		assert(this->m_markVector != NULL);

		for (std::vector<unsigned int>::iterator it = m_markedCells->begin(); it != m_markedCells->end(); ++it)
			this->m_markVector->setFalse(*it);
	}
};

/**
 * class that allows the marking of Darts
 * the marked Darts are stored to optimize the unmarking task at destruction
 * \warning no default constructor
 */
template <typename MAP, unsigned int CELL>
class CellMarkerMemo: public CellMarkerBase<MAP, CELL>
{
protected:
    std::vector<Dart> m_markedDarts ;

public:
	CellMarkerMemo(MAP& map) :
		CellMarkerBase<MAP, CELL>(map)
	{
        m_markedDarts.reserve(128);
	}

	CellMarkerMemo(const MAP& map) :
		CellMarkerBase<MAP, CELL>(map)
	{
        m_markedDarts.reserve(128);
	}

	virtual ~CellMarkerMemo()
	{
//		unmarkAll() ;
//		assert(isAllUnmarked);
//		CGoGN_ASSERT(this->isAllUnmarked())
	}

protected:
	CellMarkerMemo(const CellMarkerMemo& cm) :
		CellMarkerBase<MAP, CELL>(cm)
	{}

public:
	inline void mark(Cell<CELL> c)
	{
		if(!this->isMarked(c))
		{
//            std::cerr << "marking dart " << c << " currLvl " << m_map.getCurrentLevel() << std::endl;
			CellMarkerBase<MAP, CELL>::mark(c) ;
            m_markedDarts.push_back(c.dart) ;
		}
	}

    inline void unmark(Cell<CELL> c)
    {
        CellMarkerBase<MAP, CELL>::unmark(c);
        const unsigned cellID = this->m_map.getEmbedding(c);
        for (std::vector<Dart>::iterator dit = m_markedDarts.begin(), dend = m_markedDarts.end() ; dit != dend ; ++dit)
        {
            if (this->m_map.getEmbedding(Cell<CELL>(*dit)) == cellID)
            {
                std::swap(*dit, m_markedDarts.back());
                m_markedDarts.pop_back();
                break;
            }
        }
    }

	inline void unmarkAll()
	{
		assert(this->m_markVector != NULL);
		for (std::vector<Dart>::iterator it = m_markedDarts.begin(); it != m_markedDarts.end(); ++it)
		{
            this->CellMarkerBase<MAP, CELL>::unmark(*it) ;
		}
		m_markedDarts.clear();

	}

	inline const std::vector<Dart>& get_markedCells()
	{
		return m_markedDarts;
	}
};

/**
 * class that allows the marking of cells
 * the markers are not unmarked at destruction
 * \warning no default constructor
 */
template <typename MAP, unsigned int CELL>
class CellMarkerNoUnmark: public CellMarkerBase<MAP, CELL>
{
#ifndef NDEBUG
	int m_counter;
#endif
public:
	CellMarkerNoUnmark(MAP& map) :
		CellMarkerBase<MAP, CELL>(map)
  #ifndef NDEBUG
		,m_counter(0)
  #endif
	{}

	CellMarkerNoUnmark(const MAP& map) :
		CellMarkerBase<MAP, CELL>(map)
  #ifndef NDEBUG
		,m_counter(0)
  #endif
	{}

	virtual ~CellMarkerNoUnmark()
	{
#ifndef NDEBUG
		if (m_counter != 0)
		{
			CGoGNerr << "CellMarkerNoUnmark: Warning problem unmarking not complete"<< CGoGNendl;
			CGoGNerr << "CellMarkerNoUnmark:  -> calling unmarkAll()"<< CGoGNendl;
			unmarkAll();
		}
#endif
	}

protected:
	CellMarkerNoUnmark(const CellMarkerNoUnmark& cm) :
		CellMarkerBase<MAP, CELL>(cm)
  #ifndef NDEBUG
		,m_counter(cm.m_counter)
  #endif
	{}

public:
	inline void unmarkAll()
	{
		assert(this->m_markVector != NULL);

		AttributeContainer& cont = this->m_map.template getAttributeContainer<CELL>() ;
		if (cont.hasBrowser())
			for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
				this->m_markVector->setFalse(i);
		else
			this->m_markVector->allFalse();
	}


#ifndef NDEBUG
	inline void mark(Cell<CELL> c)
	{
		if (this->isMarked(c))
			return;
		CellMarkerBase<MAP, CELL>::mark(c) ;
		m_counter++;
	}

	inline void unmark(Cell<CELL> c)
	{
		if (!this->isMarked(c))
			return;
		CellMarkerBase<MAP, CELL>::unmark(c) ;
		m_counter--;
	}

	inline void mark(unsigned int i)
	{
		if (this->isMarked(i))
			return;
		CellMarkerBase<MAP, CELL>::mark(i) ;
		m_counter++;
	}

	/**
	 * unmark the dart
	 */
	inline void unmark(unsigned int i)
	{
		if (!this->isMarked(i))
			return;
		CellMarkerBase<MAP, CELL>::unmark(i) ;
		m_counter--;
	}

#endif
};

// Selector and count functors testing for marker existence
/********************************************************/

/**
 * selector that say if a dart has its cell marked
 */
template <typename MAP, unsigned int CELL>
class SelectorCellMarked : public FunctorSelect
{
protected:
	const CellMarkerBase<MAP, CELL>& m_cmarker ;

public:
	SelectorCellMarked(const CellMarkerBase<MAP, CELL>& cm) :
		m_cmarker(cm)
	{}

	inline bool operator()(Cell<CELL> d) const
	{
		if (m_cmarker.isMarked(d))
			return true ;
		return false ;
	}

	inline FunctorSelect* copy() const { return new SelectorCellMarked(m_cmarker); }
};

template <typename MAP, unsigned int CELL>
class SelectorCellUnmarked : public FunctorSelect
{
protected:
	const CellMarkerBase<MAP, CELL>& m_cmarker ;

public:
	SelectorCellUnmarked(const CellMarkerBase<MAP, CELL>& cm) :
		m_cmarker(cm)
	{}

	inline bool operator()(Cell<CELL> d) const
	{
		if (!m_cmarker.isMarked(d))
			return true ;
		return false ;
	}

	inline FunctorSelect* copy() const
	{
		return new SelectorCellUnmarked(m_cmarker);
	}
};

// Functor version (needed for use with foreach_xxx)

template <typename MAP, unsigned int CELL>
class FunctorCellIsMarked : public FunctorType
{
protected:
	CellMarkerBase<MAP, CELL>& m_marker;

public:
	FunctorCellIsMarked(CellMarkerBase<MAP, CELL>& cm) :
		m_marker(cm)
	{}

	inline bool operator()(Cell<CELL> d)
	{
		return m_marker.isMarked(d);
	}
};

template <typename MAP, unsigned int CELL>
class FunctorCellIsUnmarked : public FunctorType
{
protected:
	CellMarkerBase<MAP, CELL>& m_marker;
public:
	FunctorCellIsUnmarked(CellMarkerBase<MAP, CELL>& cm) :
		m_marker(cm)
	{}

	inline bool operator()(Cell<CELL> d)
	{
		return !m_marker.isMarked(d);
	}
};

} // namespace CGoGN

#endif
