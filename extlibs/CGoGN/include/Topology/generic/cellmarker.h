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
	GenericMap& m_map ;
	Mark m_mark ;
	AttributeMultiVector<Mark>* m_markVector ;
	unsigned int m_thread ;
	unsigned int m_cell ;
	bool releaseOnDestruct ;

public:
	CellMarkerGen(GenericMap& map, unsigned int cell, unsigned int thread = 0) :
		m_map(map),
		m_thread(thread),
		m_cell(cell),
		releaseOnDestruct(true)
	{}

	~CellMarkerGen()
	{}

	unsigned int getThread() { return m_thread ; }
	unsigned int getCell() { return m_cell ; }

	void updateMarkVector(AttributeMultiVector<Mark>* amv) { m_markVector = amv ; }

	/**
	 * set if the mark has to be release on destruction or not
	 */
	void setReleaseOnDestruct(bool b) { releaseOnDestruct = b ; }

//	virtual void mark(Dart d) = 0 ;
//	virtual void unmark(Dart d) = 0 ;
//	virtual bool isMarked(Dart d) const = 0 ;
//	virtual void mark(unsigned int em) = 0 ;
//	virtual void unmark(unsigned int em) = 0 ;
//	virtual bool isMarked(unsigned int em) const = 0 ;
//	virtual void markAll() = 0 ;
//	virtual void unmarkAll() = 0 ;
//	virtual bool isAllUnmarked() = 0 ;
};

/**
 * generic class that allows the marking of cells
 * \warning no default constructor
 */
template <unsigned int CELL>
class CellMarkerBase : public CellMarkerGen
{
public:
	/**
	 * constructor
	 * @param map the map on which we work
	 */
	CellMarkerBase(GenericMap& map, unsigned int thread = 0) : CellMarkerGen(map, CELL, thread)
	{
		if(!map.isOrbitEmbedded<CELL>())
			map.addEmbedding<CELL>() ;
		m_mark = m_map.getMarkerSet<CELL>(m_thread).getNewMark() ;
		m_markVector = m_map.getMarkVector<CELL>(m_thread) ;
		m_map.cellMarkers[m_thread].push_back(this) ;
	}

	/*virtual */~CellMarkerBase()
	{
		if(releaseOnDestruct)
		{
			m_map.getMarkerSet<CELL>(m_thread).releaseMark(m_mark) ;

			std::vector<CellMarkerGen*>& cmg = m_map.cellMarkers[m_thread];
			for(std::vector<CellMarkerGen*>::iterator it = cmg.begin(); it != cmg.end(); ++it)
			{
				if(*it == this)
				{
					*it = cmg.back();
					cmg.pop_back();
					return;
				}
			}
		}
	}

protected:
	// protected copy constructor to forbid its usage
	CellMarkerBase(const CellMarkerGen& cm) : CellMarkerGen(cm.m_map, CELL)
	{}

public:
	/**
	 * mark the cell of dart
	 */
	inline void mark(Dart d)
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		unsigned int a = m_map.getEmbedding<CELL>(d) ;
		if (a == EMBNULL)
			a = m_map.setOrbitEmbeddingOnNewCell<CELL>(d) ;
		m_markVector->operator[](a).setMark(m_mark) ;
	}

	/**
	 * unmark the cell of dart
	 */
	inline void unmark(Dart d)
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		unsigned int a = m_map.getEmbedding<CELL>(d) ;
		if (a == EMBNULL)
			a = m_map.setOrbitEmbeddingOnNewCell<CELL>(d) ;
		m_markVector->operator[](a).unsetMark(m_mark) ;
	}

	/**
	 * test if cell of dart is marked
	 */
	inline bool isMarked(Dart d) const
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		unsigned int a = m_map.getEmbedding<CELL>(d) ;
		if (a == EMBNULL)
			return false ;
		return m_markVector->operator[](a).testMark(m_mark) ;
	}

	/**
	 * mark the cell
	 */
	inline void mark(unsigned int em)
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		m_markVector->operator[](em).setMark(m_mark) ;
	}

	/**
	 * unmark the cell
	 */
	inline void unmark(unsigned int em)
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		m_markVector->operator[](em).unsetMark(m_mark) ;
	}

	/**
	 * test if cell is marked
	 */
	inline bool isMarked(unsigned int em) const
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		if (em == EMBNULL)
			return false ;
		return m_markVector->operator[](em).testMark(m_mark) ;
	}

	/**
	 * mark all the cells
	 */
	inline void markAll()
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		AttributeContainer& cont = m_map.getAttributeContainer<CELL>() ;
		for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
			m_markVector->operator[](i).setMark(m_mark) ;
	}

	inline bool isAllUnmarked()
	{
		assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
		assert(m_markVector != NULL);

		AttributeContainer& cont = m_map.getAttributeContainer<CELL>() ;
		for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
			if(m_markVector->operator[](i).testMark(m_mark))
				return false ;
		return true ;
	}
};

/**
 * class that allows the marking of cells
 * \warning no default constructor
 */
template <unsigned int CELL>
class CellMarker : public CellMarkerBase<CELL>
{
public:
	CellMarker(GenericMap& map, unsigned int thread = 0) : CellMarkerBase<CELL>(map, thread)
	{}

	~CellMarker()
	{
		unmarkAll() ;
	}

protected:
	CellMarker(const CellMarker& cm) : CellMarkerBase<CELL>(cm)
	{}

public:
	inline void unmarkAll()
	{
		assert(this->m_map.template getMarkerSet<CELL>(this->m_thread).testMark(this->m_mark));
		assert(this->m_markVector != NULL);

		AttributeContainer& cont = this->m_map.template getAttributeContainer<CELL>() ;
		for (unsigned int i = cont.realBegin(); i != cont.realEnd(); cont.realNext(i))
			this->m_markVector->operator[](i).unsetMark(this->m_mark) ;
	}
};

/**
 * class that allows the marking of cells
 * the marked cells are stored to optimize the unmarking task at destruction
 * \warning no default constructor
 */
template <unsigned int CELL>
class CellMarkerStore: public CellMarkerBase<CELL>
{
protected:
	std::vector<unsigned int> m_markedCells ;

public:
	CellMarkerStore(GenericMap& map, unsigned int thread = 0) : CellMarkerBase<CELL>(map, thread)
	{}

	~CellMarkerStore()
	{
		unmarkAll() ;
//		assert(isAllUnmarked);
//		CGoGN_ASSERT(this->isAllUnmarked())
	}

protected:
	CellMarkerStore(const CellMarkerStore& cm) : CellMarkerBase<CELL>(cm)
	{}

public:
	inline void mark(Dart d)
	{
		CellMarkerBase<CELL>::mark(d) ;
		m_markedCells.push_back(this->m_map.template getEmbedding<CELL>(d)) ;
	}

	inline void mark(unsigned int em)
	{
		CellMarkerBase<CELL>::mark(em) ;
		m_markedCells.push_back(em) ;
	}

	inline void unmarkAll()
	{
		assert(this->m_map.template getMarkerSet<CELL>(this->m_thread).testMark(this->m_mark));
		assert(this->m_markVector != NULL);

		for (std::vector<unsigned int>::iterator it = m_markedCells.begin(); it != m_markedCells.end(); ++it)
			this->m_markVector->operator[](*it).unsetMark(this->m_mark) ;
	}
};
/**
 * class that allows the marking of Darts
 * the marked Darts are stored to optimize the unmarking task at destruction
 * \warning no default constructor
 */
template <unsigned int CELL>
class CellMarkerMemo: public CellMarkerBase<CELL>
{
protected:
	std::vector<Dart> m_markedDarts ;

public:
	CellMarkerMemo(GenericMap& map, unsigned int thread = 0) : CellMarkerBase<CELL>(map, thread)
	{}

	~CellMarkerMemo()
	{
		unmarkAll() ;
//		assert(isAllUnmarked);
//		CGoGN_ASSERT(this->isAllUnmarked())
	}

protected:
	CellMarkerMemo(const CellMarkerMemo& cm) : CellMarkerBase<CELL>(cm)
	{}

public:
	inline void mark(Dart d)
	{
		if(!this->isMarked(d))
		{
			CellMarkerBase<CELL>::mark(d) ;
			m_markedDarts.push_back(d) ;
		}
	}

	inline void unmarkAll()
	{
		assert(this->m_map.template getMarkerSet<CELL>(this->m_thread).testMark(this->m_mark));
		assert(this->m_markVector != NULL);
		for (std::vector<Dart>::iterator it = m_markedDarts.begin(); it != m_markedDarts.end(); ++it)
		{
			this->unmark(*it) ;
		}
		m_markedDarts.clear();

	}

	inline std::vector<Dart> get_markedCells()
	{
		return m_markedDarts;
	}
};
/**
 * class that allows the marking of cells
 * the markers are not unmarked at destruction
 * \warning no default constructor
 */
template <unsigned int CELL>
class CellMarkerNoUnmark: public CellMarkerBase<CELL>
{
public:
	CellMarkerNoUnmark(GenericMap& map, unsigned int thread = 0) : CellMarkerBase<CELL>(map, thread)
	{}

	~CellMarkerNoUnmark()
	{
//		assert(isAllUnmarked()) ;
//		CGoGN_ASSERT(this->isAllUnmarked())
	}

protected:
	CellMarkerNoUnmark(const CellMarkerNoUnmark& cm) : CellMarkerBase<CELL>(cm)
	{}

public:
	inline void unmarkAll()
	{
		assert(this->m_map.template getMarkerSet<CELL>(this->m_thread).testMark(this->m_mark));
		assert(this->m_markVector != NULL);

		AttributeContainer& cont = this->m_map.template getAttributeContainer<CELL>() ;
		for (unsigned int i = cont.realBegin(); i != cont.realEnd(); cont.realNext(i))
			this->m_markVector->operator[](i).unsetMark(this->m_mark) ;
	}
};


/**
 * selector that say if a dart has its cell marked
 */
template <unsigned int CELL>
class SelectorCellMarked : public FunctorSelect
{
protected:
	const CellMarkerBase<CELL>& m_cmarker ;
public:
	SelectorCellMarked(const CellMarkerBase<CELL>& cm) : m_cmarker(cm) {}
	inline bool operator()(Dart d) const
	{
		if (m_cmarker.isMarked(d))
			return true ;
		return false ;
	}
	inline FunctorSelect* copy() const { return new SelectorCellMarked(m_cmarker); }
};

template <unsigned int CELL>
class SelectorCellUnmarked : public FunctorSelect
{
protected:
	const CellMarkerBase<CELL>& m_cmarker ;
public:
	SelectorCellUnmarked(const CellMarkerBase<CELL>& cm) : m_cmarker(cm) {}
	inline bool operator()(Dart d) const
	{
		if (!m_cmarker.isMarked(d))
			return true ;
		return false ;
	}
	inline FunctorSelect* copy() const { return new SelectorCellUnmarked(m_cmarker); }
};

// Functor version (needed for use with foreach_xxx)

template <unsigned int CELL>
class FunctorCellIsMarked : public FunctorType
{
protected:
	CellMarkerBase<CELL>& m_marker;
public:
	FunctorCellIsMarked(CellMarkerBase<CELL>& cm) : m_marker(cm) {}
	inline bool operator()(Dart d)
	{
		return m_marker.isMarked(d);
	}
};

template <unsigned int CELL>
class FunctorCellIsUnmarked : public FunctorType
{
protected:
	CellMarkerBase<CELL>& m_marker;
public:
	FunctorCellIsUnmarked(CellMarkerBase<CELL>& cm) : m_marker(cm) {}
	inline bool operator()(Dart d)
	{
		return !m_marker.isMarked(d);
	}
};

} // namespace CGoGN

#endif
