#ifndef CELLMARKER_HPP
#define CELLMARKER_HPP
#include "cellmarker.h"
#include "genericmap.h"

namespace CGoGN {

template <unsigned int CELL>
CellMarkerBase<CELL>::CellMarkerBase(GenericMap& map, unsigned int thread ) : CellMarkerGen(map, CELL, thread)
{
    if(!m_map.isOrbitEmbedded<CELL>())
        m_map.addEmbedding<CELL>() ;
    m_mark = m_map.getMarkerSet<CELL>(m_thread).getNewMark() ;
    m_markVector = m_map.getMarkVector<CELL>(m_thread) ;
    m_map.cellMarkers[m_thread].push_back(this) ;
}



template <unsigned int CELL>
CellMarkerBase<CELL>::CellMarkerBase(const GenericMap& map, unsigned int thread ) :
    CellMarkerGen(const_cast<GenericMap&>(map), CELL, thread)
{
    if(!m_map.isOrbitEmbedded<CELL>())
        m_map.addEmbedding<CELL>() ;
    m_mark = m_map.getMarkerSet<CELL>(m_thread).getNewMark() ;
    m_markVector = m_map.getMarkVector<CELL>(m_thread) ;
    m_map.cellMarkers[m_thread].push_back(this) ;
}

template <unsigned int CELL>
CellMarkerBase<CELL>::~CellMarkerBase()
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

template <unsigned int CELL>
void CellMarkerBase<CELL>::mark(Dart d)
{
    assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
    assert(m_markVector != NULL);

    unsigned int a = m_map.getEmbedding<CELL>(d) ;
    if (a == EMBNULL)
        a = m_map.setOrbitEmbeddingOnNewCell<CELL>(d) ;
    m_markVector->operator[](a).setMark(m_mark) ;
}

template <unsigned int CELL>
void CellMarkerBase<CELL>::unmark(Dart d)
{
    assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
    assert(m_markVector != NULL);

    unsigned int a = m_map.getEmbedding<CELL>(d) ;
    if (a == EMBNULL)
        a = m_map.setOrbitEmbeddingOnNewCell<CELL>(d) ;
    m_markVector->operator[](a).unsetMark(m_mark) ;
}

template <unsigned int CELL>
bool CellMarkerBase<CELL>::isMarked(Dart d) const
{
    assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
    assert(m_markVector != NULL);

    unsigned int a = m_map.getEmbedding<CELL>(d) ;
    if (a == EMBNULL)
        return false ;
    return m_markVector->operator[](a).testMark(m_mark) ;
}


template <unsigned int CELL>
void CellMarkerBase<CELL>::mark(unsigned int em)
{
    assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
    assert(m_markVector != NULL);

    m_markVector->operator[](em).setMark(m_mark) ;
}


template <unsigned int CELL>
void CellMarkerBase<CELL>::unmark(unsigned int em)
    {
        assert(m_map.getMarkerSet<CELL>(m_thread).testMark(m_mark));
        assert(m_markVector != NULL);

        m_markVector->operator[](em).unsetMark(m_mark) ;
    }








}


#endif // CELLMARKER_HPP
