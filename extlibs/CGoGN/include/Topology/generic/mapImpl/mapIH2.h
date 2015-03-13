#ifndef MAPIH2_H
#define MAPIH2_H

#include "Topology/generic/genericmap.h"


namespace sofa {
namespace cgogn_plugin {
namespace test {
class CGoGN_test ;
}
}
}


namespace CGoGN
{

class MapIH2 : public GenericMap
{
    friend class ::sofa::cgogn_plugin::test::CGoGN_test;
    template<typename MAP> friend class DartMarkerTmpl ;
    template<typename MAP> friend class DartMarkerStore ;

public:
    MapIH2()
    {}

    virtual void clear(bool removeAttrib);

protected:
    MapIH2(const MapIH2& m): GenericMap(m){}

    std::vector<AttributeMultiVector<Dart>*> m_involution;
    std::vector<AttributeMultiVector<Dart>*> m_permutation;
    std::vector<AttributeMultiVector<Dart>*> m_permutation_inv;

    /****************************************
         *          DARTS MANAGEMENT            *
         ****************************************/

    Dart newDart();
    inline void deleteDart(Dart d);
public:
    inline unsigned int dartIndex(Dart d) const;
    inline Dart indexDart(unsigned int index) const;
    inline unsigned int getNbDarts() const;
    inline AttributeContainer& getDartContainer();

    /****************************************
         *        RELATIONS MANAGEMENT          *
         ****************************************/

protected:
    void addInvolution();
    void addPermutation();
    inline void removeLastInvolutionPtr(); // for moveFrom

    inline AttributeMultiVector<Dart>* getInvolutionAttribute(unsigned int i);
    inline AttributeMultiVector<Dart>* getPermutationAttribute(unsigned int i);
    inline AttributeMultiVector<Dart>* getPermutationInvAttribute(unsigned int i);

    virtual unsigned int getNbInvolutions() const = 0;
    virtual unsigned int getNbPermutations() const = 0;

    template <int I>
    inline Dart getInvolution(Dart d) const;

    template <int I>
    inline Dart getPermutation(Dart d) const;

    template <int I>
    inline Dart getPermutationInv(Dart d) const;

    template <int I>
    inline void involutionSew(Dart d, Dart e);

    template <int I>
    inline void involutionUnsew(Dart d);

    template <int I>
    inline void permutationSew(Dart d, Dart e);

    template <int I>
    inline void permutationUnsew(Dart d);


    void compactTopo();

    /****************************************
         *           DARTS TRAVERSALS           *
         ****************************************/
public:
    inline Dart begin() const;
    inline Dart end() const;
    inline void next(Dart& d) const;

    template<unsigned ORBIT>
    inline void next(Cell<ORBIT>& c) const;

    /**
    * Apply a functor on each dart of the map
    * @param f a callable taking a Dart parameter
    **/
    template <typename FUNC>
    void foreach_dart(const FUNC& f) ;

    /****************************************
    *             SAVE & LOAD              *
    ****************************************/
    bool saveMapBin(const std::string& filename) const;
    bool loadMapBin(const std::string& filename);
    bool copyFrom(const GenericMap& map);
    void restore_topo_shortcuts();

    /***************************************************
     *              LEVELS MANAGEMENT                  *
     ***************************************************/

    inline unsigned int getCurrentLevel();
    inline void setCurrentLevel(unsigned int l) ;
    inline void incCurrentLevel();
    inline void decCurrentLevel();
    inline unsigned int getMaxLevel() ;
    inline unsigned int getDartLevel(Dart d) ;
    inline void setDartLevel(Dart d, unsigned int i) ;
    inline void setMaxLevel(unsigned int l);



protected:
    unsigned int m_curLevel ;
    unsigned int m_maxLevel ;
    unsigned int m_idCount ;
    AttributeMultiVector<unsigned int>* m_nextLevelCell[NB_ORBITS] ;
    AttributeMultiVector<unsigned int>* m_dartLevel ;

};


template <typename FUNC>
void MapIH2::foreach_dart(const FUNC& f)
{
    for (Dart d = begin(); d != end(); next(d))
        f(d);
}


template<unsigned ORBIT>
void MapIH2::next(Cell<ORBIT> &c) const
{
    m_attribs[DART].next(c.dart.index);
}

template <int I>
void MapIH2::permutationUnsew(Dart d)
{
    const Dart e = (*m_permutation[I])[d.index] ;
    const Dart f = (*m_permutation[I])[e.index] ;
    (*m_permutation[I])[d.index] = f ;
    (*m_permutation[I])[e.index] = e ;
    (*m_permutation_inv[I])[f.index] = d ;
    (*m_permutation_inv[I])[e.index] = e ;
}

template <int I>
void MapIH2::permutationSew(Dart d, Dart e)
{
    const Dart f = (*m_permutation[I])[d.index] ;
    const Dart g = (*m_permutation[I])[e.index] ;
    (*m_permutation[I])[d.index] = g ;
    (*m_permutation[I])[e.index] = f ;
    (*m_permutation_inv[I])[g.index] = d ;
    (*m_permutation_inv[I])[f.index] = e ;
}

template <int I>
void MapIH2::involutionUnsew(Dart d)
{
    const Dart e = (*m_involution[I])[d.index] ;
    (*m_involution[I])[d.index] = d ;
    (*m_involution[I])[e.index] = e ;
}

template <int I>
void MapIH2::involutionSew(Dart d, Dart e)
{
    assert((*m_involution[I])[d.index] == d) ;
    assert((*m_involution[I])[e.index] == e) ;
    (*m_involution[I])[d.index] = e ;
    (*m_involution[I])[e.index] = d ;
}

template <int I>
Dart MapIH2::getPermutationInv(Dart d) const
{
    return (*m_permutation_inv[I])[d.index];
}

template <int I>
Dart MapIH2::getPermutation(Dart d) const
{
    return (*m_permutation[I])[d.index];
}

template <int I>
Dart MapIH2::getInvolution(Dart d) const
{
    return (*m_involution[I])[d.index];
}

void MapIH2::deleteDart(Dart d)
{
    deleteDartLine(d.index) ;
}

unsigned int MapIH2::dartIndex(Dart d) const
{
    return d.index;
}

Dart MapIH2::indexDart(unsigned int index) const
{
    return Dart(index);
}

unsigned int MapIH2::getNbDarts() const
{
    return m_attribs[DART].size() ;
}

AttributeContainer &MapIH2::getDartContainer()
{
    return m_attribs[DART];
}

AttributeMultiVector<Dart> *MapIH2::getInvolutionAttribute(unsigned int i)
{
    assert(i < m_involution.size());
    return m_involution[i];
}

AttributeMultiVector<Dart> *MapIH2::getPermutationAttribute(unsigned int i)
{
    assert(i < m_permutation.size());
    return m_permutation[i];
}

AttributeMultiVector<Dart> *MapIH2::getPermutationInvAttribute(unsigned int i)
{
    assert(i < m_permutation_inv.size());
    return m_permutation_inv[i];
}

Dart MapIH2::begin() const
{
    return Dart::create(m_attribs[DART].begin()) ;
}

Dart MapIH2::end() const
{
    return Dart::create(m_attribs[DART].end()) ;
}

void MapIH2::next(Dart &d) const
{
    m_attribs[DART].next(d.index) ;
    if((*m_dartLevel)[d.index] > m_curLevel)
        d = this->end() ;
}

unsigned int MapIH2::getCurrentLevel()
{
    return m_curLevel ;
}

void MapIH2::setCurrentLevel(unsigned int l)
{
    m_curLevel = l ;
}

void MapIH2::incCurrentLevel()
{
    if(m_curLevel < m_maxLevel)
        ++m_curLevel ;
    else
        CGoGNout << "incCurrentLevel : already at maximum resolution level" << CGoGNendl ;
}

void MapIH2::decCurrentLevel()
{
    if(m_curLevel > 0)
        --m_curLevel ;
    else
        CGoGNout << "decCurrentLevel : already at minimum resolution level" << CGoGNendl ;
}

unsigned int MapIH2::getMaxLevel()
{
    return m_maxLevel ;
}

unsigned int MapIH2::getDartLevel(Dart d)
{
    return (*m_dartLevel)[d.index] ;
}

void MapIH2::setDartLevel(Dart d, unsigned int i)
{
    (*m_dartLevel)[d.index] = i ;
}

void MapIH2::setMaxLevel(unsigned int l)
{
    m_maxLevel = l;
}





} //namespace CGoGN

#include "Topology/generic/mapImpl/mapIH2.hpp"

#endif // MAPIH2_H

