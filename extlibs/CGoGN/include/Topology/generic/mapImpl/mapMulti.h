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

#ifndef __MAP_MULTI__
#define __MAP_MULTI__

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

class MapMulti : public GenericMap
{
    friend class ::sofa::cgogn_plugin::test::CGoGN_test;
    template<typename MAP> friend class DartMarkerTmpl ;
    template<typename MAP> friend class DartMarkerStore ;

public:
    MapMulti()
    {
        initMR();
    }

    inline virtual void clear(bool removeAttrib);

protected:
    // protected copy constructor to prevent the copy of map
    MapMulti(const MapMulti& m): GenericMap(m) {}

    std::vector<AttributeMultiVector<Dart>*> m_involution;
    std::vector<AttributeMultiVector<Dart>*> m_permutation;
    std::vector<AttributeMultiVector<Dart>*> m_permutation_inv;

    /**
     * container for multiresolution darts
     */
    AttributeContainer m_mrattribs ;

    /**
     * pointers to attributes of m_mrattribs that store indices of m_attribs[DART]
     * (one for each level)
     */
    std::vector< AttributeMultiVector<unsigned int>* > m_mrDarts ;

    /**
     * pointer to attribute of m_mrattribs that stores darts insertion levels
     */
    AttributeMultiVector<unsigned int>* m_mrLevels ;

    /**
     * vector that stores the number of darts inserted on each resolution level
     */
    std::vector<unsigned int> m_mrNbDarts ;

    /**
     * current level in multiresolution map
     */
    unsigned int m_mrCurrentLevel ;

    /**
     * stack for current level temporary storage
     */
    std::vector<unsigned int> m_mrLevelStack ;

    /****************************************
     *          DARTS MANAGEMENT            *
     ****************************************/

    inline Dart newDart();

    inline virtual void deleteDart(Dart d);

public:
    inline unsigned int dartIndex(Dart d) const;

//    template<unsigned ORB>
//    inline unsigned int dartIndex(Cell<ORB> c) const {
//        return (*m_mrDarts[m_mrCurrentLevel])[c.index()];
//    }

    inline Dart indexDart(unsigned int index) const;

    /**
     * get the number of darts inserted in the given leveldart
     */
    inline unsigned int getNbInsertedDarts(unsigned int level) const;

    /**
     * get the number of darts that define the map of the given leveldart
     */
    inline unsigned int getNbDarts(unsigned int level) const;

    /**
     * @return the number of darts in the map
     */
    inline unsigned int getNbDarts() const;

    inline AttributeContainer& getDartContainer();

    /**
     * get the insertion level of a dart
     */
    inline unsigned int getDartLevel(Dart d) const;

protected:
    /**
     *
     */
    inline void incDartLevel(Dart d) const ;

    /**
     * duplicate a dart starting from current level
     */
    inline void duplicateDart(Dart d) ;

    inline void duplicateDartAtOneLevel(Dart d, unsigned int level) ;

    /****************************************
     *        RELATIONS MANAGEMENT          *
     ****************************************/

    inline void addInvolution();
    inline void addPermutation();

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

    inline virtual void compactTopo();

    /****************************************
     *      MR CONTAINER MANAGEMENT         *
     ****************************************/
public:
    /**
     * get the MR attribute container
     */
    AttributeContainer& getMRAttributeContainer() ;

    /**
     * get the MR attribute container
     */
    AttributeMultiVector<unsigned int>* getMRDartAttributeVector(unsigned int level) ;
    AttributeMultiVector<unsigned int>* getMRLevelAttributeVector();

    /****************************************
     *     RESOLUTION LEVELS MANAGEMENT     *
     ****************************************/

    void printMR() ;

    /**
     * initialize the multiresolution attribute container
     */
    void initMR() ;

    /**
     * get the current resolution level
     */
    unsigned int getCurrentLevel() ;

    /**
     * set the current resolution level
     */
    void setCurrentLevel(unsigned int l) ;

    /**
     * increment the current resolution level
     */
    void incCurrentLevel() ;

    /**
     * decrement the current resolution level
     */
    void decCurrentLevel() ;

    /**
     * store current resolution level on a stack
     */
    void pushLevel() ;

    /**
     * set as current the resolution level of the top of the stack
     */
    void popLevel() ;

    /**
     * get the maximum resolution level
     */
    unsigned int getMaxLevel() ;

    /**
     * add a resolution level in the back of the level table
     */
    void addLevelBack() ;

    /**
     * add a resolution level in the front of the level table
     */
    void addLevelFront();

    /**
     * remove last resolution level
     */
    void removeLevelBack() ;

    /**
     * remove first resolution level
     */
    void removeLevelFront();

    /**
     * copy MRDarts from level-1 to level
     */
    void copyLevel(unsigned int level);

    /**
     * duplicate darts from level-1 to level
     */
    void duplicateDarts(unsigned int newlevel);

    /****************************************
     *           DARTS TRAVERSALS           *
     ****************************************/

    /**
     * Begin of map
     * @return the first dart of the map
     */
    inline Dart begin() const;

    /**
     * End of map
     * @return the end iterator (next of last) of the map
     */
    inline Dart end() const;

    /**
     * allow to go from a dart to the next
     * in the order of storage
     * @param d reference to the dart to be modified
     */
    inline void next(Dart& d) const;
    template<unsigned ORBIT>
    inline void next(Cell<ORBIT>& c) const { 	m_mrattribs.next(c.dart.index);
                                                if(getDartLevel(c.dart) > m_mrCurrentLevel)
                                                    c.dart.index = m_mrattribs.end(); }

    /**
     * Apply a functor on each dart of the map
     * @param f a callable taking a Dart parameter
     */
    template <typename FUNC>
    void foreach_dart(FUNC f) ;

    template <typename FUNC>
    void foreach_dart(FUNC& f) ;

    /****************************************
     *             SAVE & LOAD              *
     ****************************************/

    bool saveMapBin(const std::string& filename) const;

    bool loadMapBin(const std::string& filename);

    bool copyFrom(const GenericMap& map);

    void restore_topo_shortcuts();
} ;

} //namespace CGoGN

#include "Topology/generic/mapImpl/mapMulti.hpp"

#endif
