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

#ifndef __MAP_MONO__
#define __MAP_MONO__

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

class MapMono : public GenericMap
{
    friend class ::sofa::cgogn_plugin::test::CGoGN_test;
	template<typename MAP> friend class DartMarkerTmpl ;
	template<typename MAP> friend class DartMarkerStore ;

public:
	MapMono()
	{}

    virtual void clear(bool removeAttrib);

protected:
	// protected copy constructor to prevent the copy of map
	MapMono(const MapMono& m): GenericMap(m){}

	std::vector<AttributeMultiVector<Dart>*> m_involution;
	std::vector<AttributeMultiVector<Dart>*> m_permutation;
	std::vector<AttributeMultiVector<Dart>*> m_permutation_inv;

	/****************************************
	 *          DARTS MANAGEMENT            *
	 ****************************************/

	inline Dart newDart();

	inline virtual void deleteDart(Dart d);

public:
	inline unsigned int dartIndex(Dart d) const;

//    template<unsigned ORB>
//    inline unsigned int dartIndex(Cell<ORB> c) const {
//        return c.index();
//    }

	inline Dart indexDart(unsigned int index) const;

	inline unsigned int getNbDarts() const;

	inline AttributeContainer& getDartContainer();

	/****************************************
	 *        RELATIONS MANAGEMENT          *
	 ****************************************/

protected:
	inline void addInvolution();
	inline void addPermutation();
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

	inline virtual void compactTopo();

	/****************************************
	 *           DARTS TRAVERSALS           *
	 ****************************************/
public:
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
    inline void next(Cell<ORBIT>& c) const { m_attribs[DART].next(c.dart.index) ;}

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

#include "Topology/generic/mapImpl/mapMono.hpp"

#endif
