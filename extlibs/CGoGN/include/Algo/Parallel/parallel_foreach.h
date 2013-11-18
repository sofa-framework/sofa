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


#ifndef __PARALLEL_FOREACH__
#define __PARALLEL_FOREACH__

#include "Topology/generic/functor.h"

namespace CGoGN
{

namespace Algo
{

namespace Parallel
{

static unsigned int NBCORES=0;

/// enum for optimalNbThreads parameter
enum NbParam {NB_HIGHMEMORY, NB_HIGHCOMPUTE, NB_VERYHIGHMEMORY};

/// size of buffers to store darts or indexes in each threads
const unsigned int SIZE_BUFFER_THREAD = 8192;	// seems to be the best compromise

/**
 * @return How much threads has you computer
 */
inline unsigned int nbThreads();

/**
 * @param p can be NB_HIGHMEMORY (default) or NB_HIGHCOMPUTE or NB_VERYHIGHMEMORY
 * @return Number of core in fact (work only with quad core with/without hyper threading)
 */
unsigned int optimalNbThreads( NbParam p=NB_HIGHMEMORY);

/**
 * impossible to automatically determine the number of cores so ...
 */
void setNbCore(unsigned int nb);


//
//template <typename MAP>
//class Foreach
//{
//	MAP& m_map;
//
//	std::vector<FunctorMapThreaded<MAP>*> m_funcs;
//
//	std::vector<Dart>* m_vd;
//
//	unsigned int m_nbth;
//
//public:
//	Foreach(MAP& map,unsigned int nbth);
//
//	void clearFunctors();
//
//	void addFunctor(FunctorMapThreaded<MAP>* funcPtr);
//
//	template<typename T>
//	T* getFunctor(unsigned int i);
//
//	template <unsigned int ORBIT>
//	void traverseCell(bool needMarkers = false, unsigned int currentThread = 0);
//
//	template <unsigned int ORBIT>
//	void traverseEachCell(bool needMarkers = false, unsigned int currentThread = 0);
//
//	void traverseDart(bool needMarkers = false, unsigned int currentThread = 0);
//
//	void traverseEachDart(bool needMarkers = false, unsigned int currentThread = 0);
//};



/**
 * Traverse cells of a map in parallel. Use quick traversal, cell markers or dart markers if available !
 * Use this version if you need to have acces to each functors after the traversal (to compute a sum or an average for example)
 * @param map the map
 * @param funcs the functors to apply (size of vector determine number of threads, and all functors must be of the same type)
 * @param needMarkers set to yes if you want that each thread use different markers. Warning if set to false (default) do not use algo with thread id or markers !!
 */
template <typename MAP, unsigned int ORBIT>
void foreach_cell(MAP& map, std::vector<FunctorMapThreaded<MAP>*>& funcs, bool needMarkers = false);

/**
 * Traverse cells of a map in parallel. Use quick traversal, cell markers or dart markers if available !
 * Use this version if you do not need to keep functors
 * @param map the map
 * @param func the functor to apply
 * @param nbth number of threads 0 for let the system choose
 * @param needMarkers set to yes if you want that each thread use different markers. Warning if set to false (default) do not use algo with thread id or markers !!
 */
template <typename MAP, unsigned int ORBIT>
void foreach_cell(MAP& map, FunctorMapThreaded<MAP>& func, unsigned int nbth = 0, bool needMarkers = false);


/**
 * Traverse cells of a map and apply differents functors in //
 * Use this version if you need to have acces to each functors after the traversal (to compute a sum or an average for example)
 * @param map the map
 * @param funcs the functors to apply ( each functors can (should!) be here of different type)
 * @param nbth number of threads
 * @param needMarkers set to yes if you want that each thread use different markers. Warning if set to false (default) do not use algo with thread id or markers !!
 */
template <typename MAP, unsigned int ORBIT>
void foreach_cell_all_thread(MAP& map, std::vector<FunctorMapThreaded<MAP>*>& funcs, bool needMarkers = false);


/**
 * Traverse darts of a map in parallel
 * Use this version if you need to have acces to each functors after the traversal (to compute a sum or an average for example)
 * @param map the map
 * @param funcs the functors to apply (size of vector determine number of threads, and all functors must be of the same type)
 * @param needMarkers set to yes if you want that each thread use different markers.Warning if set to false (default) do not use algo with thread id or markers !!
 */
template <typename MAP>
void foreach_dart(MAP& map, std::vector<FunctorMapThreaded<MAP>*>& funcs,  unsigned int nbth, bool needMarkers = false);


/**
 * Traverse darts of a map in parallel
 * @param map the map
 * @param funcs the functor
 * @param nbth number of thread to use, 0 for let the system choose
 * @param needMarkers set to yes if you want that each thread use different markers. Warning if set to false (default) do not use algo with thread id or markers !!
 */
template <typename MAP>
void foreach_dart(MAP& map, FunctorMapThreaded<MAP>& func, unsigned int nbth = 0, bool needMarkers = false);


/**
 * Traverse all elements of an attribute container (attribute handler is placed in FunctorAttribThreaded)
 * @param attr_cont the attribute container to traverse
 * @param func the fonctors to use
 */
void foreach_attrib(AttributeContainer& attr_cont, std::vector<FunctorAttribThreaded*> funcs);

/**
 * Traverse all elements of an attribute container (attribute handler is placed in FunctorAttribThreaded
 * @param attr_cont the attribute container to traverse
 * @param func the functor to use
 * @param nbth number of thread to use for computation 0 for let the system choose
 */
void foreach_attrib(AttributeContainer& attr_cont, FunctorAttribThreaded& func, unsigned int nbth = 0);


/**
 * Optimized version for // foreach with to pass (2 functors), with several loops
 * Use this version if you need to keep functors
 * @param map the map
 * @param funcsFrontnBack nbth front pass functors followed by nbth back pass functors
 * @param nbLoops number of loops to execute
 * @param needMarkers set to yes if you want that each thread use different markers (markers are allocated if necessary)
 */
template <typename MAP, unsigned int CELL>
void foreach_cell2Pass(MAP& map, std::vector<FunctorMapThreaded<MAP>*>& funcsFrontnBack, unsigned int nbLoops, bool needMarkers = false);

/**
 * Optimized version for // foreach with to pass (2 functors), with several loops
 * Use this version if you do not need to keep functors
 * @param map the map
 * @param funcFront front pass functor
 * @param funcBack back pass functor
 * @param nbLoops number of loops to execute
 * @param nbth number of threads to use
 * @param needMarkers set to yes if you want that each thread use different markers (markers are allocated if necessary)
 */
template <typename MAP, unsigned int CELL>
void foreach_cell2Pass(MAP& map, FunctorMapThreaded<MAP>& funcFront, FunctorMapThreaded<MAP>& funcBack, unsigned int nbLoops, unsigned int nbth, bool needMarkers = false);


} // namespace Parallel

} // namespace Algo

} // namespace CGoGN

#include "Algo/Parallel/parallel_foreach.hpp"

#endif
