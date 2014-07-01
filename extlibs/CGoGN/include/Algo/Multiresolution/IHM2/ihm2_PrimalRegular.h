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

#ifndef _IHM2MR_PRIMAL_REGULAR_
#define _IHM2MR_PRIMAL_REGULAR_

#include "Topology/ihmap/ihm2.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor/traversor2.h"

#include "Algo/Multiresolution/filter.h"
#include "Algo/Import/importMRDAT.h"

#include <cmath>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MR
{

namespace Primal
{

namespace Regular
{

template <typename PFP>
class IHM2
{
public:
    typedef typename PFP::MAP MAP ;

protected:
    MAP& m_map;
	bool shareVertexEmbeddings ;

    std::vector<Algo::MR::Filter*> synthesisFilters ;
    std::vector<Algo::MR::Filter*> analysisFilters ;

public:
    IHM2(MAP& map) ;

    //if true : tri and quad else quad
    void addNewLevel(bool triQuad = true) ;

    void addSynthesisFilter(Algo::MR::Filter* f) { synthesisFilters.push_back(f) ; }
    void addAnalysisFilter(Algo::MR::Filter* f) { analysisFilters.push_back(f) ; }

    void clearSynthesisFilters() { synthesisFilters.clear() ; }
    void clearAnalysisFilters() { analysisFilters.clear() ; }

    void analysis() ;
    void synthesis() ;

	void addLevelFront();

	void import(Algo::Surface::Import::QuadTree& qt);
} ;

} // namespace Regular

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Multiresolution/IHM2/ihm2_PrimalRegular.hpp"

#endif // _IHM2MR_PRIMAL_REGULAR_
