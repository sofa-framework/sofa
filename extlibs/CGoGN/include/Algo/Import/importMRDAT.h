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

#ifndef __IMPORT_MR_DAT__
#define __IMPORT_MR_DAT__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{

class QuadTreeNode
{
public:	
	unsigned int indices[3] ;
	QuadTreeNode* children[4] ;
	QuadTreeNode* parent ;
	unsigned int level ;
	
	QuadTreeNode()
	{
		for(unsigned int i = 0; i < 3; ++i)
			indices[i] = -1 ;
		for(unsigned int i = 0; i < 4; ++i)
			children[i] = NULL ;
		parent = NULL ;
		level = 0 ;
	}

	~QuadTreeNode()
	{
		for(unsigned int i = 0; i < 4; ++i)
			if(children[i] != NULL)
				delete children[i] ;
	}

	void subdivide()
	{
		assert(!isSubdivided()) ;
		for(unsigned int i = 0; i < 4; ++i)
		{
			children[i] = new QuadTreeNode() ;
			children[i]->parent = this ;
			children[i]->level = level + 1 ;
		}
	}

	bool isSubdivided()
	{
		return children[0] != NULL ;
	}

	template <typename PFP>
	void embed(typename PFP::MAP& map, Dart d, std::vector<unsigned int>& vID)
	{
		assert(map.getCurrentLevel() == level) ;

		if(isSubdivided())
		{
			unsigned int v0 = vID[indices[0]] ;
			unsigned int v1 = vID[indices[1]] ;
//			unsigned int v2 = vID[indices[2]] ;

			Dart it = d ;
			do
			{
				Dart next = map.phi1(it) ;
				unsigned int emb = map.template getEmbedding<VERTEX>(it) ;
				unsigned int idx = emb == v0 ? 0 : emb == v1 ? 1 : 2 ;
				map.incCurrentLevel() ;
				Dart dd = map.phi1(next) ;
				unsigned int oldEmb = map.template getEmbedding<VERTEX>(dd) ;
				unsigned int newEmb = vID[children[0]->indices[idx]] ;
				if(oldEmb == EMBNULL)
				{
					map.template setOrbitEmbedding<VERTEX>(dd, newEmb) ;
					map.pushLevel() ;
					for(unsigned int i = map.getCurrentLevel() + 1; i <= map.getMaxLevel(); ++i)
					{
						map.setCurrentLevel(i) ;
						map.template setOrbitEmbedding<VERTEX>(dd, newEmb) ;
					}
					map.popLevel() ;
				}
				else
					assert(oldEmb == newEmb) ;
				map.decCurrentLevel() ;
				it = next ;
			} while(it != d) ;

			map.incCurrentLevel() ;
			Dart d0 = map.phi2(map.phi1(d)) ;
			children[0]->embed<PFP>(map, d0, vID) ;
			map.decCurrentLevel() ;

			do
			{
				unsigned int emb = map.template getEmbedding<VERTEX>(it) ;
				unsigned int idx = emb == v0 ? 0 : emb == v1 ? 1 : 2 ;
				map.incCurrentLevel() ;
				children[idx+1]->embed<PFP>(map, it, vID) ;
				map.decCurrentLevel() ;
				it = map.phi1(it) ;
			} while(it != d) ;
		}
		else
		{
			if(map.getCurrentLevel() < map.getMaxLevel())
				std::cout << "adaptive subdivision not managed yet" << std::endl ;
		}
	}

	void print()
	{
		std::cout << indices[0] << " " << indices[1] << " " << indices[2] << std::endl ;
		if(isSubdivided())
		{
			for(unsigned int i = 0; i < 4; ++i)
				children[i]->print() ;
		}
	}
} ;

class QuadTree
{
public:
	std::vector<QuadTreeNode*> roots ;
	std::vector<Dart> darts ;
	std::vector<unsigned int> verticesID ;

	~QuadTree()
	{
		for(unsigned int i = 0; i < roots.size(); ++i)
			delete roots[i] ;
	}

	template <typename PFP>
	void embed(typename PFP::MAP& map)
	{
		for(unsigned int i = 0; i < roots.size(); ++i)
			roots[i]->embed<PFP>(map, darts[i], verticesID) ;
	}

	void print()
	{
		std::cout << "printing quadtree (" << roots.size() << " roots)" << std::endl ;
		for(unsigned int i = 0; i < roots.size(); ++i)
		{
			std::cout << "root " << i << std::endl ;
			roots[i]->print() ;
		}
	}
} ;

template <typename PFP>
bool importMRDAT(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, QuadTree& qt) ;

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN

#include "Algo/Import/importMRDAT.hpp"

#endif
