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

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import
{

inline void nextNonEmptyLine(std::ifstream& fp, std::string& line)
{
	do {
		std::getline(fp, line) ;
	} while (line.size() == 0) ;
}

template <typename PFP>
bool importMRDAT(typename PFP::MAP& map, const std::string& filename, std::vector<std::string>& attrNames, QuadTree& qt)
{
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP> position = map.template getAttribute<typename PFP::VEC3, VERTEX>("position") ;
	if (!position.isValid())
		position = map.template addAttribute<typename PFP::VEC3, VERTEX>("position") ;

	attrNames.push_back(position.name()) ;

	AttributeContainer& container = map.template getAttributeContainer<VERTEX>() ;

	// open file
	std::ifstream fp(filename.c_str(), std::ios::in) ;
	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl ;
		return false ;
	}

	std::string line ;

	nextNonEmptyLine(fp, line) ;
	if (line.rfind("Multires data file") == std::string::npos)
	{
		CGoGNerr << "Problem reading MRDAT file" << CGoGNendl ;
		CGoGNerr << line << CGoGNendl ;
		return false ;
	}

	// read the depth
	unsigned int depth ;
	{
		nextNonEmptyLine(fp, line) ;
		std::stringstream oss(line) ;
		std::string s ;
		oss >> s ;
		oss >> qt.depth ;
		depth = qt.depth;
	}

	std::cout << "  MR depth -> " << depth << std::endl ;

	// read vertices
	nextNonEmptyLine(fp, line) ;
	if (line.rfind("Vertices") == std::string::npos)
	{
		CGoGNerr << "Problem reading MRDAT file" << CGoGNendl ;
		CGoGNerr << line << CGoGNendl ;
		return false ;
	}

	std::cout << "  Read vertices.." << std::flush ;

	qt.roots.clear() ;
	qt.darts.clear() ;
	qt.verticesID.clear() ;

	nextNonEmptyLine(fp, line) ;
	while(line.rfind("Triangles") == std::string::npos)
	{
		std::stringstream oss(line) ;

		unsigned int level ;
		oss >> level ;

		float x, y, z ;
		oss >> x ;
		oss >> y ;
		oss >> z ;
		typename PFP::VEC3 pos(x, y, z) ;

		unsigned int id = container.insertLine() ;
		position[id] = pos ;
		qt.verticesID.push_back(id) ;

		nextNonEmptyLine(fp, line) ;
	}

	std::cout << "..done (nb vertices -> " << qt.verticesID.size() << ")" << std::endl ;
	std::cout << "  Read triangles (build quadtree).." << std::flush ;

	QuadTreeNode* current = NULL ;
	unsigned int currentLevel = -1 ;
	std::vector<unsigned int> lastNum ;
	lastNum.resize(depth + 1) ;

	nextNonEmptyLine(fp, line) ;
	while(line.rfind("end") == std::string::npos)
	{
		std::stringstream oss(line) ;

		std::string name ;
		oss >> name ;

		unsigned int num, root, idx0, idx1, idx2 ;
		oss >> num ;
		oss >> root ;
		oss >> idx0 ;
		oss >> idx1 ;
		oss >> idx2 ;

		if(root == 1)
		{
			assert(num == 0) ;
			QuadTreeNode* n = new QuadTreeNode() ;
			n->indices[0] = idx0 ;
			n->indices[1] = idx1 ;
			n->indices[2] = idx2 ;
			qt.roots.push_back(n) ;
			current = n ;
			currentLevel = 0 ;
			lastNum[0] = 0 ;
		}
		else
		{
			if(num == lastNum[currentLevel] + 1) // on lit un autre triangle du même niveau
			{
				current = current->parent->children[num] ;
			}
			else // on monte ou on descend d'un niveau
			{
				if(num == 0) // on subdivise le triangle courant
				{
					current->subdivide() ;
					current = current->children[0] ;
					++currentLevel ;
				}
				else // on remonte d'un niveau
				{
					assert(lastNum[currentLevel] == 3) ;
					do
					{
						current = current->parent->parent->children[num] ;
						--currentLevel ;
					} while(lastNum[currentLevel] == 3) ;
				}
			}
			current->indices[0] = idx0 ;
			current->indices[1] = idx1 ;
			current->indices[2] = idx2 ;
			lastNum[currentLevel] = num ;
		}

		nextNonEmptyLine(fp, line) ;
	}

	std::cout << "..done" << std::endl ;

	fp.close() ;

	std::cout << "  Create base level mesh.." << std::flush ;

	VertexAutoAttribute<NoTypeNameAttribute<std::vector<Dart> >, typename PFP::MAP> vecDartsPerVertex(map, "incidents") ;
	DartMarkerNoUnmark<typename PFP::MAP> m(map) ;

	unsigned int vemb = EMBNULL;
	auto fsetemb = [&] (Dart d) { map.template setDartEmbedding<VERTEX>(d, vemb); };

	unsigned nbf = qt.roots.size() ;

	// for each root face
	for(unsigned int i = 0; i < nbf; ++i)
	{
		Dart d = map.newFace(3, false) ;
		qt.darts.push_back(d) ;
		for (unsigned int j = 0; j < 3; ++j)
		{
			unsigned int idx = qt.roots[i]->indices[j] ;
			vemb = qt.verticesID[idx] ;

			map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb) ;

			m.mark(d) ;								// mark on the fly to unmark on second loop
			vecDartsPerVertex[vemb].push_back(d) ;	// store incident darts for fast adjacency reconstruction
			d = map.phi1(d) ;
		}
	}

	// reconstruct neighbourhood between root faces
	unsigned int nbBoundaryEdges = 0 ;
	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (m.isMarked(d))
		{
			// darts incident to end vertex of edge
			std::vector<Dart>& vec = vecDartsPerVertex[map.phi1(d)] ;

			unsigned int embd = map.template getEmbedding<VERTEX>(d) ;
			Dart good_dart = NIL ;
			for (typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
			{
				if (map.template getEmbedding<VERTEX>(map.phi1(*it)) == embd)
					good_dart = *it ;
			}

			if (good_dart != NIL)
			{
				map.sewFaces(d, good_dart, false) ;
				m.template unmarkOrbit<EDGE>(d) ;
			}
			else
			{
				m.unmark(d) ;
				++nbBoundaryEdges ;
			}
		}
	}

	if (nbBoundaryEdges > 0)
	{
//		map.closeMap() ;
		CGoGNout << "Open mesh.. not managed yet.." << CGoGNendl ;
		return false ;
	}

	std::cout << "..done" << std::endl ;

	return true ;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN
