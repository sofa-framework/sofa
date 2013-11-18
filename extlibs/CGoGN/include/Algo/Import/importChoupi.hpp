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

template <typename PFP>
bool importChoupi(const std::string& filename, std::vector<typename PFP::VEC3>& tabV, std::vector<unsigned int>& tabE)
{
	typedef typename PFP::VEC3 VEC3;

	//open file
	std::ifstream fp(filename.c_str(), std::ios::in);
	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}

	std::string ligne;
	unsigned int nbv, nbe;
	std::getline(fp, ligne);

	std::stringstream oss(ligne);
	oss >> nbv;
	oss >> nbe;

	std::cout << "nb vertices = " << nbv << std::endl;
	std::cout << "nb edges = " << nbe << std::endl;

	std::vector<unsigned int> index;
	index.reserve(2*nbv);

	//read vertices
	unsigned int id = 0;
	for(unsigned int j=0 ; j < nbv ; ++j)
	{
		do
		{
			std::getline(fp, ligne);
		} while(ligne.size() == 0);

		std::stringstream oss(ligne);

		unsigned int i;
		float x, y, z;
		oss >> i;
		oss >> x;
		oss >> y;
		oss >> z;

		VEC3 pos(x,y,z);

		//std::cout << "vec[" << j << "] = " << pos << std::endl;

		index[i] = id;
		tabV.push_back(pos);
		//tabV[j] = pos;

		//std::cout << "vec[" << j << "] = " << tabV[j] << std::endl;

		++id;
	}

	for(unsigned int i=0 ; i < nbe ; ++i)
	{
		do
		{
			std::getline(fp, ligne);
		}while(ligne.size() == 0);

		std::stringstream oss(ligne);

		unsigned int x, y;
		oss >> x;
		oss >> x;
		oss >> y;

		tabE.push_back(index[x]);
		tabE.push_back(index[y]);
		//tabE[2*i] = index[x];
		//tabE[2*i+1] = index[y];
	}

//	for(typename std::vector<VEC3>::iterator it = tabV.begin() ; it < tabV.end() ; ++it)
//		std::cout << *it << std::endl;

//	for(std::vector<unsigned int>::iterator it = tabE.begin() ; it < tabE.end() ; it = it + 2)
//		std::cout << *it << " " << *(it + 1) << std::endl;

	fp.close();

	return true;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN

