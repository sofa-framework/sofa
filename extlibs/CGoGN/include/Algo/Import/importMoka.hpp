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

namespace Volume
{

namespace Import
{

template <typename PFP>
bool importMoka(typename PFP::MAP& gmap, const std::string& filename, std::vector<std::string>& attrNames)
{
	if(gmap.mapTypeName().compare("GMap3") != 0)
	{
		CGoGNerr << "Unable to load MOKA file " << filename << " : given map is not a 3-GMap" << CGoGNendl;
		return false ;
	}

	// open file
	igzstream fp(filename.c_str(), std::ios::in|std::ios::binary);

	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}

	VertexAttribute<typename PFP::VEC3> position =  gmap.template getAttribute<typename PFP::VEC3, VERTEX>("position");
	if (!position.isValid())
		position = gmap.template addAttribute<typename PFP::VEC3, VERTEX>("position");

	attrNames.push_back(position.name());

	AttributeContainer& vertexContainer = gmap.template getAttributeContainer<VERTEX>() ;

	std::string ligne;
	std::getline (fp, ligne);

	// check if the file format is in ascii
	if(ligne.compare("Moka file [ascii]") != 0)
	{
		CGoGNerr << "Unable to load this MOKA file " << filename << CGoGNendl;
		return false;
	}

	// ignore 2nd line
	std::getline (fp, ligne);

	DartAttribute<Dart> att_beta0 = gmap.template getAttribute<Dart, DART>("beta0");
	DartAttribute<Dart> att_beta1 = gmap.template getAttribute<Dart, DART>("beta1");
	DartAttribute<Dart> att_beta2 = gmap.template getAttribute<Dart, DART>("beta2");
	DartAttribute<Dart> att_beta3 = gmap.template getAttribute<Dart, DART>("beta3");

	std::map<Dart, unsigned int> map_dart_emb;

	while(!std::getline(fp, ligne).eof())
	{
		std::stringstream oss(ligne);
		unsigned int beta0, beta1, beta2, beta3;

		Dart d = gmap.newDart();

		// read involutions
		oss >> beta0;
		oss >> beta1;
		oss >> beta2;
		oss >> beta3;

		att_beta0[d] = beta0;
		att_beta1[d] = beta1;
		att_beta2[d] = beta2;
		att_beta3[d] = beta3;

		// ignore markers
		unsigned int tmp;
		for(unsigned int i = 0 ; i < 4 ; ++i)
			oss >> tmp;

		// check if contains embedding
		unsigned int emb;
		oss >> emb;

		if(emb == 1)
		{
			typename PFP::VEC3 pos;
			oss >> pos[0];
			oss >> pos[1];
			oss >> pos[2];

			unsigned int id = vertexContainer.insertLine();
			position[id] = pos;
			map_dart_emb.insert(std::pair<Dart,unsigned int>(d, id));
		}
	}

	for(typename std::map<Dart, unsigned int>::iterator it = map_dart_emb.begin() ; it != map_dart_emb.end() ; ++it)
		gmap.template setOrbitEmbedding<VERTEX>(it->first, it->second);

	gmap.closeMap();

	fp.close();
	return true;
}

} // namespace Import

}

} // namespace Algo

} // namespace CGoGN
