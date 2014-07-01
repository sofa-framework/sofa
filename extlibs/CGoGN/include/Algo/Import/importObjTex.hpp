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

#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/autoAttributeHandler.h"
#include "Container/fakeAttribute.h"
#include <fstream>
#include <algorithm>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Import 
{

template <typename PFP>
OBJModel<PFP>::OBJModel(typename PFP::MAP& map):
	m_map(map), m_maxTextureSize(2048),
	m_tagV(0),m_tagVT(0),m_tagVN(0),m_tagG(0),m_tagF(0),
	m_specialVertices(map),m_dirtyEdges(map)
{
}

template <typename PFP>
OBJModel<PFP>::~OBJModel()
{
	for (std::vector<MaterialOBJ*>::iterator it = m_materials.begin(); it != m_materials.end(); ++it)
		delete *it;
}

template <typename PFP>
inline void OBJModel<PFP>::setMaxTextureSize(unsigned int mts)
{
	m_maxTextureSize = mts;
}

template <typename PFP>
inline typename PFP::VEC3 OBJModel<PFP>::getPosition(Dart d)
{
	return m_positions[d];
}

template <typename PFP>
inline typename PFP::VEC3 OBJModel<PFP>::getNormal(Dart d)
{
	if (m_specialVertices.isMarked(d))
		return m_normalsF[d];
	return m_normals[d];
}

template <typename PFP>
inline Geom::Vec2f OBJModel<PFP>::getTexCoord(Dart d)
{
	if (m_specialVertices.isMarked(d))
		return m_texCoordsF[d];
	return m_texCoords[d];
}

template <typename PFP>
void OBJModel<PFP>::setPositionAttribute(VertexAttribute<Geom::Vec3f, typename PFP::MAP> position)
{
	m_positions = position;
}

template <typename PFP>
void OBJModel<PFP>::setNormalAttribute(VertexAttribute<Geom::Vec3f, typename PFP::MAP> normal)
{
	m_normals = normal;
}

template <typename PFP>
void OBJModel<PFP>::setTexCoordAttribute(VertexAttribute<Geom::Vec2f, typename PFP::MAP>texcoord)
{
	m_texCoords = texcoord;
}

template <typename PFP>
void OBJModel<PFP>::readMaterials(const std::string& filename)
{
	m_materials.reserve(m_materialNames.size());

	if (!filename.empty())
	{
		m_matFileName = filename;
	}

	// open file
	std::ifstream fp(m_matFileName.c_str());
	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << m_matFileName << CGoGNendl;
		return ;
	}
	
	MaterialOBJ* currentMat = NULL;

	m_materials.resize(m_materialNames.size(),NULL);

	std::string ligne;
	do
	{
		std::getline (fp, ligne);
		std::stringstream oss(ligne);
		std::string tag;
		oss >> tag;
		if (tag == "newmtl")
		{
			std::string name;
			oss >> name ;
			std::map<std::string,int>::iterator it = m_materialNames.find(name);
			if (it ==  m_materialNames.end())
			{
				CGoGNerr << "Skipping material "<< name << CGoGNendl;
				do
				{
					fp >> tag;
					
				}while (!fp.eof() && (tag == "new mtl")); 
			}
			else
			{
//				CGoGNout << "Reading material "<< name << CGoGNendl;
				currentMat = new MaterialOBJ();
				m_materials[it->second] = currentMat;
				currentMat->name = name;
			}
		}
		else if (currentMat != NULL)
		{
			if (tag == "Ka")
			{
				oss >> currentMat->ambiantColor[0];
				oss >> currentMat->ambiantColor[1];
				oss >> currentMat->ambiantColor[2];
			}
			if (tag == "Kd")
			{
				oss >> currentMat->diffuseColor[0];
				oss >> currentMat->diffuseColor[1];
				oss >> currentMat->diffuseColor[2];

			}
			if (tag == "Ks")
			{
				oss >> currentMat->specularColor[0];
				oss >> currentMat->specularColor[1];
				oss >> currentMat->specularColor[2];
			}
			if (tag == "Ns")
			{
				oss >> currentMat->shininess;
			}
			if (tag == "Tf")
			{
				oss >> currentMat->transparentFilter[0];
				oss >> currentMat->transparentFilter[1];
				oss >> currentMat->transparentFilter[2];
			}

			if ((tag == "illum"))
			{
				oss >> currentMat->illuminationModel;
			}


			if ((tag == "Tr") || (tag == "d"))
			{
				oss >> currentMat->transparency;
			}
			
			if (tag == "map_Kd") // diffuse texture
			{
				if (currentMat->textureDiffuse == NULL)
				{
					currentMat->textureDiffuse = new Utils::Texture<2,Geom::Vec3uc>(GL_UNSIGNED_BYTE);
					std::string buff;
					getline (oss, buff);
					std::string tname = buff.substr(buff.find_first_not_of(' '));
					if (tname[tname.length()-1] == '\r')
						tname = tname.substr(0,tname.length()-1);

					currentMat->textureDiffuse->load(m_matPath+tname);
//					CGoGNout << "Loading texture "<< m_matPath+tname << " -> "<<std::hex << currentMat->textureDiffuse <<std::dec<<CGoGNendl;
					currentMat->textureDiffuse->scaleNearest( currentMat->textureDiffuse->newMaxSize(m_maxTextureSize));
					currentMat->textureDiffuse->setFiltering(GL_LINEAR);
					currentMat->textureDiffuse->setWrapping(GL_REPEAT);
					currentMat->textureDiffuse->update();
				}
			}

			if (tag == "map_Ka") // ambiant texture
			{
//				CGoGNerr << tag << " not yet supported in OBJ material reading" << CGoGNendl;
				if (currentMat->textureDiffuse == NULL)
				{
					currentMat->textureDiffuse = new Utils::Texture<2,Geom::Vec3uc>(GL_UNSIGNED_BYTE);
					std::string buff;
					getline (oss, buff);
					std::string tname = buff.substr(buff.find_first_not_of(' '));
					if (tname[tname.length()-1] == '\r')
						tname = tname.substr(0,tname.length()-1);
					currentMat->textureDiffuse->load(m_matPath+tname);
					CGoGNout << "Loading texture "<< m_matPath+tname << " -> "<<std::hex << currentMat->textureDiffuse <<std::dec<<CGoGNendl;
					currentMat->textureDiffuse->scaleNearest( currentMat->textureDiffuse->newMaxSize(m_maxTextureSize));
					currentMat->textureDiffuse->setFiltering(GL_LINEAR);
					currentMat->textureDiffuse->setWrapping(GL_REPEAT);
					currentMat->textureDiffuse->update();
				}

			}

			if (tag == "map_d") // opacity texture
			{
				CGoGNerr << tag << " not yet supported in OBJ material reading" << CGoGNendl;
			}
			if ((tag == "map_bump") || (tag == "bump"))
			{
				CGoGNerr << tag << " not yet supported in OBJ material reading" << CGoGNendl;
			}
			tag="";
		}
	}while (!fp.eof());

	for (std::vector<MaterialOBJ*>::iterator it = m_materials.begin(); it != m_materials.end(); ++it)
	{
		if (*it == NULL)
			CGoGNerr << "Warning missing material in .mtl"<< CGoGNendl;
	}

}

template <typename PFP>
unsigned int OBJModel<PFP>::getMaterialIndex(const std::string& name) const
{
	std::map<std::string, int>::iterator it = m_materialNames.find(name);
	if (it != m_materialNames.end())
		return it->second;
	return 0xffffffff;
}

//template <typename PFP>
//bool OBJModel<PFP>::generateBrowsers(std::vector<ContainerBrowser*>& browsers)
//{
//	browsers.clear();
//	if (m_groupNames.empty())
//		return false;

//	ContainerBrowserLinked* MBLptr = new ContainerBrowserLinked(m_map,DART);
//	browsers.push_back(MBLptr);

//	for (unsigned int i = 1; i<m_groupNames.size(); ++i)
//	{
//		ContainerBrowser* MBptr = new ContainerBrowserLinked(*MBLptr);
//		browsers.push_back(MBptr);
//		m_groupMaterialID[i]= m_materialNames[m_groupMaterialNames[i]];
//	}

//	for (Dart d=m_map.begin(); d!=m_map.end(); m_map.next(d))
//	{
//		unsigned int g = m_groups[d] -1 ; // groups are name from 1
//		ContainerBrowserLinked* mb = static_cast<ContainerBrowserLinked*>(browsers[g]);
//		mb->pushBack(d.index);
//	}
//	return true;
//}
	
template <typename PFP>
short OBJModel<PFP>::readObjLine(std::stringstream& oss, std::vector<unsigned int>& indices)
{
	indices.clear();
	
	unsigned int nb=0;
	while (!oss.eof())  // lecture de tous les indices
	{
		int index;
		oss >> index;

		indices.push_back(index);

		int slash = 0;
		char sep='_';
		do
		{
			oss >> sep;
			if (sep =='/')
				++slash;
		} while ( ((sep=='/') || (sep ==' ')) && !oss.eof() ) ;

		if ((sep>='0') && (sep<='9'))
			oss.seekg(-1,std::ios_base::cur);

		if (slash == 0)
		{
			if (indices.size()%3 == 1)
			{
				indices.push_back(0);
				indices.push_back(0);
			}
			if (indices.size()%3 == 2)
			{
				indices.push_back(0);
			}
			nb++;
		}


		if (slash == 2)
		{
			indices.push_back(0);
		}
	}
	return nb;
}

template <typename PFP>
unsigned int OBJModel<PFP>::createSimpleVBO_P(Utils::VBO* positionVBO)
{
	TraversorF<typename PFP::MAP> traf(m_map);
	std::vector<Geom::Vec3f> posBuff;
	posBuff.reserve(16384);

	unsigned int nbtris = 0;
	for (Dart d=traf.begin(); d!= traf.end(); d = traf.next())
	{
		Dart e = m_map.phi1(d);
		Dart f = m_map.phi1(e);
		do
		{
			posBuff.push_back(m_positions[d]);
			posBuff.push_back(m_positions[e]);
			posBuff.push_back(m_positions[f]);
			e=f;
			f = m_map.phi1(e);
			nbtris++;
		}while (f!=d);
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();

	return 3*nbtris;
}

template <typename PFP>
unsigned int OBJModel<PFP>::createSimpleVBO_PT(Utils::VBO* positionVBO, Utils::VBO* texcoordVBO)
{
	TraversorF<typename PFP::MAP> traf(m_map);
	std::vector<Geom::Vec3f> posBuff;
	std::vector<Geom::Vec2f> TCBuff;
	posBuff.reserve(16384);
	TCBuff.reserve(16384);

	unsigned int nbtris = 0;
	for (Dart d=traf.begin(); d!= traf.end(); d = traf.next())
	{
		Dart e = m_map.phi1(d);
		Dart f = m_map.phi1(e);
		do
		{
			posBuff.push_back(m_positions[d]);
			if (m_specialVertices.isMarked(d))
				TCBuff.push_back(m_texCoordsF[d]);
			else
				TCBuff.push_back(m_texCoords[d]);

			posBuff.push_back(m_positions[e]);
			if (m_specialVertices.isMarked(e))
				TCBuff.push_back(m_texCoordsF[e]);
			else
				TCBuff.push_back(m_texCoords[e]);

			posBuff.push_back(m_positions[f]);
			if (m_specialVertices.isMarked(f))
				TCBuff.push_back(m_texCoordsF[f]);
			else
				TCBuff.push_back(m_texCoords[f]);
			e=f;
			f = m_map.phi1(e);
			nbtris++;
		}while (f!=d);
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();

	texcoordVBO->setDataSize(2);
	texcoordVBO->allocate(TCBuff.size());
	Geom::Vec2f* ptrTC = reinterpret_cast<Geom::Vec2f*>(texcoordVBO->lockPtr());
	memcpy(ptrTC,&TCBuff[0],TCBuff.size()*sizeof(Geom::Vec2f));
	texcoordVBO->releasePtr();

	return 3*nbtris;
}

template <typename PFP>
unsigned int OBJModel<PFP>::createSimpleVBO_PN(Utils::VBO* positionVBO, Utils::VBO* normalVBO )
{
	TraversorF<typename PFP::MAP> traf(m_map);
	std::vector<Geom::Vec3f> posBuff;
	std::vector<Geom::Vec2f> normalBuff;
	posBuff.reserve(16384);
	normalBuff.reserve(16384);

	unsigned int nbtris = 0;
	for (Dart d=traf.begin(); d!= traf.end(); d = traf.next())
	{
		Dart e = m_map.phi1(d);
		Dart f = m_map.phi1(e);
		do
		{
			posBuff.push_back(m_positions[d]);
			if (m_specialVertices.isMarked(d))
			{
				normalBuff.push_back(m_normalsF[d]);
			}
			else
			{
				normalBuff.push_back(m_normals[d]);
			}

			posBuff.push_back(m_positions[e]);
			if (m_specialVertices.isMarked(e))
			{
				normalBuff.push_back(m_normalsF[e]);
			}
			else
			{
				normalBuff.push_back(m_normals[e]);
			}

			posBuff.push_back(m_positions[f]);
			if (m_specialVertices.isMarked(f))
			{
				normalBuff.push_back(m_normalsF[f]);
			}
			else
			{
				normalBuff.push_back(m_normals[f]);
			}
			e=f;
			f = m_map.phi1(e);
			nbtris++;
		}while (f!=d);
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();

	normalVBO->setDataSize(3);
	normalVBO->allocate(normalBuff.size());
	Geom::Vec3f* ptrNormal = reinterpret_cast<Geom::Vec3f*>(normalVBO->lockPtr());
	memcpy(ptrNormal, &normalBuff[0], normalBuff.size()*sizeof(Geom::Vec3f));
	normalVBO->releasePtr();


	return 3*nbtris;
}

template <typename PFP>
unsigned int OBJModel<PFP>::createSimpleVBO_PTN(Utils::VBO* positionVBO, Utils::VBO* texcoordVBO, Utils::VBO* normalVBO )
{
	if (!m_normals.isValid())
	{
		CGoGNerr << "no normal attribute "<< CGoGNendl;
		return 0;
	}
	if (!m_texCoords.isValid())
	{
		CGoGNerr << "no tex coords attribute "<< CGoGNendl;
		return 0;
	}

	TraversorF<typename PFP::MAP> traf(m_map);
	std::vector<Geom::Vec3f> posBuff;
	std::vector<Geom::Vec2f> TCBuff;
	std::vector<Geom::Vec3f> normalBuff;
	posBuff.reserve(16384);
	TCBuff.reserve(16384);
	normalBuff.reserve(16384);

	unsigned int nbtris = 0;
	for (Dart d=traf.begin(); d!= traf.end(); d = traf.next())
	{
		Dart e = m_map.phi1(d);
		Dart f = m_map.phi1(e);
		do
		{
			posBuff.push_back(m_positions[d]);
			if (m_specialVertices.isMarked(d))
			{
				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[d]);
				else
					TCBuff.push_back(m_texCoords[d]);
				if (hasNormals())
					normalBuff.push_back(m_normalsF[d]);
				else
					normalBuff.push_back(m_normals[d]);
			}
			else
			{
				TCBuff.push_back(m_texCoords[d]);
				normalBuff.push_back(m_normals[d]);
			}

			posBuff.push_back(m_positions[e]);
			if (m_specialVertices.isMarked(e))
			{
				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[e]);
				else
					TCBuff.push_back(m_texCoords[e]);
				if (hasNormals())
					normalBuff.push_back(m_normalsF[e]);
				else
					normalBuff.push_back(m_normals[e]);
			}
			else
			{
				TCBuff.push_back(m_texCoords[e]);
				normalBuff.push_back(m_normals[e]);
			}

			posBuff.push_back(m_positions[f]);
			if (m_specialVertices.isMarked(f))
			{
				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[f]);
				else
					TCBuff.push_back(m_texCoords[f]);
				if (hasNormals())
					normalBuff.push_back(m_normalsF[f]);
				else
					normalBuff.push_back(m_normals[f]);
			}
			else
			{
				TCBuff.push_back(m_texCoords[f]);
				normalBuff.push_back(m_normals[f]);
			}
			e=f;
			f = m_map.phi1(e);
			nbtris++;
		}while (f!=d);
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();

	texcoordVBO->setDataSize(2);
	texcoordVBO->allocate(TCBuff.size());
	Geom::Vec2f* ptrTC = reinterpret_cast<Geom::Vec2f*>(texcoordVBO->lockPtr());
	memcpy(ptrTC,&TCBuff[0],TCBuff.size()*sizeof(Geom::Vec2f));
	texcoordVBO->releasePtr();

	normalVBO->setDataSize(3);
	normalVBO->allocate(normalBuff.size());
	Geom::Vec3f* ptrNormal = reinterpret_cast<Geom::Vec3f*>(normalVBO->lockPtr());
	memcpy(ptrNormal, &normalBuff[0], normalBuff.size()*sizeof(Geom::Vec3f));
	normalVBO->releasePtr();

	return 3*nbtris;
}

template <typename PFP>
bool OBJModel<PFP>::createGroupMatVBO_P( Utils::VBO* positionVBO)
{
	m_beginIndices.clear();
	m_nbIndices.clear();

	if (!m_normals.isValid())
	{
		CGoGNerr << "no normal attribute "<< CGoGNendl;
		return false;
	}
	if (!m_texCoords.isValid())
	{
		CGoGNerr << "no tex coords attribute "<< CGoGNendl;
		return false;
	}

	std::vector< std::vector<Dart> > group_faces; //(m_materialNames.size());
	group_faces.reserve(16384);
	m_groupIdx.reserve(16384);
	m_sgMat.reserve(16384);

	unsigned int c_sg=0;
	group_faces.resize(1);

	TraversorF<typename PFP::MAP> traf(m_map);
	Dart d=traf.begin();

	unsigned int c_grp = m_groups[d];
	unsigned int c_mat = m_attMat[d];
	m_sgMat.push_back(c_mat);

	m_groupFirstSub.push_back(0);

	if (m_tagG != 0)
		m_groupIdx.push_back(c_grp);
	else
		m_groupIdx.push_back(0);

	while (d!= traf.end())
	{
		if ((m_groups[d] != c_grp) || (m_attMat[d] != c_mat))
		{
			c_sg++;

			if (m_groups[d] != c_grp)
			{
				m_groupNbSub.push_back(c_sg-m_groupFirstSub.back());
				m_groupFirstSub.push_back(c_sg);
			}

			c_grp = m_groups[d];
			c_mat = m_attMat[d];
			m_sgMat.push_back(c_mat);

			if (m_tagG != 0)
				m_groupIdx.push_back(c_grp);
			else
				m_groupIdx.push_back(0);

			group_faces.resize(c_sg+1);
		}

		group_faces[c_sg].push_back(d);
		d = traf.next();
	}

	m_groupNbSub.push_back(c_sg+1-m_groupFirstSub.back()); // nb sub-group of last group

	// merging same material sub-groups of same group
	for (unsigned int g=0; g<m_groupNbSub.size(); ++g)
	{
		unsigned int fsg = m_groupFirstSub[g];
		unsigned int lsg = m_groupFirstSub[g]+m_groupNbSub[g]-1;

		for (unsigned int s=fsg; s<=lsg; ++s)
		{
			if (m_sgMat[s] != 0xffffffff)
			{
				for (unsigned ss=s+1; ss<=lsg; ++ss)
				{
					if (m_sgMat[ss] == m_sgMat[s])
					{
						group_faces[s].insert(group_faces[s].end(),group_faces[ss].begin(),group_faces[ss].end());
						group_faces[ss].clear();
						m_sgMat[ss] = 0xffffffff;
					}
				}
			}
		}
	}

	// compact group_faces/m_groupIdx/m_sgMat
	unsigned int outSg=0;
	for (unsigned int inSg=0; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_sgMat[inSg] != 0xffffffff)
		{
			if (outSg != inSg)
			{
				group_faces[outSg].swap(group_faces[inSg]);
				m_groupIdx[outSg] = m_groupIdx[inSg];
				m_sgMat[outSg] = m_sgMat[inSg];
			}
			outSg++;
		}
	}
	group_faces.resize(outSg);
	m_groupIdx.resize(outSg);
	m_sgMat.resize(outSg);

	// recreate m_groupFirstSub & m_groupNbSub
	unsigned int outGr=0;
	m_groupFirstSub[0] = m_groupIdx[0];
	for (unsigned int inSg=1; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_groupIdx[inSg] != m_groupIdx[inSg-1])
		{
			m_groupNbSub[outGr] = inSg - m_groupFirstSub[outGr];
			outGr++;
			m_groupFirstSub[outGr] = inSg;
		}
	}
	m_groupNbSub[outGr+1] = m_sgMat.size() - m_groupNbSub[outGr];

	// now create VBOs

	std::vector<Geom::Vec3f> posBuff;
	posBuff.reserve(16384);

	unsigned int firstIndex = 0;

	unsigned int sz = group_faces.size();
	m_beginIndices.resize(sz);
	m_nbIndices.resize(sz);
	m_groupIdx.resize(sz);

	for (unsigned int g=0; g<sz; ++g)
	{
		unsigned int nbtris = 0;
		std::vector<Dart>& traf = group_faces[g];

		if (m_tagG != 0)
			m_groupIdx[g] = m_groups[traf.front()];
		else
			m_groupIdx[g]=0;

		for (std::vector<Dart>::iterator id=traf.begin(); id!= traf.end(); ++id)
		{
			Dart d = *id;

			Dart e = m_map.phi1(d);
			Dart f = m_map.phi1(e);
			do
			{
				posBuff.push_back(m_positions[d]);

				posBuff.push_back(m_positions[e]);

				posBuff.push_back(m_positions[f]);

				e=f;
				f = m_map.phi1(e);
				nbtris++;
			}while (f!=d);
		}
		m_beginIndices[g] = firstIndex;
		m_nbIndices[g] = 3*nbtris;

		firstIndex += 3*nbtris;
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();


	// compute BBs ( TO DO: a method ?)
	computeBB(posBuff);

	return true;
}

//template <typename PFP>
//bool OBJModel<PFP>::createGroupMatVBO_P(Utils::VBO* positionVBO)
//{
//	m_beginIndices.clear();
//	m_nbIndices.clear();

//	std::vector<Geom::Vec3f> posBuff;
//	posBuff.reserve(16384);

//	std::vector< std::vector<Dart> > group_faces(m_materialNames.size());
//	TraversorF<typename PFP::MAP> traf(m_map);
//	for (Dart d=traf.begin(); d!= traf.end(); d = traf.next())
//	{
//		unsigned int g = m_attMat[d];
//		group_faces[g].push_back(d);
//	}

//	unsigned int firstIndex = 0;

//	unsigned int sz = group_faces.size();
//	m_beginIndices.resize(sz);
//	m_nbIndices.resize(sz);
//	m_groupIdx.resize(sz);

//	for (unsigned int g=0; g<sz; ++g)
//	{
//		unsigned int nbtris = 0;
//		std::vector<Dart>& traf = group_faces[g];

//		if (m_tagG != 0)
//			m_groupIdx[g] = m_groups[traf.front()];
//		else
//			m_groupIdx[g]=0;

//		for (std::vector<Dart>::iterator id=traf.begin(); id!= traf.end(); ++id)
//		{
//			Dart d = *id;
//			Dart e = m_map.phi1(d);
//			Dart f = m_map.phi1(e);
//			do
//			{
//				posBuff.push_back(m_positions[d]);
//				posBuff.push_back(m_positions[e]);
//				posBuff.push_back(m_positions[f]);
//				e=f;
//				f = m_map.phi1(e);
//				nbtris++;
//			}while (f!=d);
//		}
//		m_beginIndices[g] = firstIndex;
//		m_nbIndices[g] = 3*nbtris;
//		firstIndex += 3*nbtris;
//	}
//	positionVBO->setDataSize(3);
//	positionVBO->allocate(posBuff.size());
//	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
//	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
//	positionVBO->releasePtr();

//	updateGroups();

//	return true;
//}

template <typename PFP>
bool OBJModel<PFP>::createGroupMatVBO_PT( Utils::VBO* positionVBO,
											Utils::VBO* texcoordVBO)
{
	m_beginIndices.clear();
	m_nbIndices.clear();

	if (!m_normals.isValid())
	{
		CGoGNerr << "no normal attribute "<< CGoGNendl;
		return false;
	}
	if (!m_texCoords.isValid())
	{
		CGoGNerr << "no tex coords attribute "<< CGoGNendl;
		return false;
	}

	std::vector< std::vector<Dart> > group_faces; //(m_materialNames.size());
	group_faces.reserve(16384);
	m_groupIdx.reserve(16384);
	m_sgMat.reserve(16384);

	unsigned int c_sg=0;
	group_faces.resize(1);

	TraversorF<typename PFP::MAP> traf(m_map);
	Dart d=traf.begin();

	unsigned int c_grp = m_groups[d];
	unsigned int c_mat = m_attMat[d];
	m_sgMat.push_back(c_mat);

	m_groupFirstSub.push_back(0);

	if (m_tagG != 0)
		m_groupIdx.push_back(c_grp);
	else
		m_groupIdx.push_back(0);

	while (d!= traf.end())
	{
		if ((m_groups[d] != c_grp) || (m_attMat[d] != c_mat))
		{
			c_sg++;

			if (m_groups[d] != c_grp)
			{
				m_groupNbSub.push_back(c_sg-m_groupFirstSub.back());
				m_groupFirstSub.push_back(c_sg);
			}

			c_grp = m_groups[d];
			c_mat = m_attMat[d];
			m_sgMat.push_back(c_mat);

			if (m_tagG != 0)
				m_groupIdx.push_back(c_grp);
			else
				m_groupIdx.push_back(0);

			group_faces.resize(c_sg+1);
		}

		group_faces[c_sg].push_back(d);
		d = traf.next();
	}

	m_groupNbSub.push_back(c_sg+1-m_groupFirstSub.back()); // nb sub-group of last group

	// merging same material sub-groups of same group
	for (unsigned int g=0; g<m_groupNbSub.size(); ++g)
	{
		unsigned int fsg = m_groupFirstSub[g];
		unsigned int lsg = m_groupFirstSub[g]+m_groupNbSub[g]-1;

		for (unsigned int s=fsg; s<=lsg; ++s)
		{
			if (m_sgMat[s] != 0xffffffff)
			{
				for (unsigned ss=s+1; ss<=lsg; ++ss)
				{
					if (m_sgMat[ss] == m_sgMat[s])
					{
						group_faces[s].insert(group_faces[s].end(),group_faces[ss].begin(),group_faces[ss].end());
						group_faces[ss].clear();
						m_sgMat[ss] = 0xffffffff;
					}
				}
			}
		}
	}

	// compact group_faces/m_groupIdx/m_sgMat
	unsigned int outSg=0;
	for (unsigned int inSg=0; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_sgMat[inSg] != 0xffffffff)
		{
			if (outSg != inSg)
			{
				group_faces[outSg].swap(group_faces[inSg]);
				m_groupIdx[outSg] = m_groupIdx[inSg];
				m_sgMat[outSg] = m_sgMat[inSg];
			}
			outSg++;
		}
	}
	group_faces.resize(outSg);
	m_groupIdx.resize(outSg);
	m_sgMat.resize(outSg);

	// recreate m_groupFirstSub & m_groupNbSub
	unsigned int outGr=0;
	m_groupFirstSub[0] = m_groupIdx[0];
	for (unsigned int inSg=1; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_groupIdx[inSg] != m_groupIdx[inSg-1])
		{
			m_groupNbSub[outGr] = inSg - m_groupFirstSub[outGr];
			outGr++;
			m_groupFirstSub[outGr] = inSg;
		}
	}
	m_groupNbSub[outGr+1] = m_sgMat.size() - m_groupNbSub[outGr];

	// now create VBOs

	std::vector<Geom::Vec3f> posBuff;
	std::vector<Geom::Vec2f> TCBuff;
	posBuff.reserve(16384);
	TCBuff.reserve(16384);

	unsigned int firstIndex = 0;

	unsigned int sz = group_faces.size();
	m_beginIndices.resize(sz);
	m_nbIndices.resize(sz);
	m_groupIdx.resize(sz);

	for (unsigned int g=0; g<sz; ++g)
	{
		unsigned int nbtris = 0;
		std::vector<Dart>& traf = group_faces[g];

		if (m_tagG != 0)
			m_groupIdx[g] = m_groups[traf.front()];
		else
			m_groupIdx[g]=0;

		for (std::vector<Dart>::iterator id=traf.begin(); id!= traf.end(); ++id)
		{
			Dart d = *id;

			Dart e = m_map.phi1(d);
			Dart f = m_map.phi1(e);
			do
			{
				posBuff.push_back(m_positions[d]);

				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[d]);
				else
					TCBuff.push_back(m_texCoords[d]);


				posBuff.push_back(m_positions[e]);

				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[e]);
				else
					TCBuff.push_back(m_texCoords[e]);


				posBuff.push_back(m_positions[f]);

				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[f]);
				else
					TCBuff.push_back(m_texCoords[f]);


				e=f;
				f = m_map.phi1(e);
				nbtris++;
			}while (f!=d);
		}
		m_beginIndices[g] = firstIndex;
		m_nbIndices[g] = 3*nbtris;

		firstIndex += 3*nbtris;
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();

	texcoordVBO->setDataSize(2);
	texcoordVBO->allocate(TCBuff.size());
	Geom::Vec2f* ptrTC = reinterpret_cast<Geom::Vec2f*>(texcoordVBO->lockPtr());
	memcpy(ptrTC,&TCBuff[0],TCBuff.size()*sizeof(Geom::Vec2f));
	texcoordVBO->releasePtr();

	// compute BBs ( TO DO: a method ?)
	computeBB(posBuff);

	return true;
}

template <typename PFP>
bool OBJModel<PFP>::createGroupMatVBO_PN( Utils::VBO* positionVBO,
											Utils::VBO* normalVBO)
{
	m_beginIndices.clear();
	m_nbIndices.clear();

	if (!m_normals.isValid())
	{
		CGoGNerr << "no normal attribute "<< CGoGNendl;
		return false;
	}

	std::vector< std::vector<Dart> > group_faces; //(m_materialNames.size());
	group_faces.reserve(16384);
	m_groupIdx.reserve(16384);
	m_sgMat.reserve(16384);

	unsigned int c_sg=0;
	group_faces.resize(1);

	TraversorF<typename PFP::MAP> traf(m_map);
	Dart d=traf.begin();

	unsigned int c_grp = m_groups[d];
	unsigned int c_mat = m_attMat[d];
	m_sgMat.push_back(c_mat);

	m_groupFirstSub.push_back(0);

	if (m_tagG != 0)
		m_groupIdx.push_back(c_grp);
	else
		m_groupIdx.push_back(0);

	while (d!= traf.end())
	{
		if ((m_groups[d] != c_grp) || (m_attMat[d] != c_mat))
		{
			c_sg++;

			if (m_groups[d] != c_grp)
			{
				m_groupNbSub.push_back(c_sg-m_groupFirstSub.back());
				m_groupFirstSub.push_back(c_sg);
			}

			c_grp = m_groups[d];
			c_mat = m_attMat[d];
			m_sgMat.push_back(c_mat);

			if (m_tagG != 0)
				m_groupIdx.push_back(c_grp);
			else
				m_groupIdx.push_back(0);

			group_faces.resize(c_sg+1);
		}

		group_faces[c_sg].push_back(d);
		d = traf.next();
	}

	m_groupNbSub.push_back(c_sg+1-m_groupFirstSub.back()); // nb sub-group of last group

	// merging same material sub-groups of same group
	for (unsigned int g=0; g<m_groupNbSub.size(); ++g)
	{
		unsigned int fsg = m_groupFirstSub[g];
		unsigned int lsg = m_groupFirstSub[g]+m_groupNbSub[g]-1;

		for (unsigned int s=fsg; s<=lsg; ++s)
		{
			if (m_sgMat[s] != 0xffffffff)
			{
				for (unsigned ss=s+1; ss<=lsg; ++ss)
				{
					if (m_sgMat[ss] == m_sgMat[s])
					{
						group_faces[s].insert(group_faces[s].end(),group_faces[ss].begin(),group_faces[ss].end());
						group_faces[ss].clear();
						m_sgMat[ss] = 0xffffffff;
					}
				}
			}
		}
	}

	// compact group_faces/m_groupIdx/m_sgMat
	unsigned int outSg=0;
	for (unsigned int inSg=0; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_sgMat[inSg] != 0xffffffff)
		{
			if (outSg != inSg)
			{
				group_faces[outSg].swap(group_faces[inSg]);
				m_groupIdx[outSg] = m_groupIdx[inSg];
				m_sgMat[outSg] = m_sgMat[inSg];
			}
			outSg++;
		}
	}
	group_faces.resize(outSg);
	m_groupIdx.resize(outSg);
	m_sgMat.resize(outSg);

	// recreate m_groupFirstSub & m_groupNbSub
	unsigned int outGr=0;
	m_groupFirstSub[0] = m_groupIdx[0];
	for (unsigned int inSg=1; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_groupIdx[inSg] != m_groupIdx[inSg-1])
		{
			m_groupNbSub[outGr] = inSg - m_groupFirstSub[outGr];
			outGr++;
			m_groupFirstSub[outGr] = inSg;
		}
	}
	m_groupNbSub[outGr+1] = m_sgMat.size() - m_groupNbSub[outGr];

	// now create VBOs

	std::vector<Geom::Vec3f> posBuff;
	std::vector<Geom::Vec3f> normalBuff;
	posBuff.reserve(16384);
	normalBuff.reserve(16384);


	unsigned int firstIndex = 0;

	unsigned int sz = group_faces.size();
	m_beginIndices.resize(sz);
	m_nbIndices.resize(sz);
	m_groupIdx.resize(sz);

	for (unsigned int g=0; g<sz; ++g)
	{
		unsigned int nbtris = 0;
		std::vector<Dart>& traf = group_faces[g];

		if (m_tagG != 0)
			m_groupIdx[g] = m_groups[traf.front()];
		else
			m_groupIdx[g]=0;

		for (std::vector<Dart>::iterator id=traf.begin(); id!= traf.end(); ++id)
		{
			Dart d = *id;

			Dart e = m_map.phi1(d);
			Dart f = m_map.phi1(e);
			do
			{
				posBuff.push_back(m_positions[d]);

				if (hasNormals())
					normalBuff.push_back(m_normalsF[d]);
				else
					normalBuff.push_back(m_normals[d]);


				posBuff.push_back(m_positions[e]);

				if (hasNormals())
					normalBuff.push_back(m_normalsF[e]);
				else
					normalBuff.push_back(m_normals[e]);


				posBuff.push_back(m_positions[f]);

				if (hasNormals())
					normalBuff.push_back(m_normalsF[f]);
				else
					normalBuff.push_back(m_normals[f]);


				e=f;
				f = m_map.phi1(e);
				nbtris++;
			}while (f!=d);
		}
		m_beginIndices[g] = firstIndex;
		m_nbIndices[g] = 3*nbtris;

		firstIndex += 3*nbtris;
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();

	normalVBO->setDataSize(3);
	normalVBO->allocate(normalBuff.size());
	Geom::Vec3f* ptrNormal = reinterpret_cast<Geom::Vec3f*>(normalVBO->lockPtr());
	memcpy(ptrNormal, &normalBuff[0], normalBuff.size()*sizeof(Geom::Vec3f));
	normalVBO->releasePtr();

	// compute BBs ( TO DO: a method ?)
	computeBB(posBuff);

	return true;
}

template <typename PFP>
bool OBJModel<PFP>::createGroupMatVBO_PTN(
	Utils::VBO* positionVBO,
	Utils::VBO* texcoordVBO,
	Utils::VBO* normalVBO)
{
	m_beginIndices.clear();
	m_nbIndices.clear();

	if (!m_normals.isValid())
	{
		CGoGNerr << "no normal attribute "<< CGoGNendl;
		return false;
	}
	if (!m_texCoords.isValid())
	{
		CGoGNerr << "no tex coords attribute "<< CGoGNendl;
		return false;
	}

	std::vector< std::vector<Dart> > group_faces; //(m_materialNames.size());
	group_faces.reserve(16384);
	m_groupIdx.reserve(16384);
	m_sgMat.reserve(16384);

	unsigned int c_sg=0;
	group_faces.resize(1);

	TraversorF<typename PFP::MAP> traf(m_map);
	Dart d=traf.begin();

	unsigned int c_grp = m_groups[d];
	unsigned int c_mat = m_attMat[d];
	m_sgMat.push_back(c_mat);

	m_groupFirstSub.push_back(0);

	if (m_tagG != 0)
		m_groupIdx.push_back(c_grp);
	else
		m_groupIdx.push_back(0);

	while (d!= traf.end())
	{
		if ((m_groups[d] != c_grp) || (m_attMat[d] != c_mat))
		{
			c_sg++;

			if (m_groups[d] != c_grp)
			{
				m_groupNbSub.push_back(c_sg-m_groupFirstSub.back());
				m_groupFirstSub.push_back(c_sg);
			}

			c_grp = m_groups[d];
			c_mat = m_attMat[d];
			m_sgMat.push_back(c_mat);

			if (m_tagG != 0)
				m_groupIdx.push_back(c_grp);
			else
				m_groupIdx.push_back(0);

			group_faces.resize(c_sg+1);
		}

		group_faces[c_sg].push_back(d);
		d = traf.next();
	}

	m_groupNbSub.push_back(c_sg+1-m_groupFirstSub.back()); // nb sub-group of last group

	// merging same material sub-groups of same group
	for (unsigned int g=0; g<m_groupNbSub.size(); ++g)
	{
		unsigned int fsg = m_groupFirstSub[g];
		unsigned int lsg = m_groupFirstSub[g]+m_groupNbSub[g]-1;

		for (unsigned int s=fsg; s<=lsg; ++s)
		{
			if (m_sgMat[s] != 0xffffffff)
			{
				for (unsigned ss=s+1; ss<=lsg; ++ss)
				{
					if (m_sgMat[ss] == m_sgMat[s])
					{
						group_faces[s].insert(group_faces[s].end(),group_faces[ss].begin(),group_faces[ss].end());
						group_faces[ss].clear();
						m_sgMat[ss] = 0xffffffff;
					}
				}
			}
		}
	}

	// compact group_faces/m_groupIdx/m_sgMat
	unsigned int outSg=0;
	for (unsigned int inSg=0; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_sgMat[inSg] != 0xffffffff)
		{
			if (outSg != inSg)
			{
				group_faces[outSg].swap(group_faces[inSg]);
				m_groupIdx[outSg] = m_groupIdx[inSg];
				m_sgMat[outSg] = m_sgMat[inSg];
			}
			outSg++;
		}
	}
	group_faces.resize(outSg);
	m_groupIdx.resize(outSg);
	m_sgMat.resize(outSg);

	// recreate m_groupFirstSub & m_groupNbSub
	unsigned int outGr=0;
	m_groupFirstSub[0] = m_groupIdx[0];
	for (unsigned int inSg=1; inSg<m_sgMat.size(); ++inSg)
	{
		if (m_groupIdx[inSg] != m_groupIdx[inSg-1])
		{
			m_groupNbSub[outGr] = inSg - m_groupFirstSub[outGr];
			outGr++;
			m_groupFirstSub[outGr] = inSg;
		}
	}
	m_groupNbSub[outGr+1] = m_sgMat.size() - m_groupNbSub[outGr];

	// now create VBOs

	std::vector<Geom::Vec3f> posBuff;
	std::vector<Geom::Vec2f> TCBuff;
	std::vector<Geom::Vec3f> normalBuff;
	posBuff.reserve(16384);
	TCBuff.reserve(16384);
	normalBuff.reserve(16384);

	unsigned int firstIndex = 0;

	unsigned int sz = group_faces.size();
	m_beginIndices.resize(sz);
	m_nbIndices.resize(sz);
	m_groupIdx.resize(sz);

	for (unsigned int g=0; g<sz; ++g)
	{
		unsigned int nbtris = 0;
		std::vector<Dart>& traf = group_faces[g];

		if (m_tagG != 0)
			m_groupIdx[g] = m_groups[traf.front()];
		else
			m_groupIdx[g]=0;

		for (std::vector<Dart>::iterator id=traf.begin(); id!= traf.end(); ++id)
		{
			Dart d = *id;

			Dart e = m_map.phi1(d);
			Dart f = m_map.phi1(e);
			do
			{
				posBuff.push_back(m_positions[d]);

				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[d]);
				else
					TCBuff.push_back(m_texCoords[d]);
				if (hasNormals())
					normalBuff.push_back(m_normalsF[d]);
				else
					normalBuff.push_back(m_normals[d]);


				posBuff.push_back(m_positions[e]);

				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[e]);
				else
					TCBuff.push_back(m_texCoords[e]);
				if (hasNormals())
					normalBuff.push_back(m_normalsF[e]);
				else
					normalBuff.push_back(m_normals[e]);


				posBuff.push_back(m_positions[f]);

				if (hasTexCoords())
					TCBuff.push_back(m_texCoordsF[f]);
				else
					TCBuff.push_back(m_texCoords[f]);
				if (hasNormals())
					normalBuff.push_back(m_normalsF[f]);
				else
					normalBuff.push_back(m_normals[f]);


				e=f;
				f = m_map.phi1(e);
				nbtris++;
			}while (f!=d);
		}
		m_beginIndices[g] = firstIndex;
		m_nbIndices[g] = 3*nbtris;

		firstIndex += 3*nbtris;
	}

	positionVBO->setDataSize(3);
	positionVBO->allocate(posBuff.size());
	Geom::Vec3f* ptrPos = reinterpret_cast<Geom::Vec3f*>(positionVBO->lockPtr());
	memcpy(ptrPos,&posBuff[0],posBuff.size()*sizeof(Geom::Vec3f));
	positionVBO->releasePtr();

	texcoordVBO->setDataSize(2);
	texcoordVBO->allocate(TCBuff.size());
	Geom::Vec2f* ptrTC = reinterpret_cast<Geom::Vec2f*>(texcoordVBO->lockPtr());
	memcpy(ptrTC,&TCBuff[0],TCBuff.size()*sizeof(Geom::Vec2f));
	texcoordVBO->releasePtr();

	normalVBO->setDataSize(3);
	normalVBO->allocate(normalBuff.size());
	Geom::Vec3f* ptrNormal = reinterpret_cast<Geom::Vec3f*>(normalVBO->lockPtr());
	memcpy(ptrNormal, &normalBuff[0], normalBuff.size()*sizeof(Geom::Vec3f));
	normalVBO->releasePtr();


	// compute BBs ( TO DO: a method ?)
	computeBB(posBuff);

	return true;
}

template <typename PFP>
void OBJModel<PFP>::computeBB(const std::vector<Geom::Vec3f>& pos)
{
	m_groupBBs.resize(nbObjGroups());

	for (unsigned int i=0; i<nbObjGroups(); ++i )
	{
		Geom::BoundingBox<VEC3>& bb = m_groupBBs[i];
		bb.reset();

		unsigned int begInd = beginIndex(i,0);
		unsigned int endInd = begInd;
		for (unsigned int j= 0; j< nbSubGroup(i); ++j)
			endInd += nbIndices(i,j);

		for (unsigned int j=begInd; j<endInd; ++j )
		{
			bb.addPoint(pos[j]);
		}
	}
}

template <typename PFP>
bool OBJModel<PFP>::import( const std::string& filename, std::vector<std::string>& attrNames)
{
	attrNames.clear();
	// open file
	std::ifstream fp(filename.c_str()/*, std::ios::binary*/);
	if (!fp.good())
	{
		CGoGNerr << "Unable to open file " << filename << CGoGNendl;
		return false;
	}

	std::string ligne;
	std::string tag;
	do
	{
		std::getline (fp, ligne);
		std::stringstream oss(ligne);
		oss >> tag;
		if (tag == "v")
			m_tagV++;
		if (tag == "vn")
			m_tagVN++;
		if (tag == "vt")
			m_tagVT++;
		if ((tag == "g") || (tag == "o"))
			m_tagG++;
		if (tag == "f")
			m_tagF++;

		if (tag == "mtllib")
		{
			unsigned found = filename.find_last_of("/\\");
			std::string mtfn;
			oss >> mtfn;
			m_matPath = filename.substr(0,found) + "/";
			m_matFileName = m_matPath + mtfn;
		}
		tag.clear();

	} while (!fp.eof());

	m_positions =  m_map.template getAttribute<VEC3, VERTEX, MAP>("position") ;
	if (!m_positions.isValid())
		m_positions = m_map.template addAttribute<VEC3, VERTEX, MAP>("position") ;
	attrNames.push_back(m_positions.name()) ;

	m_texCoords =  m_map.template getAttribute<VEC2, VERTEX, MAP>("texCoord") ;
	if (!m_texCoords.isValid())
		m_texCoords = m_map.template addAttribute<VEC2, VERTEX, MAP>("texCoord") ;
	attrNames.push_back(m_texCoords.name()) ;

	if (m_tagVT != 0)
	{
		m_texCoordsF =  m_map.template getAttribute<VEC2, VERTEX1, MAP>("texCoordF") ;
		if (!m_texCoordsF.isValid())
			m_texCoordsF = m_map.template addAttribute<VEC2, VERTEX1, MAP>("texCoordF") ;
	}

	m_normals =  m_map.template getAttribute<VEC3, VERTEX, MAP>("normal") ;
	if (!m_normals.isValid())
		m_normals = m_map.template addAttribute<VEC3, VERTEX, MAP>("normal") ;
	attrNames.push_back(m_normals.name()) ;

	if (m_tagVN != 0)
	{
		m_normalsF =  m_map.template getAttribute<VEC3, VERTEX1, MAP>("normalF") ;
		if (!m_normalsF.isValid())
			m_normalsF = m_map.template addAttribute<VEC3, VERTEX1, MAP>("normalF") ;
	}
	
//	if (m_tagG != 0) always use group even if not in the file
	{
		m_groups =  m_map.template getAttribute<unsigned int, FACE, MAP>("groups") ;
		if (!m_groups.isValid())
			m_groups = m_map.template addAttribute<unsigned int, FACE, MAP>("groups") ;
		attrNames.push_back(m_groups.name()) ;
	}
	
	m_attMat =  m_map.template getAttribute<unsigned int, FACE, MAP>("material") ;
	if (!m_attMat.isValid())
		m_attMat = m_map.template addAttribute<unsigned int, FACE, MAP>("material") ;
	attrNames.push_back(m_attMat.name()) ;

	AttributeContainer& container = m_map.template getAttributeContainer<VERTEX>() ;

	fp.close();
	fp.clear();
	fp.open(filename.c_str());

	std::vector<VEC3> normalsBuffer;
	normalsBuffer.reserve(m_tagVN);
	
	std::vector<VEC2> texCoordsBuffer;
	texCoordsBuffer.reserve(m_tagVT);
	
	std::vector<unsigned int> verticesID;
	verticesID.reserve(m_tagV);
	
	std::vector<unsigned int> normalsID;
	normalsID.reserve(m_tagV);
	std::vector<unsigned int> texCoordsID;
	texCoordsID.reserve(m_tagV);
	
	std::vector<unsigned int> localIndices;
	localIndices.reserve(64*3);

//	unsigned int vemb = EMBNULL;
//	auto fsetemb = [&] (Dart d) { m_map.template initDartEmbedding<VERTEX>(d, vemb); };

	VertexAutoAttribute< NoTypeNameAttribute< std::vector<Dart> >, MAP> vecDartsPerVertex(m_map, "incidents");
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<unsigned int> >, MAP> vecNormIndPerVertex(m_map, "incidentsN");
	VertexAutoAttribute< NoTypeNameAttribute< std::vector<unsigned int> >, MAP> vecTCIndPerVertex(m_map, "incidentsTC");

	int currentGroup = -1;
	unsigned int currentMat = 0;
	unsigned int nextMat = 0;

	DartMarkerNoUnmark<MAP> mk(m_map) ;
	unsigned int i = 0;
	fp >> tag;
	std::getline(fp, ligne);
	do
	{
		if (tag == std::string("v"))
		{
			std::stringstream oss(ligne);

			float x,y,z;
			oss >> x;
			oss >> y;
			oss >> z;

			VEC3 pos(x,y,z);

			unsigned int id = container.insertLine();
			m_positions[id] = pos;

			verticesID.push_back(id);
			i++;
		}

		if (tag == std::string("vn"))
		{
			std::stringstream oss(ligne);

			VEC3 norm;
			oss >> norm[0];
			oss >> norm[1];
			oss >> norm[2];
			normalsBuffer.push_back(norm);
		}

		if (tag == std::string("vt"))
		{
			std::stringstream oss(ligne);
			VEC2 tc;
			oss >> tc[0];
			oss >> tc[1];
			texCoordsBuffer.push_back(tc);
		}


		if (tag == std::string("usemtl"))
		{
			std::stringstream oss(ligne);
			std::string matName;
			oss >> matName;
			std::map<std::string, int>::iterator it = m_materialNames.find(matName);

			if (it==m_materialNames.end())
			{
				m_materialNames.insert(std::pair<std::string,int>(matName,nextMat));
				currentMat = nextMat++;
//				std::cout << "New Material Name = "<< matName << "  index = "<< currentMat << std::endl;

			}
			else
			{
				currentMat = it->second;
//				std::cout << "Using Material Name = "<<  matName << "  index = "<< currentMat << std::endl;
			}
		}

		if ( (tag == std::string("g")) || (tag == std::string("o")) )
		{
			std::string name = ligne.substr(1);
			if (name[name.size()-1]==13)
				name = name.substr(0,name.size()-1);

			m_groupNames.push_back(name);
			currentGroup++;
		}

		if (tag == std::string("f"))
		{
			std::stringstream oss(ligne);

			short nbe = readObjLine(oss,localIndices);

			Dart d = m_map.newFace(nbe, false);
			if (m_tagG!=0)
			{
				if (currentGroup == -1)
					currentGroup = 0;
				m_groups[d] = currentGroup;
			}

			m_attMat[d] = currentMat;
			
			for (short j = 0; j < nbe; ++j)
			{
				unsigned int vemb = localIndices[3*j]-1;		// get embedding
				m_map.template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { m_map.template initDartEmbedding<VERTEX>(dd, vemb); });
				mk.mark(d) ;								// mark on the fly to unmark on second loop
				vecDartsPerVertex[vemb].push_back(d);		// store incident darts for fast adjacency reconstruction
				vecTCIndPerVertex[vemb].push_back(localIndices[3*j+1]-1);
				vecNormIndPerVertex[vemb].push_back(localIndices[3*j+2]-1);
				d = m_map.phi1(d);
			}
		}
		fp >> tag;
		std::getline(fp, ligne);
	} while (!fp.eof());
	fp.close ();

	// reconstruct neighbourhood
	unsigned int nbBoundaryEdges = 0;
	for (Dart d = m_map.begin(); d != m_map.end(); m_map.next(d))
	{
		if (mk.isMarked(d))
		{
			// darts incident to end vertex of edge
			std::vector<Dart>& vec = vecDartsPerVertex[m_map.phi1(d)];

			unsigned int embd = m_map.template getEmbedding<VERTEX>(d);
			Dart good_dart = NIL;
			for (typename std::vector<Dart>::iterator it = vec.begin(); it != vec.end() && good_dart == NIL; ++it)
			{
				if (m_map.template getEmbedding<VERTEX>(m_map.phi1(*it)) == embd)
					good_dart = *it;
			}

			if (good_dart != NIL)
			{
				if (good_dart == m_map.phi2(good_dart) && (d == m_map.phi2(d)))
					m_map.sewFaces(d, good_dart, false);
				else
					m_dirtyEdges.mark(d);
				mk.template unmarkOrbit<EDGE>(d);
			}
			else
			{
				mk.unmark(d);
				m_dirtyEdges.mark(d);
				++nbBoundaryEdges;
			}
		}
	}


	// A SIMPLIFIER ???

	TraversorV<MAP> tra(m_map);
	for (Dart d = tra.begin(); d != tra.end(); d = tra.next())
	{
		std::vector<Dart>& vec			= vecDartsPerVertex[d];
		std::vector<unsigned int>& vtc	= vecTCIndPerVertex[d];
		std::vector<unsigned int>& vn	= vecNormIndPerVertex[d];

		// test if normal vertex or multi-attrib vertex
		unsigned int nb = vtc.size();
		bool same=true;
		for (unsigned int j=1; (j<nb)&&(same); ++j)
		{
			if ( (vtc[j] != vtc[0]) || (vn[j] != vn[0]) )
				same = false;
		}

		// if not all faces the same embedding
//		if (!same)
//		{
			for (unsigned int j=0; j<nb; ++j)
			{
				Dart e = vec[j];
				if (m_tagVT)
				{
					if (vecTCIndPerVertex[e][j] != 0xffffffff)
						m_texCoordsF[e] = texCoordsBuffer[ vecTCIndPerVertex[e][j] ];
					else
						m_texCoordsF[e] = Geom::Vec2f(0);
				}
				if (m_tagVN)
				{
					if (vecNormIndPerVertex[e][j] != 0xffffffff)
						m_normalsF[e] = normalsBuffer[ vecNormIndPerVertex[e][j] ];
					else
						m_normalsF[e] = VEC3(0);
				}
			}
			m_specialVertices.mark(d);
//		}
//		else
//		{
//			if (tagVT)
//				m_texCoords[d] = texCoordsBuffer[ vecTCIndPerVertex[d][0] ];
//			if (tagVN)
//				m_normals[d] = normalsBuffer[ vecNormIndPerVertex[d][0] ];
//			m_specialVertices.unmark(d);
//		}
	}

	readMaterials();

	return true;
}

template <typename PFP>
unsigned int OBJModel<PFP>::storeFacesOfGroup(unsigned int groupId, std::vector<Dart>& dartFaces)
{
	unsigned int nb=dartFaces.size();

	TraversorF<typename PFP::MAP> traf(m_map);
	for (Dart d=traf.begin(); d!= traf.end(); d = traf.next())
	{
		if (m_groups[d] == groupId)
		{
			dartFaces.push_back(d);
		}
	}

	return dartFaces.size()-nb;
}

} // namespace Import

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
