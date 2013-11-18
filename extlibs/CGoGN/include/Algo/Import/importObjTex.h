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
#ifndef IMPORTOBJTEX_H
#define IMPORTOBJTEX_H

//#include "Topology/generic/mapBrowser.h"
#include "Container/containerBrowser.h"
#include "Topology/generic/cellmarker.h"
#include "Utils/textures.h"

namespace CGoGN
{
namespace Algo
{
namespace Surface
{
namespace Import
{


class MaterialOBJ
{
public:
	MaterialOBJ():
		illuminationModel(0),
		ambiantColor(0.0f,0.0f,0.0f),
		diffuseColor(0.0f,0.0f,0.0f),
		specularColor(0.0f,0.0f,0.0f),
		shininess(100.0f),
		transparentFilter(0.0f,0.0f,0.0f),
		transparency(0.0f),
		textureDiffuse(NULL)
	{}

	~MaterialOBJ()
	{
		if (textureDiffuse!=NULL)
			delete textureDiffuse;
	}

	std::string name;
	int illuminationModel; // 0 Diffu / 1 D+Ambiant / 3 A+D+Specular
	Geom::Vec3f ambiantColor;
	Geom::Vec3f diffuseColor;
	Geom::Vec3f specularColor;
	float shininess;
	Geom::Vec3f transparentFilter;
	float transparency;
	Utils::Texture<2,Geom::Vec3uc>* textureDiffuse;

//	static bool compare(MaterialOBJ* m1, MaterialOBJ* m2)
//	{
//		if (m1->textureDiffuse > m2->textureDiffuse)
//			return true;
//		if (m1->textureDiffuse == m2->textureDiffuse)
//			return m1->illuminationModel > m2->illuminationModel;
//		return false;
//	}
};



template <typename PFP>
class OBJModel
{
	typedef  typename PFP::VEC3 VEC3;
	typedef Geom::Vec2f VEC2;

protected:
	typename PFP::MAP& m_map;

	std::vector<unsigned int> m_beginIndices;
	std::vector<unsigned int> m_nbIndices;

	unsigned int m_maxTextureSize;

	/// vector of group name
//	std::vector<std::string> m_groupNames;
//	std::vector<std::string> m_groupMaterialNames;
//	std::vector<int> m_groupMaterialID;

	/// map of material names -> group id
	std::map<std::string,int> m_materialNames;

	/// filename of mtllib
	std::string m_matFileName;

	/// path of mtllib
	std::string m_matPath;

	/// vector of material struct
	std::vector<MaterialOBJ*> m_materials;

	/// read face line with different indices v  v/t v//n v/t/n
	short readObjLine(std::stringstream& oss, std::vector<unsigned int>& indices);

	unsigned int m_tagV ;
	unsigned int m_tagVT ;
	unsigned int m_tagVN ;
	unsigned int m_tagG ;
	unsigned int m_tagF ;

public:

	/// marker for special vertices (with several normals & tex coords)
	CellMarker<VERTEX> m_specialVertices;

	/// marker for darts with phi2 reconstruction face
	DartMarker m_dirtyEdges;

	/// Face Attribute for group ID storage
	FaceAttribute<unsigned int> m_groups;
	FaceAttribute<unsigned int> m_attMat;

	/// Vertex Attribute Handlers
	VertexAttribute<VEC3> m_positions;
	VertexAttribute<VEC3> m_normals;
	VertexAttribute<Geom::Vec2f> m_texCoords;

	/// Vertex of face Attribute Handlers
	AttributeHandler<VEC3,VERTEX1> m_normalsF;
	AttributeHandler<Geom::Vec2f,VERTEX1> m_texCoordsF;


	/**
	 * @brief Constructeur
	 * @param map
	 */
	OBJModel(typename PFP::MAP& map);

	~OBJModel();

	/**
	 * @brief resize texture (at import) to max size
	 * @param mts max texture size in x & y
	 */
	void setMaxTextureSize(unsigned int mts);

	/**
	 * @brief set position attribute
	 * @param position attribute
	 */
	void setPositionAttribute(VertexAttribute<Geom::Vec3f> position);

	/**
	 * @brief set position attribute
	 * @param position attribute
	 */
	void setNormalAttribute(VertexAttribute<Geom::Vec3f> normal);

	/**
	 * @brief set texture coordinate attribute
	 * @param texcoord attribute
	 */
	void setTexCoordAttribute(VertexAttribute<Geom::Vec2f>texcoord);


	bool hasTexCoords() const { return m_tagVT!=0; }

	bool hasNormals() const { return m_tagVN!=0; }

	bool hasGroups() const { return m_tagG!=0; }

	/**
	 * @brief import
	 * @param filename
	 * @param attrNames
	 * @return
	 */
	bool import(const std::string& filename, std::vector<std::string>& attrNames);



	// Faire un handler ?
	/**
	 * @brief getNormal
	 * @param d
	 * @return
	 */
	typename PFP::VEC3 getNormal(Dart d);

	/**
	 * @brief getTexCoord
	 * @param d
	 * @return
	 */
	Geom::Vec2f getTexCoord(Dart d);

	/**
	 * @brief getPosition
	 * @param d
	 * @return
	 */
	typename PFP::VEC3 getPosition(Dart d);


	/**
	 * @brief Generate one browser per group
	 * @param browsers vector of MapBrowers representing the groups
	 * @return ok or not
	 */
	bool generateBrowsers(std::vector<ContainerBrowser*>& browsers);

	/**
	 * @brief getMaterialNames
	 * @return
	 */
	std::vector<std::string>& getMaterialNames();

	/**
	 * @brief getMaterialIndex
	 * @param name name of material
	 * @return index in generated vector of material by readMaterials
	 */
	unsigned int getMaterialIndex(const std::string& name) const;

	/**
	 * @brief read materials from files. Call after creating VBOs !!
	 * @param filename name of file
	 */
	void readMaterials(const std::string& filename="");

	/**
	 * @brief getMaterials
	 * @return the vector of MaterialObj*
	 */
	const std::vector<MaterialOBJ*>& getMaterials() const { return m_materials;}


	/**
	 * @brief nb group of indices created by createGroupMatVBO_XXX
	 * @return
	 */
	unsigned int nbMatGroups() { return m_beginIndices.size(); }

	/**
	 * @brief get the begin index of each group in VBOs (for glDrawArrays)
	 * @param i id of group
	 * @return begin index
	 */
	unsigned int beginIndex(unsigned int i) const { return m_beginIndices[i]; }

	/**
	 * @brief get the number of indices of each group in VBOs (for glDrawArrays)
	 * @param i id of group
	 * @return number of indices
	 */
	unsigned int nbIndices(unsigned int i) const { return m_nbIndices[i]; }

	/**
	 * @brief create simple VBO for separated triangles
	 * @param positionVBO
	 * @param normalVBO
	 * @return number of indices to draw
	 */
	unsigned int createSimpleVBO_P(Utils::VBO* positionVBO);

	/**
	 * @brief create simple VBO for separated triangles
	 * @param positionVBO
	 * @param texcoordVBO
	 * @return number of indices to draw
	 */
	unsigned int createSimpleVBO_PT(Utils::VBO* positionVBO, Utils::VBO* texcoordVBO);

	/**
	 * @brief create simple VBO for separated triangles
	 * @param positionVBO
	 * @param normalVBO
	 * @return number of indices to draw
	 */
	unsigned int createSimpleVBO_PN(Utils::VBO* positionVBO, Utils::VBO* normalVBO);


	/**
	 * @brief create simple VBO for separated triangles
	 * @param positionVBO
	 * @param texcoordVBO
	 * @param normalVBO
	 * @return number of indices to draw
	 */
	unsigned int createSimpleVBO_PTN(Utils::VBO* positionVBO, Utils::VBO* texcoordVBO, Utils::VBO* normalVBO);


	/**
	 * @brief create VBOs with group by material
	 * @param positionVBO
	 * @param beginIndices
	 * @param nbIndices
	 * @return
	 */
	bool createGroupMatVBO_P(Utils::VBO* positionVBO);
	/**
	 * @brief create VBOs with group by material
	 * @param positionVBO
	 * @param texcoordVBO
	 * @param beginIndices
	 * @param nbIndices
	 * @return
	 */
	bool createGroupMatVBO_PT(Utils::VBO* positionVBO, Utils::VBO* texcoordVBO);

	/**
	 * @brief create VBOs with group by material
	 * @param positionVBO
	 * @param normalVBO
	 * @param beginIndices
	 * @param nbIndices
	 * @return
	 */
	bool createGroupMatVBO_PN(Utils::VBO* positionVBO, Utils::VBO* normalVBO);

	/**
	 * @brief create VBOs with group by material
	 * @param positionVBO
	 * @param texcoordVBO
	 * @param normalVBO
	 * @param nbIndices
	 * @return
	 */
	bool createGroupMatVBO_PTN( Utils::VBO* positionVBO, Utils::VBO* texcoordVBO, Utils::VBO* normalVBO);


};


}
}
} // end namespaces
}

#include "importObjTex.hpp"

#endif // IMPORTOBJTEX_H
