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
#include "Geometry/bounding_box.h"

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
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::MAP MAP;
	typedef Geom::Vec2f VEC2;

protected:
	MAP& m_map;

	// infof of sub-groups (group/material)
	std::vector<unsigned int> m_beginIndices;
	std::vector<unsigned int> m_nbIndices;
	std::vector<unsigned int> m_groupIdx;
	std::vector<unsigned int> m_sgMat;

	std::vector<unsigned int> m_objGroups;


	std::vector<unsigned int> m_groupFirstSub;
	std::vector<unsigned int> m_groupNbSub;

	std::vector<std::string> m_groupNames;
	std::vector< Geom::BoundingBox<VEC3> > m_groupBBs;

	unsigned int m_maxTextureSize;

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

	void computeBB(const std::vector<Geom::Vec3f>& pos);

public:

	/// marker for special vertices (with several normals & tex coords)
	CellMarker<MAP, VERTEX> m_specialVertices;

	/// marker for darts with phi2 reconstruction face
	DartMarker<MAP> m_dirtyEdges;

	/// Face Attribute for group ID storage
	FaceAttribute<unsigned int, MAP> m_groups;
	FaceAttribute<unsigned int, MAP> m_attMat;

	/// Vertex Attribute Handlers
	VertexAttribute<VEC3, MAP> m_positions;
	VertexAttribute<VEC3, MAP> m_normals;
	VertexAttribute<Geom::Vec2f, MAP> m_texCoords;

	/// Vertex of face Attribute Handlers
	AttributeHandler<VEC3, VERTEX1, MAP> m_normalsF;
	AttributeHandler<Geom::Vec2f, VERTEX1, MAP> m_texCoordsF;

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
	void setPositionAttribute(VertexAttribute<Geom::Vec3f, MAP> position);

	/**
	 * @brief set position attribute
	 * @param position attribute
	 */
	void setNormalAttribute(VertexAttribute<Geom::Vec3f, MAP> normal);

	/**
	 * @brief set texture coordinate attribute
	 * @param texcoord attribute
	 */
	void setTexCoordAttribute(VertexAttribute<Geom::Vec2f, MAP>texcoord);

	bool hasTexCoords() const { return m_tagVT != 0; }

	bool hasNormals() const { return m_tagVN != 0; }

	bool hasGroups() const { return m_tagG != 0; }

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
	VEC3 getNormal(Dart d);

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
	VEC3 getPosition(Dart d);

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
//	unsigned int nbMatGroups() { return m_beginIndices.size(); }

	/**
	 * @brief number of sub-group in group
	 * @param grp id of group
	 * @return
	 */
	inline unsigned int nbSubGroup(unsigned int grp) const { return m_groupNbSub[grp];}

	/**
	 * @brief get the begin index of a sub-group in VBOs (for glDrawArrays)
	 * @param i id of group
	 * @param j id of subgroup in group
	 * @return begin index
	 */
	inline unsigned int beginIndex(unsigned int i, unsigned int j) const { return m_beginIndices[ m_groupFirstSub[i]+j ]; }

	/**
	 * @brief get the number of indices of a sub-group in VBOs (for glDrawArrays)
	 * @param i id of group
	 * @param j id of subgroup in group
	 * @return number of indices
	 */
	inline unsigned int nbIndices(unsigned int i, unsigned int j) const { return m_nbIndices[ m_groupFirstSub[i]+j ]; }

	/**
	 * @brief material id of a sub-group
	 * @param i id of group
	 * @param j id of subgroup in group
	 * @return id of material
	 */
	inline unsigned int materialIdOf(unsigned int i, unsigned int j) const { return m_sgMat[ m_groupFirstSub[i]+j ]; }

	/**
	 * @brief material of a sub-group
	 * @param i id of group
	 * @param j id of subgroup in group
	 * @return material ptr
	 */
	inline const MaterialOBJ* materialOf(unsigned int i, unsigned int j) const { return m_materials[materialIdOf(i,j)]; }

	/**
	 * @brief get the id of group in OBJ file os sub-group
	 * @param i id of sub-group
	 * @return obj group index
	 */
	inline unsigned int groupIdx(unsigned int i) const { return m_groupIdx[i]; }

	/**
	 * @brief get the number of groups in OBJ file
	 * @return number of groups
	 */
	unsigned int nbObjGroups() { return m_groupFirstSub.size(); }

	/**
	 * @brief get the index of first group mat of obj
	 * @param i id of obj group
	 * @return id of first group mat
	 */
//	unsigned int objGroup(unsigned int i) const { return m_objGroups[i]; }


	const Geom::BoundingBox<VEC3>& getGroupBB(unsigned int i) const { return m_groupBBs[i]; }

	Geom::BoundingBox<VEC3>& getGroupBB(unsigned int i) { return m_groupBBs[i];}

	const std::string& objGroupName(unsigned int i) const { return m_groupNames[i];}

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

	/**
	 * @brief add a dart by each face of group in a vector
	 * @param groupId the group to add
	 * @param dartFaces the vector in which we want to add
	 * @return the number of faces added.
	 */
	unsigned int storeFacesOfGroup(unsigned int groupId, std::vector<Dart>& dartFaces);
};

} // namespace Import

} // namepsace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Import/importObjTex.hpp"

#endif // IMPORTOBJTEX_H
