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

#ifndef __SCENE_GRAPH_
#define __SCENE_GRAPH_

#include <string>
#include "Geometry/matrix.h"
#include "Algo/Render/GL2/mapRender.h"
#include "Utils/GLSLShader.h"


namespace CGoGN
{
namespace Utils
{
namespace SceneGraph
{

class Material_Node;

/**
 * Simple Node class
 */
class Node
{
protected:
	std::string m_label;
	unsigned int m_nbRefs;
	static Material_Node* m_current_material;
public:
	/**
	 * Constructor
	 */
	Node();

	/**
	 * Constructor
	 */
	Node(const std::string& lab);

	/**
	 * Destructor
	 */
	virtual ~Node();

	/**
	 * increment reference counter
	 */
	void ref();

	/**
	 * decrement reference counter
	 * @return true if counter is 0 and node can be destructed
	 */
	bool unref();

	/**
	 * get the label
	 * @return the label
	 */
	const std::string& getLabel() const;

	/**
	 * set the label
	 * @param lab label
	 */
	void setLabel(const std::string& lab);

	/**
	 * Rendering callback (used in traversal by renderGraph)
	 */
	virtual void render() = 0;
};

/**
 * Class describing OpenGL materials
 */
class Material_Node: public Node
{
protected:
	Geom::Vec4f m_diffuse;
	Geom::Vec4f m_specular;
	Geom::Vec4f m_ambient;
	float m_shininess;
	Geom::Vec4f m_color;
	bool m_has_diffuse;
	bool m_has_specular;
	bool m_has_ambient;
	bool m_has_shininess;
	bool m_has_color;
	Utils::GLSLShader* m_shader;
	bool m_disable_shader;
public:
	/**
	 * Constructor
	 */
	Material_Node();

	/**
	 * set diffuse color material
	 * @param v the color RGBA
	 */
	void setDiffuse(const Geom::Vec4f& v);

	/**
	 * set specular color material
	 * @param v the color RGBA
	 */
	void setSpecular(const Geom::Vec4f& v);

	/**
	 * set ambient color material
	 * @param v the color RGBA
	 */
	void setAmbient(const Geom::Vec4f& v);

	/**
	 * set shinineess (OGL)
	 * @param s the shininess
	 */
	void setShininess(float s);

	/**
	 * set color (for rendering with no lighting)
	 * @param v the color RGBA
	 */
	void setColor(const Geom::Vec4f& v);

	/**
	 * set the shader that will be used in children
	 */
	void setShader(Utils::GLSLShader* shader);

	/**
	 * get the shader
	 */
	Utils::GLSLShader* getShader() { return m_shader;}

	/**
	 * Explicitly disable shaders when traverse this node
	 */
	void setNoShader();

	/**
	 * Rendering callback (used in traversal by renderGraph)
	 */
	virtual void render();

	void apply();
};


/**
 * class of group node
 * contain a list of child, a transformation, and material
 */
class Group_Node: public Node
{
protected:
	std::list<Node*> m_children;

	Geom::Matrix44f* m_matrix_transfo;

	Material_Node* m_material;

public:
	/**
	 * Constructor
	 */
	Group_Node();

	/**
	 * Constructor
	 */
	Group_Node(const std::string& label);

	/**
	 * Destructor
	 */
	virtual ~Group_Node();

	/**
	 * get the children list
	 * @return the list (no modification allowed)
	 */
	const std::list<Node*>& getChildren() const;

	/**
	 * add a child
	 * @param child
	 */
	void addChild(Node* child);

	/**
	 * remove a child
	 * @param child
	 */
	void removeChild(Node* child);

	/**
	 * set transformation matrix (4x4)
	 * @param mat matrix (copy and stored here transposed for OGL)
	 */
	void setMatrixTransfo(const Geom::Matrix44f& mat);

	/**
	 * get the transformation matrix
	 * @ return a copy of the matrix (I if none)
	 */
	Geom::Matrix44f getMatrixTransfo() const;

	/**
	 * set material
	 * @param v the color RGBA
	 */
	void setMaterial(Material_Node* mat);

	/**
	 * Rendering callback (used in traversal by renderGraph)
	 */
	virtual void render();

};









class VBO_Node: public Node
{
protected:
	Algo::Render::GL2::MapRender* m_vbo;

	unsigned int m_primitives;
public:
	/**
	 * Constructor
	 */
	VBO_Node();

	/**
	 * Constructor
	 */
	VBO_Node(Algo::Render::GL2::MapRender* vbo);

	/**
	 * set the VBO render object ptr
	 * @param ptr on MapRender object (not released by destructor
	 */
	void setVBO(Algo::Render::GL2::MapRender* vbo);

	/**
	 * set the primitives to render
	 * @param p primtives (TRIANGLES, LINES, POINTS, FLAT_TRIANGLES-
	 */
	void setPrimitives(unsigned int p);

	/**
	 * set the primitives to render
	 * @param p primtives (TRIANGLES, LINES, POINTS, FLAT_TRIANGLES
	 */
	void unsetPrimitives(unsigned int p);

	/**
	 * Rendering callback (used in traversal by renderGraph)
	 */
	virtual void render();
};


/**
 * Render the graph of scene
 * @param node the root node of scene
 */
void renderGraph(Node* node);


/**
 * Erase the graph of scene
 * @param node the root node of scene
 */
void eraseGraph(Node* node);




}
}
}

#endif
