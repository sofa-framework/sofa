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
#include "Utils/scene_graph.h"

namespace CGoGN
{
namespace Utils
{
namespace SceneGraph
{

Material_Node* Node::m_current_material = NULL;


// NODE METHODS

Node::Node()
: m_label("SceneGraph::Node"),
  m_nbRefs(0)
{}


Node::~Node()
{}

Node::Node(const std::string& lab)
: m_label(lab),
  m_nbRefs(0)
{}

void Node::ref()
{
	m_nbRefs++;
}

bool Node::unref()
{
	m_nbRefs--;
	return m_nbRefs==0;
}


const std::string& Node::getLabel() const
{
	return m_label;
}

void Node::setLabel(const std::string& lab)
{
	m_label = lab ;
}



// GROUP_NODE METHODS

Group_Node::Group_Node()
:Node("SceneGraph::Group_Node"),
m_matrix_transfo(NULL),
m_material(NULL)
{}

Group_Node::Group_Node(const std::string& label)
: Node(label)
{}

Group_Node::~Group_Node()
{
	if (m_matrix_transfo != NULL)
		delete m_matrix_transfo;

	if (m_material != NULL)
		if (m_material->unref())
					delete m_material;


	for (std::list<Node*>::const_iterator it = m_children.begin(); it !=m_children.end(); ++it)
		if ( (*it)->unref())
			delete *it;
}

const std::list<Node*>& Group_Node::getChildren() const
{
	return m_children;
}

void Group_Node::addChild(Node* child)
{
	child->ref();
	m_children.push_back(child);
}

void Group_Node::removeChild(Node* child)
{
	child->unref();
	m_children.remove( child);
}

void Group_Node::setMatrixTransfo(const Geom::Matrix44f& mat)
{
	if (m_matrix_transfo == NULL)
		m_matrix_transfo = new Geom::Matrix44f(mat);
	*m_matrix_transfo = mat.transposed();
}


Geom::Matrix44f Group_Node::getMatrixTransfo() const
{
	if (m_matrix_transfo == NULL)
	{
		Geom::Matrix44f m;
		m.identity();
		return m;
	}

	return m_matrix_transfo->transposed();

}

void Group_Node::setMaterial(Material_Node* mat)
{
	mat->ref();
	m_material = mat;
}

void Group_Node::render()
{
	if (m_material != NULL)
		m_material->render();

	if (m_matrix_transfo != NULL)
		glMultMatrixf(&((*m_matrix_transfo)(0,0)));

	for (std::list<Node*>::const_iterator it = m_children.begin(); it !=m_children.end(); ++it)
		(*it)->render();
}




VBO_Node::VBO_Node()
: Node("SceneGraph::GL2_Node"), m_vbo(NULL), m_primitives(0)
{}


VBO_Node::VBO_Node(Algo::Render::GL2::MapRender* vbo)
: m_vbo(vbo), m_primitives(0)
{}


void VBO_Node::setVBO(Algo::Render::GL2::MapRender* vbo)
{
	m_vbo = vbo;
}


void VBO_Node::setPrimitives(unsigned int p)
{
	m_primitives |= p ;
}


void VBO_Node::unsetPrimitives(unsigned int p)
{
	m_primitives &= ~p ;
}


void VBO_Node::render()
{
	if (m_vbo != NULL)
	{
	/*
		if (m_primitives & Algo::Render::GL2::TRIANGLES)
		{
			glEnable(GL_POLYGON_OFFSET_FILL);
			glPolygonOffset(1.0f, 1.0f);
			glEnable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) ;
			glShadeModel(GL_SMOOTH);
			m_vbo->draw(m_current_material->getShader(), Algo::Render::GL2::TRIANGLES);
		}

		if (m_primitives & Algo::Render::GL2::LINES)
		{
			GLint prg;
			glGetIntegerv(GL_CURRENT_PROGRAM,&prg);
			glUseProgram(0);
			glDisable(GL_POLYGON_OFFSET_FILL);
			glDisable(GL_LIGHTING);
			m_vbo->draw(m_current_material->getShader(), Algo::Render::GL2::LINES);
			glUseProgram(prg);
		}

		if (m_primitives & Algo::Render::GL2::POINTS)
		{
			GLint prg;
			glGetIntegerv(GL_CURRENT_PROGRAM,&prg);
			glUseProgram(0);
			glDisable(GL_POLYGON_OFFSET_FILL);
			glDisable(GL_LIGHTING);
			glPointSize(3.0f);
			m_vbo->draw(m_current_material->getShader(), Algo::Render::GL2::POINTS);
			glUseProgram(prg);
		}
	*/
//		if (m_primitives & Algo::Render::GL2::FLAT_TRIANGLES)
//		{
//			glEnable(GL_POLYGON_OFFSET_FILL);
//			glPolygonOffset(1.0f, 1.0f);
//			glEnable(GL_LIGHTING);
//			m_vbo->draw(Algo::Render::GL2::FLAT_TRIANGLES);
//		}
	}
}



// MATERIAL METHODS

Material_Node::Material_Node()
:Node("SceneGraph::Material_Node"),
 m_has_diffuse(false),
 m_has_specular(false),
 m_has_ambient(false),
 m_has_shininess(false),
 m_has_color(false),
m_shader(NULL)
{}

void Material_Node::setDiffuse(const Geom::Vec4f& v)
{
	m_diffuse = v;
	m_has_diffuse=true;
}


void Material_Node::setSpecular(const Geom::Vec4f& v)
{
	m_specular = v;
	m_has_specular=true;
}


void Material_Node::setAmbient(const Geom::Vec4f& v)
{
	m_ambient = v;
	m_has_ambient=true;
}


void Material_Node::setShininess(float v)
{
	m_shininess = v;
	m_has_shininess=true;
}


void Material_Node::setColor(const Geom::Vec4f& v)
{
	m_color = v;
	m_has_color=true;
}

void Material_Node::setShader(Utils::GLSLShader *shader)
{
	m_shader = shader;
}

void Material_Node::setNoShader()
{
	m_disable_shader=true;
}


void Material_Node::render()
{
	Node::m_current_material = this;
//	if (m_has_diffuse)
//		glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,m_diffuse.data());
//	if (m_has_specular)
//		glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,m_specular.data());
//	if (m_has_ambient)
//		glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,m_ambient.data());
//	if (m_has_shininess)
//		glMaterialfv(GL_FRONT_AND_BACK,GL_SHININESS,&m_shininess);
//	if (m_has_color)
//		glColor3fv(m_color.data());
//
//	if (m_shader!=NULL)
//		m_shader->bind();
//	else
//		if (m_disable_shader)
//			glUseProgramObject(0);
}


void Material_Node::apply()
{

	// A MODIFIER: PASSER VARIABLES AU SHADER ?
	if (m_has_diffuse)
		glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,m_diffuse.data());
	if (m_has_specular)
		glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,m_specular.data());
	if (m_has_ambient)
		glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,m_ambient.data());
	if (m_has_shininess)
		glMaterialfv(GL_FRONT_AND_BACK,GL_SHININESS,&m_shininess);
	if (m_has_color)
		glColor3fv(m_color.data());

}



void renderGraph(Node* node)
{
	node->render();
}

void eraseGraph(Node* node)
{
	delete node;
}


}
}
}

