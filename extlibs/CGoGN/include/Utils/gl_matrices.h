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


#ifndef __GL_MATRICES_H_
#define __GL_MATRICES_H_

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <stack>
#include "Geometry/vector_gen.h"

namespace CGoGN
{
namespace Utils
{

class GL_Matrices
{
public:
	/// 0: projection / 1: modelView /2: transfo / 3:PMV /4:normal
	glm::mat4 m_matrices[5];
	/// stack of transfo matrix
	std::stack<glm::mat4> m_stack;

public:
	void pushTransfo()
	{
		m_stack.push(m_matrices[2]);
	}

	void popTransfo()
	{
		if (m_stack.empty())
			return;
		m_matrices[2] = m_stack.top();
		m_stack.pop();
	}

	const glm::mat4&  getTransfo() const
	{
		return m_matrices[2];
	}

	glm::mat4& getTransfo()
	{
		return m_matrices[2];
	}

	void rotate(float angle, const Geom::Vec3f& Axis)
	{
		glm::mat4 X = glm::rotate(glm::mat4(1.f), glm::radians(angle), glm::vec3(Axis[0],Axis[1],Axis[2])) * m_matrices[2];
		m_matrices[2] = X;
	}

	void translate(const Geom::Vec3f& P)
	{
		
		glm::mat4 X = glm::translate(glm::mat4(1.f), glm::vec3(P[0],P[1],P[2])) * m_matrices[2];
		m_matrices[2] = X;
	}

	void scale(const Geom::Vec3f& S)
	{
		glm::mat4 X = glm::scale(glm::mat4(1.f), glm::vec3(S[0],S[1],S[2])) * m_matrices[2];
		m_matrices[2] = X;
	}

	void scale(float s)
	{
		glm::mat4 X	= glm::scale(glm::mat4(1.f), glm::vec3(s,s,s)) * m_matrices[2];
		m_matrices[2] = X;
	}

	void apply (const glm::mat4& m)
	{
		glm::mat4 X = m * m_matrices[2];
		m_matrices[2] = X;
	}

};


}
}

#endif
