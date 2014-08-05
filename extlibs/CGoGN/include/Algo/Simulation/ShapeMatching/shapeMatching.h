/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
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

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#ifndef _SHAPE_MATCHING_H_
#define _SHAPE_MATCHING_H_

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Simulation
{

namespace ShapeMatching
{

template <typename PFP>
class ShapeMatching
{
public:
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;
    typedef typename PFP::REAL REAL;

protected:
    MAP& m_map;
    VertexAttribute<VEC3>& m_position; // x_i : position
    VertexAttribute<REAL>& m_mass;  // m_i : mass
    VertexAttribute<VEC3> m_goal;

    // q_{i} = x^{0} - x^{0}_{cm}
    std::vector<Eigen::Vector3d> m_q;

public:
    ShapeMatching(MAP& map, VertexAttribute<VEC3>& position, VertexAttribute<REAL>& mass);

    virtual ~ShapeMatching();

    Eigen::Vector3d massCenter();

    void initialize();

    void shapeMatch();

    void computeVelocities(VertexAttribute<VEC3>& velocity, VertexAttribute<VEC3>& fext, REAL h, REAL alpha);

    void applyVelocities(VertexAttribute<VEC3>& velocity, REAL h);
};

} // namespace ShapeMatching

} // namespace Simulation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN


#include "shapeMatching.hpp"

#endif
