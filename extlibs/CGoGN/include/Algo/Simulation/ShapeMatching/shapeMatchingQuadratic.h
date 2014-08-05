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


#include "shapeMatching.h"

#ifndef _SHAPE_MATCHING_QUADRATIC_H_
#define _SHAPE_MATCHING_QUADRATIC_H_

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
class ShapeMatchingQuadratic : public ShapeMatching<PFP>
{
public:
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;
    typedef typename PFP::REAL REAL;

    typedef Eigen::Matrix<double, 9, 9> Matrix9d;
    typedef Eigen::Matrix<double, 9, 1> Vec9d;
    typedef Eigen::Matrix<double, 3, 9> Matrix39d;

protected:
    REAL m_beta;

    Matrix9d m_aqqtild;

    // q^{~}_{i} where q^{~} = [q_x, q_y, q_z, q^2_x, q^2_y, q^2_z, q_x q_y , q_y q_z, q_z q_x]
    std::vector<Vec9d > m_qtild;

public:
    ShapeMatchingQuadratic(MAP& map, VertexAttribute<VEC3>& position, VertexAttribute<REAL>& mass, REAL beta):
        ShapeMatching<PFP>(map, position, mass),
        m_beta(beta)
    { }

    ~ShapeMatchingQuadratic()
    { }

    void initialize();

    void shapeMatch();
};

} // namespace ShapeMatching

} // namespace Simulation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "shapeMatchingQuadratic.hpp"

#endif
