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

#ifndef _SHAPE_MATCHING_LINEAR_H_
#define _SHAPE_MATCHING_LINEAR_H_

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
class ShapeMatchingLinear : public ShapeMatching<PFP>
{
public:
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;
    typedef typename PFP::REAL REAL;

protected:
    REAL m_beta;

    // A_{qq}
    Eigen::Matrix3d m_aqq;

private:
    void computeAqqMatrix();

public:
    ShapeMatchingLinear(MAP& map, VertexAttribute<VEC3>& position, VertexAttribute<REAL>& mass, REAL beta):
        ShapeMatching<PFP>(map, position, mass),
        m_beta(beta)
    { }

    ~ShapeMatchingLinear()
    { }

    void initialize();

    void shapeMatch();
};

} // namespace ShapeMatching

} // namespace Simulation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "shapeMatchingLinear.hpp"

#endif
