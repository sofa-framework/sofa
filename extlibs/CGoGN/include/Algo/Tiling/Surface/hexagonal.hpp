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

namespace Surface
{

namespace Tilings
{

namespace Hexagonal
{

/*! Grid
 *************************************************************************/

template <typename PFP>
void Grid<PFP>::grid(unsigned int x, unsigned int y, bool close)
{
    // nb vertices
    int nb = x*y +1;

    // vertice reservation
    this->m_tableVertDarts.reserve(nb);

    // creation of triangles and storing vertices
    for (unsigned int i = 0; i < y; ++i)
    {
        for (unsigned int j = 1; j <= x; ++j)
        {
            Dart d = this->m_map.newFace(6, false);
            this->m_tableVertDarts.push_back(d);
            this->m_tableVertDarts.push_back(this->m_map.phi1(d));
            if (j== 1)
                this->m_tableVertDarts.push_back(this->m_map.phi_1(d));
            if (j == x)
                this->m_tableVertDarts.push_back(this->m_map.phi1(this->m_map.phi1(d)));
        }
    }

    // store last row of vertices
    for (unsigned int i = 0; i < x; ++i)
    {
        this->m_tableVertDarts.push_back(this->m_map.phi_1(this->m_tableVertDarts[(y-1)*(x+1) + i]) );
    }
    //this-> m_tableVertDarts.push_back(this->m_map.phi1(this->m_tableVertDarts[(y-1)*(x+1) +x]) );

    //sewing the triangles
    for (unsigned int i = 0; i < y; ++i)
    {
        for (unsigned int j = 0; j < x; ++j)
        {
            if (i > 0) // sew with preceeding row
            {
                int pos = i*(x+1)+j;
                Dart d = this->m_tableVertDarts[pos];
                Dart e = this->m_tableVertDarts[pos-(x+1)];
                e = this->m_map.phi1(this->m_map.phi1(e));
                this->m_map.sewFaces(d, e, false);
            }
            if (j > 0) // sew with preceeding column
            {
                int pos = i*(x+1)+j;
                Dart d = this->m_tableVertDarts[pos];
                d = this->m_map.phi_1(d);
                Dart e = this->m_tableVertDarts[pos-1];
                e = this->m_map.phi1(e);
                this->m_map.sewFaces(d, e, false);
            }
        }
    }

    if(close)
        this->m_map.closeHole(this->m_tableVertDarts[0]) ;
}

template <typename PFP>
void Grid<PFP>::embedIntoGrid(VertexAttribute<VEC3, MAP>& position, float x, float y, float z)
{
    float dx = x / float(this->m_nx);
    float dy = y / float(this->m_ny);

    for(unsigned int i = 0; i <= this->m_ny; ++i)
    {
        for(unsigned int j = 0; j <= this->m_nx;++j)
        {
            position[this->m_tableVertDarts[i*(this->m_nx+1)+j] ] = VEC3(-x/2 + dx*float(j), -y/2 + dy*float(i), z);
        }
    }
}

} // namespace Hexagonal

} // namespace Tilings

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
