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

#include "Algo/Modelisation/subdivision.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Tilings
{

namespace Square
{

/*! Grid
 *************************************************************************/

template <typename PFP>
void Grid<PFP>::grid(unsigned int x, unsigned int y, bool close)
{
    // nb vertices
    int nb = (x+1)*(y+1);

    // vertice reservation
    this->m_tableVertDarts.reserve(nb);

    // creation of quads and storing vertices
    for (unsigned int i = 0; i < y; ++i)
    {
        for (unsigned int j = 1; j <= x; ++j)
        {
            Dart d = this->m_map.newFace(4, false);
            this->m_tableVertDarts.push_back(d);
            if (j == x)
                this->m_tableVertDarts.push_back(this->m_map.phi1(d));
        }
    }

    // store last row of vertices
    for (unsigned int i = 0; i < x; ++i)
    {
        this->m_tableVertDarts.push_back(this->m_map.phi_1(this->m_tableVertDarts[(y-1)*(x+1) + i]) );
    }
    this-> m_tableVertDarts.push_back(this->m_map.phi1(this->m_tableVertDarts[(y-1)*(x+1) +x]) );

    //sewing the quads
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

    this->m_dart = this->m_tableVertDarts[0];
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

template <typename PFP>
void Grid<PFP>::embedIntoTwistedStrip(VertexAttribute<VEC3, MAP>& position, float radius_min, float radius_max, float turns)
{
    float alpha = float(2.0*M_PI/this->m_ny);
    float beta = turns/float(this->m_ny);

    float radius = (radius_max + radius_min)/2.0f;
    float rdiff = (radius_max - radius_min)/2.0f;

    for(unsigned int i = 0; i <= this->m_ny; ++i)
    {
        for(unsigned int j = 0; j <= this->m_nx; ++j)
        {
            float rw = -rdiff + float(j)*2.0f*rdiff/float(this->m_nx);
            float r = radius + rw*cos(beta*float(i));
            VEC3 pos(r*cos(alpha*float(i)), r*sin(alpha*float(i)), rw*sin(beta*float(i)));
            position[this->m_tableVertDarts[ i*(this->m_nx+1)+j ] ] = pos;
        }
    }
}

template <typename PFP>
void Grid<PFP>::embedIntoHelicoid(VertexAttribute<VEC3, MAP>& position, float radius_min,  float radius_max, float maxHeight, float nbTurn, int orient)
{
    float alpha = float(2.0*M_PI/this->m_nx)*nbTurn;
    float hS = maxHeight/this->m_nx;

    // 	float radius = (radius_max + radius_min)/2.0f;
    // 	float rdiff = (radius_max - radius_min)/2.0f;

    for(unsigned int i = 0; i <= this->m_ny; ++i)
    {
        for(unsigned int j = 0; j <= this->m_nx; ++j)
        {
            // 			float r = radius_max + radius_min*cos(beta*float(j));
            float r,x,y;
            // 			if(i==1) {
            // 				r = radius_max;
            // 			}
            // 			else {
            r = radius_min+(radius_max-radius_min)*float(i)/float(this->m_ny);
            // 			}
            x = orient*r*sin(alpha*float(j));
            y = orient*r*cos(alpha*float(j));

            VEC3 pos(x, y, j*hS);
            Dart d = this->m_tableVertDarts[i*(this->m_nx+1)+j];
            position[d] = pos;
        }
    }
}

/*! Cylinder
 *************************************************************************/

template <typename PFP>
void Cylinder<PFP>::cylinder(unsigned int n, unsigned int z)
{
    int nb = (n)*(z+1)+2;

    // vertice reservation
    this->m_tableVertDarts.reserve(nb);

    // creation of quads and storing vertices
    for (unsigned int i = 0; i < z; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
        {
            Dart d = this->m_map.newFace(4, false);
            this->m_tableVertDarts.push_back(d);
        }
    }

    for (unsigned int i = 0; i < n; ++i)
    {
        this->m_tableVertDarts.push_back(this->m_map.phi_1(this->m_tableVertDarts[(z-1)*n+i]) );
    }

    //sewing the quads
    for (unsigned int i = 0; i < z; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
        {
            if (i > 0) // sew with preceeding row
            {
                int pos = i*n+j;
                Dart d = this->m_tableVertDarts[pos];
                Dart e = this->m_tableVertDarts[pos-n];
                e = this->m_map.phi1(this->m_map.phi1(e));
                this->m_map.sewFaces(d, e, false);
            }
            if (j > 0) // sew with preceeding column
            {
                int pos = i*n+j;
                Dart d = this->m_tableVertDarts[pos];
                d = this->m_map.phi_1(d);
                Dart e = this->m_tableVertDarts[pos-1];
                e = this->m_map.phi1(e);
                this->m_map.sewFaces(d, e, false);
            }
            else
            {
                int pos = i*n;
                Dart d = this->m_tableVertDarts[pos];
                d = this->m_map.phi_1(d);
                Dart e = this->m_tableVertDarts[pos+(n-1)];
                e = this->m_map.phi1(e);
                this->m_map.sewFaces(d, e, false);
            }
        }
    }

    if(m_top_closed)
        closeTop();

    if(m_bottom_closed)
        closeBottom();

    this->m_dart = this->m_tableVertDarts.front();
}

template <typename PFP>
void Cylinder<PFP>::closeTop()
{
    this->m_map.closeHole(this->m_map.phi_1(this->m_tableVertDarts[this->m_nx*this->m_nz]));
    m_top_closed = true;
}

template <typename PFP>
void Cylinder<PFP>::triangleTop()
{
    if (m_top_closed)
    {
        Dart d =  this->m_map.phi_1(this->m_tableVertDarts[this->m_nx*this->m_nz]);
        this->m_map.fillHole(d);

        d = this->m_map.phi2(d);
        if(this->m_map.faceDegree(d) > 3)
        {
            Algo::Surface::Modelisation::trianguleFace<PFP>(this->m_map, d);
            //this->m_tableVertDarts.push_back(this->m_map.phi_1(d));
            m_topVertDart = this->m_map.phi_1(d);
        }

        m_top_triangulated = true;
    }
}

template <typename PFP>
void Cylinder<PFP>::closeBottom()
{
    this->m_map.closeHole(this->m_tableVertDarts[0]);
    m_bottom_closed = true;
}

template <typename PFP>
void Cylinder<PFP>::triangleBottom()
{
    if (m_bottom_closed)
    {
        Dart d = this->m_tableVertDarts[0];
        this->m_map.fillHole(d);

        d = this->m_map.phi2(d);
        if(this->m_map.faceDegree(d) > 3)
        {
            Algo::Surface::Modelisation::trianguleFace<PFP>(this->m_map, d);
            //this->m_tableVertDarts.push_back(this->m_map.phi_1(d));
            m_bottomVertDart = this->m_map.phi_1(d);
        }

        m_bottom_triangulated = true;
    }
}

template <typename PFP>
void Cylinder<PFP>::embedIntoCylinder(VertexAttribute<VEC3, MAP>& position, float bottom_radius, float top_radius, float height)
{
    float alpha = float(2.0*M_PI/this->m_nx);
    float dz = height/float(this->m_nz);

    for(unsigned int i = 0; i <= this->m_nz; ++i)
    {
        float a = float(i)/float(this->m_nz);
        float radius = a*top_radius + (1.0f-a)*bottom_radius;
        for(unsigned int j = 0; j < this->m_nx; ++j)
        {

            float x = radius*cos(alpha*float(j));
            float y = radius*sin(alpha*float(j));
            position[this->m_tableVertDarts[i*(this->m_nx)+j] ] = VEC3(x, y, -height/2 + dz*float(i));
        }
    }

    //int indexUmbrella = this->m_nx*(this->m_nz+1);

    if (m_bottom_triangulated)
    {
        //position[this->m_tableVertDarts[indexUmbrella++] ] = VEC3(0.0f, 0.0f, height/2 );
        position[m_bottomVertDart] = VEC3(0.0f, 0.0f, -height/2 );
    }

    if (m_top_triangulated)
    {
        //position[this->m_tableVertDarts[indexUmbrella] ] = VEC3(0.0f, 0.0f, -height/2 );
        position[m_topVertDart] = VEC3(0.0f, 0.0f, height/2 );
    }
}

template <typename PFP>
void Cylinder<PFP>::embedIntoSphere(VertexAttribute<VEC3, MAP>& position, float radius)
{
    float alpha = float(2.0*M_PI/this->m_nx);
    float beta = float(M_PI/(this->m_nz+2));

    for(unsigned int i = 0; i <= this->m_nz; ++i)
    {
        for(unsigned int j = 0; j < this->m_nx; ++j)
        {
            float h = float(radius * sin(-M_PI/2.0+(i+1)*beta));
            float rad = float(radius * cos(-M_PI/2.0+(i+1)*beta));

            float x = rad*cos(alpha*float(j));
            float y = rad*sin(alpha*float(j));

            position[this->m_tableVertDarts[i*(this->m_nx)+j] ] = VEC3(x, y, h );
        }
    }

    // bottom  pole
    if (m_bottom_triangulated)
    {
        //position[this->m_tableVertDarts[this->m_nx*(this->m_nz+1)] ] = VEC3(0.0f, 0.0f, radius);
        position[m_bottomVertDart] = VEC3(0.0f, 0.0f, -radius);
    }

    //  top pole
    if (m_top_triangulated)
    {
        //position[this->m_tableVertDarts[this->m_nx*(this->m_nz+1)+1] ] = VEC3(0.0f, 0.0f, -radius);
        position[m_topVertDart] = VEC3(0.0f, 0.0f, radius);
    }
}

template <typename PFP>
void Cylinder<PFP>::embedIntoCone(VertexAttribute<VEC3, MAP>& position, float radius, float height)
{
    if(m_top_closed && m_top_triangulated)
    {
        float alpha = float(2.0*M_PI/this->m_nx);
        float dz = height/float(this->m_nz+1);
        for( unsigned int i = 0; i <= this->m_nz; ++i)
        {
            for(unsigned int j = 0; j < this->m_nx; ++j)
            {
                float rad = radius * float(this->m_nz+1-i)/float(this->m_nz+1);
                float h = -height/2 + dz*float(i);
                float x = rad*cos(alpha*float(j));
                float y = rad*sin(alpha*float(j));

                position[this->m_tableVertDarts[i*(this->m_nx)+j] ] = VEC3(x, y, h);
            }
        }

        //int indexUmbrella = this->m_nx*(this->m_nz+1);
        if (m_bottom_triangulated)
        {
            //position[this->m_tableVertDarts[indexUmbrella] ] = VEC3(0.0f, 0.0f, -height/2 );
            position[m_bottomVertDart] = VEC3(0.0f, 0.0f, -height/2 );
        }

        //  top always closed in cone
        //position[this->m_tableVertDarts[indexUmbrella++] ] = VEC3(0.0f, 0.0f, height/2 );
        position[m_topVertDart] = VEC3(0.0f, 0.0f, height/2 );
    }

}

/*! Cube
 *************************************************************************/

template <typename PFP>
void Cube<PFP>::cube(unsigned int x, unsigned int y, unsigned int z)
{
    //this->cylinder(2*(x+y), z, false, false);
    this->m_nx = x;
    this->m_ny = y;
    this->m_nz = z;

    int nb = 2*(x+y)*(z+1) + 2*(x-1)*(y-1);
    this->m_tableVertDarts.reserve(nb);

    // we now have the 4 sides, just need to create store and sew top & bottom
    // the top
    Grid<PFP> gtop(this->m_map,x,y,false);
    std::vector<Dart>& tableTop = gtop.getVertexDarts();

    int index_side = 2*(x+y)*z;
    for(unsigned int i = 0; i < x; ++i)
    {
        Dart d = this->m_map.phi_1(this->m_tableVertDarts[index_side++]);
        Dart e = tableTop[i];
        this->m_map.sewFaces(d, e, false);
    }
    for(unsigned int i = 0; i < y; ++i)
    {
        Dart d = this->m_map.phi_1(this->m_tableVertDarts[index_side++]);
        Dart e = tableTop[x+i*(x+1)];
        this->m_map.sewFaces(d, e, false);
    }
    for(unsigned int i = 0; i < x; ++i)
    {
        Dart d = this->m_map.phi_1(this->m_tableVertDarts[index_side++]);
        Dart e = tableTop[(x+1)*(y+1)-2 - i];
        e = this->m_map.phi_1(e);
        this->m_map.sewFaces(d, e, false);
    }
    for(unsigned int i = 0; i < y; ++i)
    {
        Dart d = this->m_map.phi_1(this->m_tableVertDarts[index_side++]);
        Dart e = tableTop[(y-1-i)*(x+1)];
        e = this->m_map.phi_1(e);
        this->m_map.sewFaces(d, e, false);
    }

    // the bottom
    Grid<PFP> gBottom(this->m_map,x,y,false);
    std::vector<Dart>& tableBottom = gBottom.getVertexDarts();

    index_side = 3*(x+y)+(x-1);
    for(unsigned int i = 0; i < x; ++i)
    {
        Dart d = this->m_tableVertDarts[(index_side--)%(2*(x+y))];
        Dart e = tableBottom[i];
        this->m_map.sewFaces(d, e, false);
    }
    for(unsigned int i = 0; i < y; ++i)
    {
        Dart d = this->m_tableVertDarts[(index_side--)%(2*(x+y))];
        Dart e = tableBottom[x+i*(x+1)];
        this->m_map.sewFaces(d, e, false);
    }
    for(unsigned int i = 0; i < x; ++i)
    {
        Dart d = this->m_tableVertDarts[(index_side--)%(2*(x+y))];
        Dart e = tableBottom[(x+1)*(y+1)-2 - i];
        e = this->m_map.phi_1(e);
        this->m_map.sewFaces(d, e, false);
    }
    for(unsigned int i = 0; i < y; ++i)
    {
        Dart d = this->m_tableVertDarts[(index_side--)%(2*(x+y))];
        Dart e = tableBottom[(y-1-i)*(x+1)];
        e = this->m_map.phi_1(e);
        this->m_map.sewFaces(d, e, false);
    }

    // and add new vertex in m_tableVertDarts
    //top  first
    for(unsigned int i = 1; i < y; ++i)
    {
        for(unsigned int j = 1; j < x; ++j)
            this->m_tableVertDarts.push_back(tableTop[i*(x+1)+j]);
    }

    // then bottom
    for(unsigned int i = 1; i < y; ++i)
    {
        for(unsigned int j = 1; j < x; ++j)
            this->m_tableVertDarts.push_back(tableBottom[i*(x+1)+j]);
    }

}

template <typename PFP>
void Cube<PFP>::embedIntoCube(VertexAttribute<VEC3, MAP>& position, float sx, float sy, float sz)
{
    float dz = sz/float(this->m_nz);
    float dy = sy/float(this->m_ny);
    float dx = sx/float(this->m_nx);

    // first embedding the sides
    int index = 0;
    for (unsigned int k = 0; k <= this->m_nz; ++k)
    {
        float z = float(k)*dz - sz/2.0f;
        for (unsigned int i = 0; i < this->m_nx; ++i)
        {
            float x = float(i)*dx - sx/2.0f;
            position[this->m_tableVertDarts[ index++ ] ] = VEC3(x, -sy/2.0f, z);
        }
        for (unsigned int i = 0; i < this->m_ny; ++i)
        {
            float y = float(i)*dy - sy/2.0f;
            position[this->m_tableVertDarts[ index++ ] ] = VEC3(sx/2.0f, y, z);
        }
        for (unsigned int i = 0; i < this->m_nx; ++i)
        {
            float x = sx/2.0f-float(i)*dx;
            position[this->m_tableVertDarts[ index++ ] ] = VEC3(x, sy/2.0f, z);
        }
        for (unsigned int i = 0; i < this->m_ny ;++i)
        {
            float y = sy/2.0f - float(i)*dy;
            position[this->m_tableVertDarts[ index++ ] ] = VEC3(-sx/2.0f, y, z);
        }
    }

    // the top
    for(unsigned int i = 1; i  < this->m_ny; ++i)
    {
        for(unsigned int j = 1; j < this->m_nx; ++j)
        {
            VEC3 pos(-sx/2.0f+float(j)*dx, -sy/2.0f+float(i)*dy, sz/2.0f);
            position[this->m_tableVertDarts[ index++ ] ] = pos;
        }
    }

    // the bottom
    for(unsigned int i = 1; i < this->m_ny; ++i)
    {
        for(unsigned int j = 1; j < this->m_nx; ++j)
        {
            VEC3 pos(-sx/2.0f+float(j)*dx, sy/2.0f-float(i)*dy, -sz/2.0f);
            position[this->m_tableVertDarts[ index++ ] ] = pos;
        }
    }
}

/*! Tore
 *************************************************************************/

template <typename PFP>
void Tore<PFP>::tore(unsigned int n, unsigned int m)
{
    //this->cylinder(n, m);
    this->m_nx = n;
    this->m_ny = m;
    this->m_nz = -1;

    // just finish to sew
    for(unsigned int i = 0; i < n; ++i)
    {
        Dart d = this->m_tableVertDarts[i];
        Dart e = this->m_tableVertDarts[(m*n)+i];
        e = this->m_map.phi_1(e);
        this->m_map.sewFaces(d, e);
    }

    // remove the last n vertex darts that are no more necessary (sewed with n first)
    // memory not freed (but will be when destroy the Polyhedron), not important ??
    this->m_tableVertDarts.resize(m*n);
}

template <typename PFP>
void Tore<PFP>::embedIntoTore(VertexAttribute<VEC3, MAP>& position, float big_radius, float small_radius)
{
    float alpha = float(2.0*M_PI/this->m_nx);
    float beta = float(2.0*M_PI/this->m_ny);

    for (unsigned int i = 0; i < this->m_nx; ++i)
    {
        for(unsigned int j = 0; j < this->m_ny; ++j)
        {
            float z = small_radius*sin(beta*float(j));
            float r = big_radius + small_radius*cos(beta*float(j));
            float x = r*cos(alpha*float(i));
            float y = r*sin(alpha*float(i));
            position[this->m_tableVertDarts[j*(this->m_nx)+i] ] = VEC3(x, y, z);
        }
    }
}

} // namespace Square

} // namespace Tilings

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
