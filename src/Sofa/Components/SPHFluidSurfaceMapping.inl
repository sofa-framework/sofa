#ifndef SOFA_COMPONENTS_SPHFLUIDSURFACEMAPPING_INL
#define SOFA_COMPONENTS_SPHFLUIDSURFACEMAPPING_INL

#include "SPHFluidSurfaceMapping.h"
#include "SpatialGridContainer.inl"

#include "Sofa/Core/Mapping.inl"
#include "Common/rmath.h"
#include "Graph/GNode.h"
#include "GL/template.h"

#include <map>
#include <list>

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::init()
{
    this->Inherit::init();
    Graph::GNode* node = dynamic_cast<Graph::GNode*>(this->getFrom()->getContext());
    if (node)
    {
        //the following line produces a compilation error with GCC 3.3 :(
        //sph = node->getNodeObject<SPHForceField>();
        node->getNodeObject(sph);
    }
    if (sph)
    {
        //mRadius = sph->getParticleRadius();
        mIsoValue /= sph->getParticleFieldConstant((InReal)mRadius);
    }
    grid = new Grid((InReal)mStep);
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::createPoints(OutVecCoord& out, int x, int y, int z, Cell* c, const Cell* cx, const Cell* cy, const Cell* cz, const OutReal isoval)
{
    if (c->data.val>isoval)
    {
        if (!(cx->data.val>isoval))
            c->data.p[0] = addPoint<0>(out, x,y,z,c->data.val,cx->data.val,isoval);
        if (!(cy->data.val>isoval))
            c->data.p[1] = addPoint<1>(out, x,y,z,c->data.val,cy->data.val,isoval);
        if (!(cz->data.val>isoval))
            c->data.p[2] = addPoint<2>(out, x,y,z,c->data.val,cz->data.val,isoval);
    }
    else
    {
        if (cx->data.val>isoval)
            c->data.p[0] = addPoint<0>(out, x,y,z,c->data.val,cx->data.val,isoval);
        if (cy->data.val>isoval)
            c->data.p[1] = addPoint<1>(out, x,y,z,c->data.val,cy->data.val,isoval);
        if (cz->data.val>isoval)
            c->data.p[2] = addPoint<2>(out, x,y,z,c->data.val,cz->data.val,isoval);
    }
}


template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::createFaces(OutVecCoord& out, const Cell** cells, const OutReal isoval)
{

    /* Convention:
     *          Z
     *          ^
     *          |
     *          4----4----5
     *         /|        /|
     *        7 |       5 |
     *       /  8      /  9
     *      7---+6----6   |
     *      |   |     |   |
     *      |   0----0+---1--> X
     *     11  /     10  /
     *      | 3       | 1
     *      |/        |/
     *      3----2----2
     *     /
     *    /
     *  |_
     * Y
     */

    static const int edgecell[12] = { 0, 1, 2, 0, 4, 5, 6, 4, 0, 1, 3, 2 };
    static const int edgepts [12] = { 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2 };

    int mk;
    if (cells[0]->data.val > isoval) mk = 1; else mk=0;
    if (cells[1]->data.val > isoval) mk|= 2;
    if (cells[3]->data.val > isoval) mk|= 4;
    if (cells[2]->data.val > isoval) mk|= 8;
    if (cells[4]->data.val > isoval) mk|= 16;
    if (cells[5]->data.val > isoval) mk|= 32;
    if (cells[7]->data.val > isoval) mk|= 64;
    if (cells[6]->data.val > isoval) mk|= 128;

    const int* tri=MarchingCubeTriTable[mk];
    while (*tri>=0)
    {
        if (addFace(cells[edgecell[tri[0]]]->data.p[edgepts[tri[0]]],
                cells[edgecell[tri[1]]]->data.p[edgepts[tri[1]]],
                cells[edgecell[tri[2]]]->data.p[edgepts[tri[2]]], out.size())<0)
        {
            std::cerr << "  mk=0x"<<std::hex<<mk<<std::dec<<" p1="<<tri[0]<<" p2="<<tri[1]<<" p3="<<tri[2]<<std::endl;
            for (int e=0; e<12; e++) std::cerr << "  e"<<e<<"="<<cells[edgecell[e]]->data.p[edgepts[e]];
            std::cerr<<std::endl;
        }
        tri+=3;
    }
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::apply( OutVecCoord& out, const InVecCoord& in )
{
    if (!sph) return;
    //const InReal invStep = (InReal)(1/mStep);
    out.resize(0);
    clear();
    if (in.size()==0) return;
    const InReal r = (InReal)(getRadius()); // / mStep);
    grid->begin();
    for (unsigned int ip=0; ip<in.size(); ip++)
    {
        grid->add(ip, in[ip], true);
    }
    grid->end();

    //////// EVALUATE COLOR FUNCTION ////////
    grid->computeField(sph, r);

    //////// MARCHING CUBE ////////

    const OutReal isoval = (OutReal) getIsoValue();
    typename Grid::iterator end = grid->gridEnd();
    typename Grid::iterator it;

    // Create points
    for (it = grid->gridBegin(); it!=end; ++it)
    {
        typename Grid::Key p0 = it->first;
        const int x0 = p0[0]*GRIDDIM;
        const int y0 = p0[1]*GRIDDIM;
        const int z0 = p0[2]*GRIDDIM;
        typename Grid::Grid* g = it->second;
        const typename Grid::Grid* gx1 = g->neighbors[1];
        const typename Grid::Grid* gy1 = g->neighbors[3];
        const typename Grid::Grid* gz1 = g->neighbors[5];
        int x,y,z;
        Cell* c = g->cell;
        const Cell* cx1 = gx1->cell;
        const Cell* cy1 = gy1->cell;
        const Cell* cz1 = gz1->cell;
        for (z=0; z<GRIDDIM-1; z++)
        {
            for (y=0; y<GRIDDIM-1; y++)
            {
                for (x=0; x<GRIDDIM-1; x++)
                {
                    createPoints(out, x0+x, y0+y, z0+z, c, c+DX, c+DY, c+DZ, isoval);
                    c+=DX;
                }
                // X border
                createPoints(out, x0+x, y0+y, z0+z, c, cx1, c+DY, c+DZ, isoval);
                c+=DX;
                cx1+=DY;
            }
            // Y BORDER
            {
                for (x=0; x<GRIDDIM-1; x++)
                {
                    createPoints(out, x0+x, y0+y, z0+z, c, c+DX, cy1, c+DZ, isoval);
                    c+=DX;
                    cy1+=DX;
                }
                // X border
                createPoints(out, x0+x, y0+y, z0+z, c, cx1, cy1, c+DZ, isoval);
                c+=DX;
                cx1+=DY;
                cy1+=DZ+DX-DY;
            }
        }
        // Z BORDER
        for (y=0; y<GRIDDIM-1; y++)
        {
            for (x=0; x<GRIDDIM-1; x++)
            {
                createPoints(out, x0+x, y0+y, z0+z, c, c+DX, c+DY, cz1, isoval);
                c+=DX;
                cz1+=DX;
            }
            // X border
            createPoints(out, x0+x, y0+y, z0+z, c, cx1, c+DY, cz1, isoval);
            c+=DX;
            cx1+=DY;
            cz1+=DX;
        }
        // Y BORDER
        {
            for (x=0; x<GRIDDIM-1; x++)
            {
                createPoints(out, x0+x, y0+y, z0+z, c, c+DX, cy1, cz1, isoval);
                c+=DX;
                cy1+=DX;
                cz1+=DX;
            }
            // X border
            createPoints(out, x0+x, y0+y, z0+z, c, cx1, cy1, cz1, isoval);
        }
    }

    // Create faces
    for (it = grid->gridBegin(); it!=end; ++it)
    {
        typename Grid::Grid* g = it->second;
        const typename Grid::Grid* gx1 = g->neighbors[1];
        const typename Grid::Grid* gy1 = g->neighbors[3];
        const typename Grid::Grid* gz1 = g->neighbors[5];
        int x,y,z;
        const Cell* cells[8];
        const Cell* c = g->cell;
        const Cell* cx1 = gx1->cell;
        const Cell* cy1 = gy1->cell;
        const Cell* cz1 = gz1->cell;
        const Cell* cx1y1 = g->neighbors[1]->neighbors[3]->cell;
        const Cell* cx1z1 = g->neighbors[1]->neighbors[5]->cell;
        const Cell* cy1z1 = g->neighbors[3]->neighbors[5]->cell;
        const Cell* cx1y1z1 = g->neighbors[1]->neighbors[3]->neighbors[5]->cell;
        for (z=0; z<GRIDDIM-1; z++)
        {
            for (y=0; y<GRIDDIM-1; y++)
            {
                for (x=0; x<GRIDDIM-1; x++)
                {
                    cells[0] = c;               cells[1] = c+DX;            cells[2] = c+DY;            cells[3] = c+DX+DY;
                    cells[4] = cells[0]+DZ;     cells[5] = cells[1]+DZ;     cells[6] = cells[2]+DZ;     cells[7] = cells[3]+DZ;
                    createFaces(out, cells, isoval);
                    c+=DX;
                }
                // X border
                cells[0] = c;               cells[1] = cx1;             cells[2] = c+DY;            cells[3] = cx1+DY;
                cells[4] = cells[0]+DZ;     cells[5] = cells[1]+DZ;     cells[6] = cells[2]+DZ;     cells[7] = cells[3]+DZ;
                createFaces(out, cells, isoval);
                c+=DX;
                cx1+=DY;
            }
            // Y BORDER
            {
                for (x=0; x<GRIDDIM-1; x++)
                {
                    cells[0] = c;               cells[1] = c+DX;            cells[2] = cy1;             cells[3] = cy1+DX;
                    cells[4] = cells[0]+DZ;     cells[5] = cells[1]+DZ;     cells[6] = cells[2]+DZ;     cells[7] = cells[3]+DZ;
                    createFaces(out, cells, isoval);
                    c+=DX;
                    cy1+=DX;
                }
                // X border
                cells[0] = c;               cells[1] = cx1;            cells[2] = cy1;              cells[3] = cx1y1;
                cells[4] = cells[0]+DZ;     cells[5] = cells[1]+DZ;     cells[6] = cells[2]+DZ;     cells[7] = cells[3]+DZ;
                createFaces(out, cells, isoval);
                c+=DX;
                cx1+=DY;
                cy1+=DZ+DX-DY;
                cx1y1+=DZ;
            }
        }
        // Z BORDER
        for (y=0; y<GRIDDIM-1; y++)
        {
            for (x=0; x<GRIDDIM-1; x++)
            {
                cells[0] = c;               cells[1] = c+DX;            cells[2] = c+DY;            cells[3] = c+DX+DY;
                cells[4] = cz1;             cells[5] = cz1+DX;          cells[6] = cz1+DY;          cells[7] = cz1+DX+DY;
                createFaces(out, cells, isoval);
                c+=DX;
                cz1+=DX;
            }
            // X border
            cells[0] = c;               cells[1] = cx1;            cells[2] = c+DY;            cells[3] = cx1+DY;
            cells[4] = cz1;             cells[5] = cx1z1;          cells[6] = cz1+DY;          cells[7] = cx1z1+DY;
            createFaces(out, cells, isoval);
            c+=DX;
            cx1+=DY;
            cz1+=DX;
            cx1z1+=DY;
        }
        // Y BORDER
        {
            for (x=0; x<GRIDDIM-1; x++)
            {
                cells[0] = c;               cells[1] = c+DX;            cells[2] = cy1;            cells[3] = cy1+DX;
                cells[4] = cz1;             cells[5] = cz1+DX;          cells[6] = cy1z1;          cells[7] = cy1z1+DX;
                createFaces(out, cells, isoval);
                c+=DX;
                cy1+=DX;
                cz1+=DX;
                cy1z1+=DX;
            }
            // X border
            cells[0] = c;               cells[1] = cx1;            cells[2] = cy1;            cells[3] = cx1y1;
            cells[4] = cz1;             cells[5] = cx1z1;          cells[6] = cy1z1;          cells[7] = cx1y1z1;
            createFaces(out, cells, isoval);
        }
    }

    //std::cout << out.size() << " points, "<<seqTriangles.size()<<" faces."<<std::endl;
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::applyJ( OutVecDeriv& /*out*/, const InVecDeriv& /*in*/ )
{
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::draw()
{
    if (!getContext()->getShowMappings()) return;
    if (!grid) return;
    grid->draw();

    float scale = (float)mStep;
    typename Grid::iterator end = grid->gridEnd();
    typename Grid::iterator it;

    glPointSize(3);
    glColor4f(1,1,1,1);
    glBegin(GL_POINTS);
    for (it = grid->gridBegin(); it!=end; ++it)
    {
        typename Grid::Key p0 = it->first;
        const int x0 = p0[0]*GRIDDIM;
        const int y0 = p0[1]*GRIDDIM;
        const int z0 = p0[2]*GRIDDIM;
        typename Grid::Grid* g = it->second;
        int x,y,z;
        Cell* c = g->cell;
        for (z=0; z<GRIDDIM; z++)
        {
            for (y=0; y<GRIDDIM; y++)
            {
                for (x=0; x<GRIDDIM; x++)
                {
                    if (c->data.val > mIsoValue)
                        glVertex3f((x0+x)*scale,(y0+y)*scale,(z0+z)*scale);
                    c+=DX;
                }
            }
        }
    }
    glEnd();
    glPointSize(1);

    glPointSize(5);
    glColor4f(0.5f,1,0.5f,1);
    glBegin(GL_POINTS);
    const OutVecCoord& out = *this->toModel->getX();
    for (unsigned int i=0; i<out.size(); ++i)
    {
        GL::glVertexT(out[i]);
    }
    glEnd();
    glPointSize(1);

    //glLineSize(3);
    glColor4f(0,1,0,1);
    glBegin(GL_LINES);
    for (it = grid->gridBegin(); it!=end; ++it)
    {
        typename Grid::Key p0 = it->first;
        const int x0 = p0[0]*GRIDDIM;
        const int y0 = p0[1]*GRIDDIM;
        const int z0 = p0[2]*GRIDDIM;
        typename Grid::Grid* g = it->second;
        int x,y,z;
        Cell* c = g->cell;
        for (z=0; z<GRIDDIM; z++)
        {
            for (y=0; y<GRIDDIM; y++)
            {
                for (x=0; x<GRIDDIM; x++)
                {
                    if (c->data.p[0]>0)
                    {
                        glVertex3f((x0+x)*scale,(y0+y)*scale,(z0+z)*scale);
                        glVertex3f((x0+x+1)*scale,(y0+y)*scale,(z0+z)*scale);
                    }
                    if (c->data.p[1]>0)
                    {
                        glVertex3f((x0+x)*scale,(y0+y)*scale,(z0+z)*scale);
                        glVertex3f((x0+x)*scale,(y0+y+1)*scale,(z0+z)*scale);
                    }
                    if (c->data.p[2]>0)
                    {
                        glVertex3f((x0+x)*scale,(y0+y)*scale,(z0+z)*scale);
                        glVertex3f((x0+x)*scale,(y0+y)*scale,(z0+z+1)*scale);
                    }
                    c+=DX;
                }
            }
        }
    }
    glEnd();
    //glLineSize(1);
}

} // namespace Components

} // namespace Sofa

#endif
