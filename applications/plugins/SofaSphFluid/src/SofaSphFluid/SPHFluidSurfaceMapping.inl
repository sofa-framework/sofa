/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SPHFLUIDSURFACEMAPPING_INL

#include <SofaSphFluid/SPHFluidSurfaceMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaSphFluid/SpatialGridContainer.inl>

#include <sofa/helper/rmath.h>
#include <sofa/helper/MarchingCubeUtility.h> // for marching cube tables

#include <sofa/simulation/Node.h>

#include <map>
#include <list>


namespace sofa
{

namespace component
{

namespace mapping
{


template <class In, class Out>
SPHFluidSurfaceMapping<In, Out>::SPHFluidSurfaceMapping()
    : Inherit()
    , d_mStep(initData(&d_mStep, 0.5, "step", "Step"))
    , d_mRadius(initData(&d_mRadius, 2.0, "radius", "Radius"))
    , d_mIsoValue(initData(&d_mIsoValue, 0.5, "isoValue", "Iso Value"))
    , sph(nullptr)
    , grid(nullptr)
    , firstApply(true)
{

}

template <class In, class Out>
double SPHFluidSurfaceMapping<In, Out>::getStep() const
{
    return d_mStep.getValue();
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In, Out>::setStep(double val)
{
    d_mStep.setValue(val);
}

template <class In, class Out>
double SPHFluidSurfaceMapping<In, Out>::getRadius() const
{
    return d_mRadius.getValue();
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In, Out>::setRadius(double val)
{
    d_mRadius.setValue(val);
}

template <class In, class Out>
double SPHFluidSurfaceMapping<In, Out>::getIsoValue() const
{
    return d_mIsoValue.getValue();
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In, Out>::setIsoValue(double val)
{
    d_mIsoValue.setValue(val);
}


template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::init()
{
    this->Inherit::init();
    simulation::Node* node = dynamic_cast<simulation::Node*>(this->getFrom()[0]->getContext());
    if (node)
    {
        //the following line produces a compilation error with GCC 3.3 :(
        //sph = node->getNodeObject<SPHForceField>();
        node->getNodeObject(sph);
    }
    if (sph)
    {
        //d_mRadius.getValue() = sph->getParticleRadius();
        if (d_mIsoValue.getValue() == 0.5)
            d_mIsoValue.setValue( d_mIsoValue.getValue()/ sph->getParticleFieldConstant((InReal)d_mRadius.getValue()));
    }

    grid = new Grid((InReal)d_mStep.getValue());
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::createPoints(OutVecCoord& out, OutVecDeriv* normals, const GridEntry& g, int x, int y, int z, Cell* c, const Cell* cx, const Cell* cy, const Cell* cz, const OutReal isoval)
{
    if (c->data.val>isoval)
    {
        if (!(cx->data.val>isoval))
            c->data.p[0] = addPoint<0>(out,normals,g, x,y,z,c->data.val,cx->data.val,isoval);
        if (!(cy->data.val>isoval))
            c->data.p[1] = addPoint<1>(out,normals,g, x,y,z,c->data.val,cy->data.val,isoval);
        if (!(cz->data.val>isoval))
            c->data.p[2] = addPoint<2>(out,normals,g, x,y,z,c->data.val,cz->data.val,isoval);
    }
    else
    {
        if (cx->data.val>isoval)
            c->data.p[0] = addPoint<0>(out,normals,g, x,y,z,c->data.val,cx->data.val,isoval);
        if (cy->data.val>isoval)
            c->data.p[1] = addPoint<1>(out,normals,g, x,y,z,c->data.val,cy->data.val,isoval);
        if (cz->data.val>isoval)
            c->data.p[2] = addPoint<2>(out,normals,g, x,y,z,c->data.val,cz->data.val,isoval);
    }
}


template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::createFaces(OutVecCoord& out, OutVecDeriv* /*normals*/, const Cell** cells, const OutReal isoval)
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

    const int* tri=helper::MarchingCubeTriTable[mk];
    while (*tri>=0)
    {
        if (addFace(cells[edgecell[tri[0]]]->data.p[edgepts[tri[0]]],
                cells[edgecell[tri[1]]]->data.p[edgepts[tri[1]]],
                cells[edgecell[tri[2]]]->data.p[edgepts[tri[2]]], int(out.size()))<0)
        {
            msg_error() << "  mk=0x" << std::hex << mk << std::dec << " p1=" << tri[0] << " p2=" << tri[1] << " p3=" << tri[2];
            for (int e = 0; e < 12; e++)
                msg_error() << "  e" << e << "=" << cells[edgecell[e]]->data.p[edgepts[e]];
        }
        tri+=3;
    }
}


template <class In, class Out>
typename SPHFluidSurfaceMapping<In, Out>::OutReal SPHFluidSurfaceMapping<In, Out>::getValue(const SubGrid* g, int cx, int cy, int cz)
{
    if (cx < 0) { g = g->neighbors[0]; cx += GRIDDIM; }
    else if (cx >= GRIDDIM) { g = g->neighbors[1]; cx -= GRIDDIM; }
    if (cy < 0) { g = g->neighbors[2]; cy += GRIDDIM; }
    else if (cy >= GRIDDIM) { g = g->neighbors[3]; cy -= GRIDDIM; }
    if (cz < 0) { g = g->neighbors[4]; cz += GRIDDIM; }
    else if (cz >= GRIDDIM) { g = g->neighbors[5]; cz -= GRIDDIM; }
    return g->cell[(cz*GRIDDIM + cy)*GRIDDIM + cx].data.val;
}


template <class In, class Out>
typename SPHFluidSurfaceMapping<In, Out>::OutDeriv SPHFluidSurfaceMapping<In, Out>::calcGrad(const GridEntry& g, int x, int y, int z)
{
    x -= g.first[0] * GRIDDIM;
    y -= g.first[1] * GRIDDIM;
    z -= g.first[2] * GRIDDIM;
    OutDeriv n;
    n[0] = getValue(g.second, x + 1, y, z) - getValue(g.second, x - 1, y, z);
    n[1] = getValue(g.second, x, y + 1, z) - getValue(g.second, x, y - 1, z);
    n[2] = getValue(g.second, x, y, z + 1) - getValue(g.second, x, y, z - 1);
    return n;
}


template <class In, class Out>
int SPHFluidSurfaceMapping<In, Out>::addFace(int p1, int p2, int p3, int nbp)
{
    if ((unsigned)p1<(unsigned)nbp &&
        (unsigned)p2<(unsigned)nbp &&
        (unsigned)p3<(unsigned)nbp)
    {
        SeqTriangles& triangles = *seqTriangles.beginEdit();
        int f = int(triangles.size());
        triangles.push_back(Triangle(p1, p3, p2));
        seqTriangles.endEdit();
        return f;
    }
    else
    {
        msg_error() << "Invalid face " << p1 << " " << p2 << " " << p3;
        return -1;
    }
}


template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    OutVecCoord& out = *dOut.beginEdit();
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;

    //if (!sph) return;
    if (!grid) return;
    //const InReal invStep = (InReal)(1/d_mStep.getValue());
    Data< OutVecDeriv > *normals_data = this->toModel->write(core::VecDerivId::normal());
    OutVecDeriv *normals;
    //if toModel is not a VisualModelImpl
    //(consequently, it does not have any normal vector)

    if(normals_data == nullptr)
    {
        normals = new OutVecDeriv();
    }
    else
        normals = normals_data->beginEdit();

    out.resize(0);

    if (normals)
        normals->resize(0);

    clear();

    if (in.size()==0)
        return;

    const InReal r = (InReal)(getRadius()); // / d_mStep.getValue());
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
                    createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, c+DX, c+DY, c+DZ, isoval);
                    c+=DX;
                }
                // X border
                createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, cx1, c+DY, c+DZ, isoval);
                c+=DX;
                cx1+=DY;
            }
            // Y BORDER
            {
                for (x=0; x<GRIDDIM-1; x++)
                {
                    createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, c+DX, cy1, c+DZ, isoval);
                    c+=DX;
                    cy1+=DX;
                }
                // X border
                createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, cx1, cy1, c+DZ, isoval);
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
                createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, c+DX, c+DY, cz1, isoval);
                c+=DX;
                cz1+=DX;
            }
            // X border
            createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, cx1, c+DY, cz1, isoval);
            c+=DX;
            cx1+=DY;
            cz1+=DX;
        }
        // Y BORDER
        {
            for (x=0; x<GRIDDIM-1; x++)
            {
                createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, c+DX, cy1, cz1, isoval);
                c+=DX;
                cy1+=DX;
                cz1+=DX;
            }
            // X border
            createPoints(out, normals, *it, x0+x, y0+y, z0+z, c, cx1, cy1, cz1, isoval);
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
                    createFaces(out, normals, cells, isoval);
                    c+=DX;
                }
                // X border
                cells[0] = c;               cells[1] = cx1;             cells[2] = c+DY;            cells[3] = cx1+DY;
                cells[4] = cells[0]+DZ;     cells[5] = cells[1]+DZ;     cells[6] = cells[2]+DZ;     cells[7] = cells[3]+DZ;
                createFaces(out, normals, cells, isoval);
                c+=DX;
                cx1+=DY;
            }
            // Y BORDER
            {
                for (x=0; x<GRIDDIM-1; x++)
                {
                    cells[0] = c;               cells[1] = c+DX;            cells[2] = cy1;             cells[3] = cy1+DX;
                    cells[4] = cells[0]+DZ;     cells[5] = cells[1]+DZ;     cells[6] = cells[2]+DZ;     cells[7] = cells[3]+DZ;
                    createFaces(out, normals, cells, isoval);
                    c+=DX;
                    cy1+=DX;
                }
                // X border
                cells[0] = c;               cells[1] = cx1;            cells[2] = cy1;              cells[3] = cx1y1;
                cells[4] = cells[0]+DZ;     cells[5] = cells[1]+DZ;     cells[6] = cells[2]+DZ;     cells[7] = cells[3]+DZ;
                createFaces(out, normals, cells, isoval);
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
                createFaces(out, normals, cells, isoval);
                c+=DX;
                cz1+=DX;
            }
            // X border
            cells[0] = c;               cells[1] = cx1;            cells[2] = c+DY;            cells[3] = cx1+DY;
            cells[4] = cz1;             cells[5] = cx1z1;          cells[6] = cz1+DY;          cells[7] = cx1z1+DY;
            createFaces(out, normals, cells, isoval);
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
                createFaces(out, normals, cells, isoval);
                c+=DX;
                cy1+=DX;
                cz1+=DX;
                cy1z1+=DX;
            }
            // X border
            cells[0] = c;               cells[1] = cx1;            cells[2] = cy1;            cells[3] = cx1y1;
            cells[4] = cz1;             cells[5] = cx1z1;          cells[6] = cy1z1;          cells[7] = cx1y1z1;
            createFaces(out, normals, cells, isoval);
        }
    }

    if (normals_data == nullptr)
    {
        std::cout << "no Normal data" << std::endl;
        delete normals;
    }
    else
    {
        normals_data->endEdit();
    }

    dOut.endEdit();
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& /*dOut*/, const Data<InVecDeriv>& /*dIn*/)
{
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& /*dOut*/, const Data<OutVecDeriv>& /*dIn*/)
{
}

template <class In, class Out>
void SPHFluidSurfaceMapping<In,Out>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings())
        return;
    if (!grid)
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    grid->draw(vparams);

    float scale = (float)d_mStep.getValue();
    typename Grid::iterator end = grid->gridEnd();
    typename Grid::iterator it;

    std::vector< sofa::type::Vec3 > points1;
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
                    if (c->data.val > d_mIsoValue.getValue())
                        points1.push_back(type::Vec3((x0+x)*scale,(y0+y)*scale,(z0+z)*scale));
                    c+=DX;
                }
            }
        }
    }
    vparams->drawTool()->drawPoints(points1, 3, sofa::type::RGBAColor(1.,1.,1.,1.));


    std::vector< sofa::type::Vec3 > points2;
    const OutVecCoord& out = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<out.size(); ++i)
    {
        points2.push_back(out[i]);
    }
    vparams->drawTool()->drawPoints(points2, 5, sofa::type::RGBAColor(0.5,1,0.5,1));


    std::vector< sofa::type::Vec3 > points3;
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
                        points3.push_back(type::Vec3((x0+x)*scale,(y0+y)*scale,(z0+z)*scale));
                        points3.push_back(type::Vec3((x0+x+1)*scale,(y0+y)*scale,(z0+z)*scale));
                    }
                    if (c->data.p[1]>0)
                    {
                        points3.push_back(type::Vec3((x0+x)*scale,(y0+y)*scale,(z0+z)*scale));
                        points3.push_back(type::Vec3((x0+x)*scale,(y0+y+1)*scale,(z0+z)*scale));
                    }
                    if (c->data.p[2]>0)
                    {
                        points3.push_back(type::Vec3((x0+x)*scale,(y0+y)*scale,(z0+z)*scale));
                        points3.push_back(type::Vec3((x0+x)*scale,(y0+y)*scale,(z0+z+1)*scale));
                    }
                    c+=DX;
                }
            }
        }
    }
    vparams->drawTool()->drawLines(points3, 1, sofa::type::RGBAColor(0.,1.,0.,1.));
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
