/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaEulerianFluid/Fluid3D.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include <string.h>
#include <sofa/defaulttype/BoundingBox.h>

namespace sofa
{

namespace component
{

namespace behaviormodel
{

namespace eulerianfluid
{

using namespace sofa::helper;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(Fluid3D)

int Fluid3DClass = core::RegisterObject("Eulerian 3D fluid")
        .add< Fluid3D >()
        .addLicense("LGPL")
        .addAuthor("Jeremie Allard")
        ;

Fluid3D::Fluid3D():
    f_nx ( initData(&f_nx, (int)16, "nx", "grid size along x axis") ),
    f_ny ( initData(&f_ny, (int)16, "ny", "grid size along y axis") ),
    f_nz ( initData(&f_nz, (int)16, "nz", "grid size along z axis") ),
    f_cellwidth ( initData(&f_cellwidth, (real)1.0, "cellwidth", "width of each cell") ),
    f_center ( initData(&f_center, vec3(0,0,0), "center", "position of grid center") ),
    f_height ( initData(&f_height, 5.0f, "height", "initial fluid height") ),
    f_dir ( initData(&f_dir, vec3(0,1,0), "dir", "initial fluid surface normal") ),
    f_tstart ( initData(&f_tstart, 0.0f, "tstart", "starting time for fluid source") ),
    f_tstop ( initData(&f_tstop, 60.0f, "tstop", "stopping time for fluid source") )
{
    fluid = new Grid3D;
    fnext = new Grid3D;
    ftemp = new Grid3D;
}

Fluid3D::~Fluid3D()
{
    delete fluid;
    delete fnext;
    delete ftemp;
}

void Fluid3D::init()
{
    int& nx = *f_nx.beginEdit();
    int& ny = *f_ny.beginEdit();
    int& nz = *f_nz.beginEdit();

    fluid->clear(nx,ny,nz);
    fnext->clear(nx,ny,nz);
    ftemp->clear(nx,ny,nz);
    if (f_height.getValue() != 0)
    {
        //fluid->seed(f_height.getValue());
        fluid->seed(f_height.getValue(), f_dir.getValue());
        //fluid->seed(vec3(3.5,3.5,3.5), vec3(12.5,8.5,12.5));
    }
    fluid->t = -f_tstart.getValue();
    fluid->tend = f_tstop.getValue() - f_tstart.getValue();

    f_nx.endEdit();
    f_ny.endEdit();
    f_nz.endEdit();
}

void Fluid3D::reset()
{
    init();
}

void Fluid3D::updatePosition(SReal dt)
{
    fnext->gravity = getContext()->getGravity()/f_cellwidth.getValue();
    fnext->step(fluid, ftemp, (real)dt);
    Grid3D* p = fluid; fluid=fnext; fnext=p;
}

void Fluid3D::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    updateVisual();
    glPushMatrix();
    vec3 center = f_center.getValue();
    const int& nx = f_nx.getValue();
    const int& ny = f_ny.getValue();
    const int& nz = f_nz.getValue();
    const real& cellwidth = f_cellwidth.getValue();
    glTranslatef(center[0]-(nx-1)*cellwidth/2,center[1]-(ny-1)*cellwidth/2,center[2]-(nz-1)*cellwidth/2);
    glScalef(cellwidth,cellwidth,cellwidth);
    //if (vparams->displayFlags().getShowBehaviorModels())
    {
        glDisable(GL_LIGHTING);
        glColor4f(1,1,1,1);
        glBegin(GL_LINES);
        glVertex3i(    0,    0,    0 ); glVertex3i( nx-1,    0,    0 );
        glVertex3i(    0, ny-1,    0 ); glVertex3i( nx-1, ny-1,    0 );
        glVertex3i(    0,    0, nz-1 ); glVertex3i( nx-1,    0, nz-1 );
        glVertex3i(    0, ny-1, nz-1 ); glVertex3i( nx-1, ny-1, nz-1 );

        glVertex3i(    0,    0,    0 ); glVertex3i(    0, ny-1,    0 );
        glVertex3i( nx-1,    0,    0 ); glVertex3i( nx-1, ny-1,    0 );
        glVertex3i(    0,    0, nz-1 ); glVertex3i(    0, ny-1, nz-1 );
        glVertex3i( nx-1,    0, nz-1 ); glVertex3i( nx-1, ny-1, nz-1 );

        glVertex3i(    0,    0,    0 ); glVertex3i(    0,    0, nz-1 );
        glVertex3i( nx-1,    0,    0 ); glVertex3i( nx-1,    0, nz-1 );
        glVertex3i(    0, ny-1,    0 ); glVertex3i(    0, ny-1, nz-1 );
        glVertex3i( nx-1, ny-1,    0 ); glVertex3i( nx-1, ny-1, nz-1 );
        glEnd();
    }
    if (vparams->displayFlags().getShowBehaviorModels())
    {
        glDisable(GL_LIGHTING);
        const real s = (real)getContext()->getDt()*5;
        glBegin(GL_LINES);
        for (int z=0; z<nz; z++)
            for (int y=0; y<ny; y++)
                for (int x=0; x<nx; x++)
                {
                    vec3 u = fluid->get(x,y,z)->u;
                    real r;
                    r = u[0]*s;
                    if (rabs(r) > 0.001f)
                    {
                        if (r>0.9f) r=0.9f;
                        glColor4f(1,0,0,1);
                        glVertex3f((real)x-0.5f  , (real)y, (real)z);
                        glVertex3f((real)x-0.5f+r, (real)y, (real)z);
                    }
                    r = u[1]*s;
                    if (rabs(r) > 0.001f)
                    {
                        if (r>0.9f) r=0.9f;
                        glColor4f(0,1,0,1);
                        glVertex3f((real)x, (real)y-0.5f  , (real)z);
                        glVertex3f((real)x, (real)y-0.5f+r, (real)z);
                    }
                    r = u[2]*s;
                    if (rabs(r) > 0.001f)
                    {
                        if (r>1) r=1;
                        glColor4f(0,0,1,1);
                        glVertex3f((real)x, (real)y, (real)z-0.5f  );
                        glVertex3f((real)x, (real)y, (real)z-0.5f+r);
                    }
                }
        glEnd();
        glPointSize(3);
        glBegin(GL_POINTS);
        for (int z=0; z<nz; z++)
            for (int y=0; y<ny; y++)
                for (int x=0; x<nx; x++)
                {
                    real l = *fluid->getlevelset(x,y,z);
                    if (rabs(l)>=5) continue;
                    if (l<0)
                    {
                        glColor4f(0,1+l/5,1+l/5,1);
                    }
                    else
                    {
                        glColor4f(1-l/5,1-l/5,0,1);
                    }
                    glVertex3i(x,y,z);
                }
        glEnd();
        glPointSize(1);
    }
    if (vparams->displayFlags().getShowVisualModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glEnable(GL_LIGHTING);

        static const float color[4] = { 0.0f, 1.0f, 1.0f, 1.0f};
        glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
        static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
        static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
        glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 20);

        glBegin(GL_TRIANGLES);
        for (size_t i=0; i<facets.size(); i++)
        {
            for (size_t j=0; j<3; j++)
            {
                int idx = facets[i].p[j];
                glNormal3fv(points[idx].n.ptr());
                glVertex3fv(points[idx].p.ptr());
            }
        }
        glEnd();

        glDisable(GL_LIGHTING);
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    glPopMatrix();
#endif /* SOFA_NO_OPENGL */
}

void Fluid3D::exportOBJ(std::string name, std::ostream* out, std::ostream* /*mtl*/, int& vindex, int& nindex, int& /*tindex*/)
{
    vec3 corner = f_center.getValue() - vec3((real)f_nx.getValue()-1,(real)f_ny.getValue()-1,(real)f_nz.getValue()-1)*f_cellwidth.getValue()/2;

    *out << "g "<<name<<"\n";

    for(size_t i=0; i<points.size(); ++i)
    {
        vec3 p = points[i].p*f_cellwidth.getValue() + corner;
        *out << "v "<< std::fixed << p[0]<<' '<< std::fixed <<p[1]<<' '<< std::fixed <<p[2]<<'\n';
    }


    for(size_t i=0; i<points.size(); ++i)
        *out << "vn "<< std::fixed << points[i].n[0]<<' '<< std::fixed <<points[i].n[1]<<' '<< std::fixed <<points[i].n[2]<<'\n';


    for (size_t i = 0; i < facets.size() ; i++)
        *out << "f "<<facets[i].p[0]+vindex+1<<"//"<<facets[i].p[0]+nindex+1<<' '<< facets[i].p[1]+vindex+1<<"//"<<facets[i].p[1]+nindex+1<<' '<< facets[i].p[2]+vindex+1<<"//"<<facets[i].p[2]+nindex+1<<'\n';


    *out << std::endl;

    vindex+=points.size();
    nindex+=points.size();
}

void Fluid3D::updateVisual()
{
    points.clear();
    facets.clear();

    const real* data = fluid->levelset;
    const int& nx = f_nx.getValue();
    const int& ny = f_ny.getValue();

    planes.resize(2*nx*ny);
    CubeData *P = &(planes[0]);
    CubeData *P0 = P;
    CubeData *P1 = P+(nx*ny);
    //P0 = planes.begin()+0;
    //P1 = planes.begin()+nx*ny;

    //////// MARCHING CUBE ////////

    const real isoval = 0.0f;
    const real iso = isoval;

    const int dx = 1;
    const int dy = nx;
    const int dz = nx*ny;

    int x,y,z,i,mk;
    const int *tri;


    i=0;
    // First plane
    memset(P1,-1,nx*ny*sizeof(CubeData));
    {
        z=0;
        // First line
        {
            y=0;
            ++i;
            for(x=1; x<nx; x++)
            {
                if ((data[i]>isoval)^(data[i-dx]>isoval))
                {
                    P1[i].p[0] = addPoint<0>(x,y,z,data[i],data[i-dx],iso);
                }
                ++i;
            }
        }
        for(y=1; y<ny; y++)
        {
            // First column
            x=0;
            if ((data[i]>isoval)^(data[i-dy]>isoval))
            {
                P1[i].p[1] = addPoint<1>(x,y,z,data[i],data[i-dy],iso);
            }
            ++i;

            for(x=1; x<nx; x++)
            {
                if ((data[i]>isoval)^(data[i-dx]>isoval))
                {
                    P1[i].p[0] = addPoint<0>(x,y,z,data[i],data[i-dx],iso);
                }
                if ((data[i]>isoval)^(data[i-dy]>isoval))
                {
                    P1[i].p[1] = addPoint<1>(x,y,z,data[i],data[i-dy],iso);
                }
                ++i;
            }
        }
    }
    for (z=1; z<f_nz.getValue(); z++)
    {
        i=0; data+=dz;

        { CubeData* p=P0; P0=P1; P1=p; }
        memset(P1,-1,nx*ny*sizeof(CubeData));

        int edgepts[12];
        int* base = &(P[0].p[0]);
        edgepts[0] = &(P0[-dy].p[0])-base;
        edgepts[1] = &(P0[0  ].p[1])-base;
        edgepts[2] = &(P0[0  ].p[0])-base;
        edgepts[3] = &(P0[-dx].p[1])-base;
        edgepts[4] = &(P1[-dy].p[0])-base;
        edgepts[5] = &(P1[0  ].p[1])-base;
        edgepts[6] = &(P1[0  ].p[0])-base;
        edgepts[7] = &(P1[-dx].p[1])-base;
        edgepts[8] =  &(P1[-dx-dy].p[2])-base;
        edgepts[9] =  &(P1[   -dy].p[2])-base;
        edgepts[10] = &(P1[0     ].p[2])-base;
        edgepts[11] = &(P1[-dx   ].p[2])-base;

        // First line
        {
            y=0;
            x=0;
            if ((data[i]>isoval)^(data[i-dz]>isoval))
            {
                P1[i].p[2] = addPoint<2>(x,y,z,data[i],data[i-dz],iso);
            }
            ++i;
            for(x=1; x<nx; x++)
            {
                if ((data[i]>isoval)^(data[i-dx]>isoval))
                {
                    P1[i].p[0] = addPoint<0>(x,y,z,data[i],data[i-dx],iso);
                }
                if ((data[i]>isoval)^(data[i-dz]>isoval))
                {
                    P1[i].p[2] = addPoint<2>(x,y,z,data[i],data[i-dz],iso);
                }
                ++i;
            }
        }
        for(y=1; y<ny; y++)
        {
            // First column
            x=0;
            if ((data[i]>isoval)^(data[i-dy]>isoval))
            {
                P1[i].p[1] = addPoint<1>(x,y,z,data[i],data[i-dy],iso);
            }
            if ((data[i]>isoval)^(data[i-dz]>isoval))
            {
                P1[i].p[2] = addPoint<2>(x,y,z,data[i],data[i-dz],iso);
            }
            ++i;

            for(x=1; x<nx; x++)
            {
                if ((data[i]>isoval)^(data[i-dx]>isoval))
                {
                    P1[i].p[0] = addPoint<0>(x,y,z,data[i],data[i-dx],iso);
                }
                if ((data[i]>isoval)^(data[i-dy]>isoval))
                {
                    P1[i].p[1] = addPoint<1>(x,y,z,data[i],data[i-dy],iso);
                }
                if ((data[i]>isoval)^(data[i-dz]>isoval))
                {
                    P1[i].p[2] = addPoint<2>(x,y,z,data[i],data[i-dz],iso);
                }

                // All points should now be created

                if (data[i-dx-dy-dz]>iso) mk = 1; else mk=0;
                if (data[i   -dy-dz]>iso) mk|= 2;
                if (data[i      -dz]>iso) mk|= 4;
                if (data[i-dx   -dz]>iso) mk|= 8;
                if (data[i-dx-dy   ]>iso) mk|= 16;
                if (data[i   -dy   ]>iso) mk|= 32;
                if (data[i         ]>iso) mk|= 64;
                if (data[i-dx      ]>iso) mk|= 128;

                tri = helper::MarchingCubeTriTable[mk];
                while (*tri>=0)
                {
                    int* b = base+3*i;
                    if (addFace(b[edgepts[tri[0]]],b[edgepts[tri[1]]],b[edgepts[tri[2]]])<0)
                    {
                        std::stringstream tmp;
                        tmp<< "  mk=0x"<<std::hex<<mk<<std::dec<<" p1="<<tri[0]<<" p2="<<tri[1]<<" p3="<<tri[2];
                        for (int e=0; e<12; e++)
                            tmp << "  e"<<e<<"="<<b[edgepts[e]];
                        tmp<<msgendl;
                        for (int ddz=-1; ddz<=0; ddz++)
                            for (int ddy=-1; ddy<=0; ddy++)
                                for (int ddx=-1; ddx<=0; ddx++)
                                {
                                    tmp << " val("<<x+ddx<<','<<y+ddy<<','<<z+ddz<<")="<<(double)data[i+ddx*dx+ddy*dy+ddz*dz]<<msgendl;
                                }

                        msg_info() << tmp.str();
                    }
                    tri+=3;
                }
                ++i;
            }
        }
    }

    for (unsigned int i=0; i<points.size(); i++)
    {
        points[i].n.clear();
    }

    for (unsigned int i=0; i<facets.size(); i++)
    {
        Vec3f n = cross(points[facets[i].p[1]].p-points[facets[i].p[0]].p,points[facets[i].p[2]].p-points[facets[i].p[0]].p);
        n.normalize();
        points[facets[i].p[0]].n += n;
        points[facets[i].p[1]].n += n;
        points[facets[i].p[2]].n += n;
    }

    for (unsigned int i=0; i<points.size(); i++)
        points[i].n.normalize();
}

void Fluid3D::computeBBox(const core::ExecParams*  params , bool /*onlyVisible*/)
{
    vec3 center = f_center.getValue(params);
    const real& cellwidth = f_cellwidth.getValue(params);
    SReal size[3] = { (f_nx.getValue()-1)*cellwidth, (f_ny.getValue(params)-1)*cellwidth, (f_nz.getValue(params)-1)*cellwidth };
    SReal minBBox[3] = { center[0]-size[0]/2, center[1]-size[1]/2, center[2]-size[2]/2 };
    SReal maxBBox[3];
    for (int c=0; c<3; c++)
    {
        maxBBox[c] = minBBox[c]+size[c];
    }
    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
}

} // namespace eulerianfluid

} // namespace behaviormodel

} // namespace component

} // namespace sofa

