#include "Fluid3D.h"
#include <Sofa/Components/GL/template.h>
#include <Sofa/Components/Common/ObjectFactory.h>
#include <iostream>

namespace Sofa
{

namespace Contrib
{

namespace FluidGrid
{

void create(Fluid3D*& obj, ObjectDescription* arg)
{
    obj = new Fluid3D;
    obj->parseFields( arg->getAttributeMap() );
}

SOFA_DECL_CLASS(Fluid3D)

Creator<ObjectFactory, Fluid3D> Fluid3DClass("Fluid3D");

Fluid3D::Fluid3D()
    : nx(16), ny(16), nz(16), cellwidth(1.0f),
      f_nx ( field(&f_nx, &nx, "nx", "grid size along x axis") ),
      f_ny ( field(&f_ny, &ny, "ny", "grid size along y axis") ),
      f_nz ( field(&f_nz, &nz, "nz", "grid size along z axis") ),
      f_cellwidth ( field(&f_cellwidth, &cellwidth, "cellwidth", "width of each cell") ),
      f_height ( dataField(&f_height, 5.0f, "height", "initial fluid height") ),
      f_dir ( dataField(&f_dir, vec3(0,1,0), "dir", "initial fluid surface normal") ),
      f_tstart ( dataField(&f_tstart, 0.0f, "tstart", "starting time for fluid source") ),
      f_tstop ( dataField(&f_tstop, 60.0f, "tstop", "stopping time for fluid source") )
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
    fluid->clear(nx,ny,nz);
    fnext->clear(nx,ny,nz);
    ftemp->clear(nx,ny,nz);
    if (f_height.getValue() > 0)
    {
        //fluid->seed(f_height.getValue());
        fluid->seed(f_height.getValue(), f_dir.getValue());
        //fluid->seed(vec3(3.5,3.5,3.5), vec3(12.5,8.5,12.5));
    }
    fluid->t = -f_tstart.getValue();
    fluid->tend = f_tstop.getValue() - f_tstart.getValue();
}

void Fluid3D::reset()
{
    init();
}

void Fluid3D::updatePosition(double dt)
{
    fnext->step(fluid, ftemp, dt);
    Grid3D* p = fluid; fluid=fnext; fnext=p;
}

void Fluid3D::draw()
{
    //if (getContext()->getShowBehaviorModels())
    {
        const real dx = (nx-1)*cellwidth;
        const real dy = (ny-1)*cellwidth;
        const real dz = (nz-1)*cellwidth;
        glDisable(GL_LIGHTING);
        glColor4f(1,1,1,1);
        glBegin(GL_LINES);
        glVertex3f( 0 ,  0,  0 ); glVertex3f( dx,  0,  0 );
        glVertex3f( 0 , dy,  0 ); glVertex3f( dx, dy,  0 );
        glVertex3f( 0 ,  0, dz ); glVertex3f( dx,  0, dz );
        glVertex3f( 0 , dy, dz ); glVertex3f( dx, dy, dz );

        glVertex3f( 0 ,  0,  0 ); glVertex3f( 0 , dy,  0 );
        glVertex3f( dx,  0,  0 ); glVertex3f( dx, dy,  0 );
        glVertex3f( 0 ,  0, dz ); glVertex3f( 0 , dy, dz );
        glVertex3f( dx,  0, dz ); glVertex3f( dx, dy, dz );

        glVertex3f( 0 ,  0,  0 ); glVertex3f( 0 ,  0, dz );
        glVertex3f( dx,  0,  0 ); glVertex3f( dx,  0, dz );
        glVertex3f( 0 , dy,  0 ); glVertex3f( 0 , dy, dz );
        glVertex3f( dx, dy,  0 ); glVertex3f( dx, dy, dz );
        glEnd();
    }
    if (getContext()->getShowBehaviorModels())
    {
        glDisable(GL_LIGHTING);
        const real s = cellwidth*getContext()->getDt()*5;
        const real d = cellwidth;
        glBegin(GL_LINES);
        for (int z=0; z<nz; z++)
            for (int y=0; y<ny; y++)
                for (int x=0; x<nx; x++)
                {
                    vec3 u = fluid->get(x,y,z)->u;
                    real r;
                    r = u[0]*s;
                    if (rabs(r) > 0.001)
                    {
                        if (r>0.9) r=0.9;
                        glColor4f(1,0,0,1);
                        glVertex3f((x-0.5f  )*d, y*d, z*d);
                        glVertex3f((x-0.5f+r)*d, y*d, z*d);
                    }
                    r = u[1]*s;
                    if (rabs(r) > 0.001)
                    {
                        if (r>0.9) r=0.9;
                        glColor4f(0,1,0,1);
                        glVertex3f(x*d, (y-0.5f  )*d, z*d);
                        glVertex3f(x*d, (y-0.5f+r)*d, z*d);
                    }
                    r = u[2]*s;
                    if (rabs(r) > 0.001)
                    {
                        if (r>1) r=1;
                        glColor4f(0,0,1,1);
                        glVertex3f(x*d, y*d, (z-0.5f  )*d);
                        glVertex3f(x*d, y*d, (z-0.5f+r)*d);
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
                    glVertex3f(x*d,y*d,z*d);
                }
        glEnd();
        glPointSize(1);
    }
    if (getContext()->getShowVisualModels())
    {
        if (getContext()->getShowWireFrame())
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
        for (unsigned int i=0; i<facets.size(); i++)
        {
            for (int j=0; j<3; j++)
            {
                int idx = facets[i].p[j];
                glNormal3fv(points[idx].n);
                glVertex3fv(points[idx].p);
            }
        }
        glEnd();

        glDisable(GL_LIGHTING);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}

void Fluid3D::update()
{
    //const real invStep = (1.0f/f_cellwidth.getValue());
    points.clear();
    facets.clear();

    const real* data = fluid->levelset;

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
    for (z=1; z<nz; z++)
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

                tri=Sofa::Components::MarchingCubeTriTable[mk];
                while (*tri>=0)
                {
                    int* b = base+3*i;
                    if (addFace(b[edgepts[tri[0]]],b[edgepts[tri[1]]],b[edgepts[tri[2]]])<0)
                    {
                        std::cerr << "  mk=0x"<<std::hex<<mk<<std::dec<<" p1="<<tri[0]<<" p2="<<tri[1]<<" p3="<<tri[2]<<std::endl;
                        for (int e=0; e<12; e++) std::cerr << "  e"<<e<<"="<<b[edgepts[e]];
                        std::cerr<<std::endl;
                        for (int ddz=-1; ddz<=0; ddz++)
                            for (int ddy=-1; ddy<=0; ddy++)
                                for (int ddx=-1; ddx<=0; ddx++)
                                {
                                    std::cerr << " val("<<x+ddx<<','<<y+ddy<<','<<z+ddz<<")="<<(double)data[i+ddx*dx+ddy*dy+ddz*dz]<<std::endl;
                                }
                    }
                    tri+=3;
                }
                ++i;
            }
        }
    }
    //std::cout << points.size() << " points, "<<facets.size() <<" faces"<<std::endl;

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

bool Fluid3D::addBBox(double* minBBox, double* maxBBox)
{
    if (minBBox[0] > 0) minBBox[0] = 0;
    if (maxBBox[0] < nx) maxBBox[0] = nx;
    if (minBBox[1] > 0) minBBox[1] = 0;
    if (maxBBox[1] < ny) maxBBox[1] = ny;
    if (minBBox[2] > 0) minBBox[2] = 0;
    if (maxBBox[2] < nz) maxBBox[2] = nz;
    return true;
}

} // namespace FluidGrid

} // namespace Contrib

} // namespace Sofa
