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
#include <iostream>
#include <SofaEulerianFluid/Grid2D.h>
#include <sofa/core/visual/VisualParams.h>
#include <cstring>


// set to true/false to activate extra verbose FMM.
#define EMIT_EXTRA_FMM_MESSAGE false

namespace sofa
{

namespace component
{

namespace behaviormodel
{

namespace eulerianfluid
{

using namespace sofa::helper;

const unsigned long* Grid2D::obstacles = NULL;

// For loop macros

#define FOR_ALL_CELLS(grid,cmd)               \
{                                             \
  int ind = index(0,0);                       \
    for (int y=0;y<ny;y++)                    \
      for (int x=0;x<nx;x++,ind+=index(1,0))  \
      {                                       \
    cmd;                                  \
      }                                       \
}

#define FOR_READ_INNER_CELLS(cmd)             \
{                                             \
  int ind = index(1,1);                       \
    for (int y=1;y<ny-1;y++,ind+=index(2,0))  \
      for (int x=1;x<nx-1;x++,ind+=index(1,0))\
      {                                       \
    cmd;                                  \
      }                                       \
}

#define FOR_INNER_CELLS(grid,cmd)             \
{                                             \
  int ind = index(1,1);                       \
    for (int y=1;y<ny-1;y++,ind+=index(2,0))  \
      for (int x=1;x<nx-1;x++,ind+=index(1,0))\
      {                                       \
    cmd;                                  \
      }                                       \
}

// Surface cells  are inner  cells and borders  between a  fluid inner
// cell and an empty out cell (right or bottom side)

#define FOR_SURFACE_CELLS(grid,cmd)           \
{                                             \
  int ind = index(1,1);                       \
    for (int y=1;y<ny-1;y++,ind+=index(2,0))  \
      for (int x=1;x<nx-1;x++,ind+=index(1,0))\
      {                                       \
    cmd;                                  \
      }                                       \
}

Grid2D::Grid2D()
    : nx(0), ny(0), ncell(0),
      fdata(NULL), pressure(NULL), levelset(NULL),
      t(0), tend(60),
      max_pressure(0.0),
      fmm_status(NULL),
      fmm_heap(NULL),
      fmm_heap_size(0)
{
    bcell.u.clear();
    bcell.type=-1;
}

Grid2D::~Grid2D()
{
    if (fdata!=NULL) delete[] fdata;
    if (pressure!=NULL) delete[] pressure;
    if (levelset!=NULL) delete[] levelset;
    if (fmm_status!=NULL) delete[] fmm_status;
    if (fmm_heap!=NULL) delete[] fmm_heap;
}

void Grid2D::clear(int _nx, int _ny)
{
    t = 0;
    tend = 40;

    int old_ncell=ncell;
    nx=_nx; ny=_ny;
    ncell = nx*ny;

    if (ncell != old_ncell)
    {
        if (fdata!=NULL) delete[] fdata;
        if (ncell>0)  fdata = new Cell[ncell];
        else          fdata = NULL;
        if (pressure!=NULL) delete[] pressure;
        if (ncell>0)  pressure = new real[ncell];
        else          pressure = NULL;
        if (levelset!=NULL) delete[] levelset;
        if (ncell>0)  levelset = new real[ncell];
        else          levelset = NULL;
        if (fmm_status!=NULL) delete[] fmm_status;
        if (ncell>0)  fmm_status = new int[ncell];
        else          fmm_status = NULL;
        if (fmm_heap!=NULL) delete[] fmm_heap;
        if (ncell>0)  fmm_heap = new int[ncell];
        else          fmm_heap = NULL;
    }
    if (ncell>0)
        memset(fdata,0,ncell*sizeof(Cell));
    if (ncell>0)
        memset(pressure,0,ncell*sizeof(real));
    if (ncell>0)
        memset(levelset,0,ncell*sizeof(real));
}

void Grid2D::seed(real height)
{
    //seed(vec2(0,0), vec2(nx,height));
    FOR_ALL_CELLS(fdata,
    {
        real d = y - height;
        levelset[ind] = d;
    });
}

void Grid2D::seed(real height, vec2 normal)
{
    normal.normalize();
    FOR_ALL_CELLS(fdata,
    {
        real d = vec2((real)x,(real)y)*normal - height;
        levelset[ind] = d;
    });
}

void Grid2D::seed(vec2 p0, vec2 p1, vec2 velocity)
{
    if (p0[0]<0.5f) p0[0]=0.5f;
    if (p0[1]<0.5f) p0[1]=0.5f;
    if (p1[0]>nx-1.5f) p1[0]=nx-1.5f;
    if (p1[1]>ny-1.5f) p1[1]=ny-1.5f;
    if (p0[0]>=p1[0]) return;
    if (p0[1]>=p1[1]) return;
    msg_info("Grid2D") << "p0="<<p0<<" p1="<<p1;
    vec2 center = (p0+p1)*0.5f;
    vec2 dim = (p1-p0)*0.5f;
    FOR_ALL_CELLS(fdata,
    {
        vec2 v ((real)x,(real)y);
        v -= center;
        v[0] = rabs(v[0]) - dim[0];
        v[1] = rabs(v[1]) - dim[1];
        real d;
        if (v[0] <= 0 && v[1] <= 0)
        {
            d = rmax(v[0],v[1]);
            fdata[ind].u = velocity;
        }
        else
        {
            if (v[0] < 0) v[0] = 0;
            if (v[1] < 0) v[1] = 0;
            d = v.norm();
        }
        levelset[ind] = d;
    });
}

void Grid2D::step(Grid2D* prev, Grid2D* temp, real dt, real diff)
{
    t = prev->t+dt;
    tend = prev->tend;
    step_init(prev, temp, dt, diff);      // init fluid obstacles
    step_levelset(prev, temp, dt, diff); // advance levelset
    step_forces(prev, temp, dt, diff);    // init fluid u with prev u, particles and gravity
    step_surface(prev, temp, dt, diff);   // calc fluid u at air/fluid surfaces
    step_advect(prev, temp, dt, diff);    // advance fluid u to temp u
    step_diffuse(prev, temp, dt, diff);   // calc diffusion of temp u to fluid u
    step_project(prev, temp, dt, diff);   // calc pressure and project fluid u to divergent free field. use temp as temporary scalar fields

    // And that should be it!
}

//////////////////////////////////////////////////////////////////
//// Init grid with obstacles

void Grid2D::step_init(const Grid2D* /*prev*/, Grid2D* /*temp*/, real /*dt*/, real /*diff*/)
{
    // Currently: only borders are obstacle

    int bsize = 1;

    const unsigned char* obs = (const unsigned char*)obstacles;
    int lnsize = (nx+7)/8;

    FOR_ALL_CELLS(fdata,
    {
        if (x<bsize || y<bsize || x>=nx-bsize || y>=ny-bsize
        || (obs!=NULL && ((obs[(y)*lnsize+((x)>>3)])&(1<<((x)&7))))
           )
        {
            fdata[ind].type = PART_WALL;
        }
        else
        {
            fdata[ind].type = PART_EMPTY;
        }
        levelset[ind] = 5;
    });
}

////////////////////////////////////////////////////////////////
//// Move the levelset

#define RMAX 1e10

void Grid2D::step_levelset(Grid2D* prev, Grid2D* temp, real dt, real /*diff*/)
{
    // advect levelset into temp

    // Modified Eulerian / Midpoint method
    // Carlson Thesis page 22
    FOR_INNER_CELLS(levelset,
    {
        //if (prev->fdata[ind].type != PART_WALL && rabs(prev->levelset[ind]) < 5)
        if (rabs(prev->levelset[ind]) < 5)
        {
            vec2 xn ( (real)x, (real)y );
            vec2 un = prev->interp(xn);
            vec2 xn1 = xn - un*dt; // xn1_2 at this time

            // TODO: Check if xn1_2 is not in empty cell
            un = (un + prev->interp(xn1))*0.5;
            xn1 = xn - un*dt;

            temp->levelset[ind] = prev->getlevelset(xn1);
        }
        else
            temp->levelset[ind] = prev->levelset[ind];
    });

    // fill border levelset using neighbors

    FOR_ALL_CELLS(fmm_status,
    {
        fmm_status[ind] = FMM_FAR;
        if (fdata[ind].type == PART_WALL)
        {
            // find a neighbor cell
            real phi = 0;
            int n = 0;
            for (int dy=-1; dy<=1; dy++)
                if ((unsigned)(y+dy)<(unsigned)ny)
                    for (int dx=-1; dx<=1; dx++)
                        if ((unsigned)(x+dx)<(unsigned)nx)
                        {
                            int ind2 = ind+index(dx,dy);
                            if (fdata[ind2].type != PART_WALL)
                            {
                                phi += temp->levelset[ind2];
                                ++n;
                            }
                        }
            temp->levelset[ind] = phi/n;
        }
    });

    // Levelset Reinitilization using Fast Marching Method
    //TODO do not reinit at every steps

    fmm_heap_size = 0;
    const int cmax[2] = { nx-1, ny-1 };
    const int dind[2] = { 1, nx };

    // Compute all known points
    FOR_ALL_CELLS(fdata,
    {
        int c[2]; c[0] = x; c[1] = y;
        bool known = false;
        real phi0 = temp->levelset[ind];
        real inv_dist2 = 0;
        for (int dim = 0; dim < 2; dim++)
        {
            real dist = RMAX;
            bool border = false;
            if (c[dim]>0)
            {
                real phi1 = temp->levelset[ind-dind[dim]];
                if (phi1*phi0 < 0)
                {
                    dist = phi0/(phi0-phi1);
                    border = true;
                }
            }
            if (c[dim]<cmax[dim])
            {
                real phi1 = temp->levelset[ind+dind[dim]];
                if (phi1*phi0 < 0)
                {
                    real d = phi0/(phi0-phi1);
                    if (d < dist) dist = d;
                    border = true;
                }
            }
            if (border)
            {
                known = true;
                inv_dist2 += 1 / (dist*dist);
            }
        }
        if (known)
        {
            //real phi = 1 / sqrt(inv_dist2);
            real phi = rabs(phi0);
            levelset[ind] = phi;
            fmm_status[ind] = FMM_KNOWN;
        }
        else
        {
            levelset[ind] = 5; //RMAX;
            fmm_status[ind] = FMM_FAR;
        }
    });

    // Update known points neighbors
    FOR_ALL_CELLS(fdata,
    {
        if (fmm_status[ind] == FMM_KNOWN)
        {
            int c[2]; c[0] = x; c[1] = y;
            real phi1 = levelset[ind];
            phi1+=1;
            for (int dim = 0; dim < 2; dim++)
            {
                if (c[dim]>0)
                {
                    int ind2 = ind-dind[dim];
                    if (fmm_status[ind2] >= FMM_FAR)
                    {
                        real phi2 = levelset[ind2];
                        if ((phi1) < (phi2))
                        {
                            levelset[ind2] = phi1;
                            fmm_push(ind2); // create or update the corresponding entry in the heap
                        }
                    }
                }
                if (c[dim]<cmax[dim])
                {
                    int ind2 = ind+dind[dim];
                    if (fmm_status[ind2] >= FMM_FAR)
                    {
                        real phi2 = levelset[ind2];
                        if ((phi1) < (phi2))
                        {
                            levelset[ind2] = phi1;
                            fmm_push(ind2); // create or update the corresponding entry in the heap
                        }
                    }
                }
            }
        }
    });

    while (fmm_heap_size > 0)
    {
        int ind = fmm_pop();
        real phi1 = levelset[ind]+1;
        if ((phi1) >= 5) break;
        int c[2] = { ind % nx, (ind / nx) };
        for (int dim = 0; dim < 2; dim++)
        {
            if (c[dim]>0)
            {
                int ind2 = ind-dind[dim];
                if (fmm_status[ind2] >= FMM_FAR)
                {
                    real phi2 = levelset[ind2];
                    if ((phi1) < (phi2))
                    {
                        levelset[ind2] = phi1;
                        fmm_push(ind2); // create or update the corresponding entry in the heap
                    }
                }
            }
            if (c[dim]<cmax[dim])
            {
                int ind2 = ind+dind[dim];
                if (fmm_status[ind2] >= FMM_FAR)
                {
                    real phi2 = levelset[ind2];
                    if ((phi1) < (phi2))
                    {
                        levelset[ind2] = phi1;
                        fmm_push(ind2); // create or update the corresponding entry in the heap
                    }
                }
            }
        }
    }

    FOR_ALL_CELLS(levelset,
    {
        if(temp->levelset[ind] < 0)
        {
            levelset[ind] = -levelset[ind];
            if (fdata[ind].type == PART_EMPTY)
                fdata[ind].type = PART_FULL;
        }
    });
}

inline void Grid2D::fmm_swap(int entry1, int entry2)
{
    int ind1 = fmm_heap[entry1];
    int ind2 = fmm_heap[entry2];
    fmm_heap[entry1] = ind2;
    fmm_heap[entry2] = ind1;
    fmm_status[ind1] = entry2 + FMM_FRONT0;
    fmm_status[ind2] = entry1 + FMM_FRONT0;
}

int Grid2D::fmm_pop()
{
    int res = fmm_heap[0];

    --fmm_heap_size;
    if (fmm_heap_size>0)
    {
        fmm_swap(0, fmm_heap_size);
        int i=0;
        real phi = (levelset[fmm_heap[i]]);
        while (i*2+1 < fmm_heap_size)
        {
            real phi1 = (levelset[fmm_heap[i*2+1]]);
            if (i*2+2 < fmm_heap_size)
            {
                real phi2 = (levelset[fmm_heap[i*2+2]]);
                if (phi1 < phi)
                {
                    if (phi1 < phi2)
                    {
                        fmm_swap(i, i*2+1);
                        i = i*2+1;
                    }
                    else
                    {
                        fmm_swap(i, i*2+2);
                        i = i*2+2;
                    }
                }
                else if (phi2 < phi)
                {
                    fmm_swap(i, i*2+2);
                    i = i*2+2;
                }
                else break;
            }
            else if (phi1 < phi)
            {
                fmm_swap(i, i*2+1);
                i = i*2+1;
            }
            else break;
        }
    }

    fmm_status[res] = FMM_KNOWN;
    return res;
}

void Grid2D::fmm_push(int index)
{
    real phi = (levelset[index]);
    int i;
    if (fmm_status[index] >= FMM_FRONT0)
    {
        i = fmm_status[index] - FMM_FRONT0;

        while (i>0 && phi < (levelset[fmm_heap[(i-1)/2]]))
        {
            fmm_swap(i,(i-1)/2);
            i = (i-1)/2;
        }
        while (i*2+1 < fmm_heap_size)
        {
            real phi1 = (levelset[fmm_heap[i*2+1]]);
            if (i*2+2 < fmm_heap_size)
            {
                real phi2 = (levelset[fmm_heap[i*2+2]]);
                if (phi1 < phi)
                {
                    if (phi1 < phi2)
                    {
                        fmm_swap(i, i*2+1);
                        i = i*2+1;
                    }
                    else
                    {
                        fmm_swap(i, i*2+2);
                        i = i*2+2;
                    }
                }
                else if (phi2 < phi)
                {
                    fmm_swap(i, i*2+2);
                    i = i*2+2;
                }
                else break;
            }
            else if (phi1 < phi)
            {
                fmm_swap(i, i*2+1);
                i = i*2+1;
            }
            else break;
        }
    }
    else
    {
        i = fmm_heap_size;
        ++fmm_heap_size;
        fmm_heap[i] = index;
        fmm_status[index] = i;
        while (i>0 && phi < (levelset[fmm_heap[(i-1)/2]]))
        {
            fmm_swap(i,(i-1)/2);
            i = (i-1)/2;
        }
    }
}

//////////////////////////////////////////////////////////////////
//// Set velocity for previously empty cells
//// And boundary conditions for static obstacles
//// Also add forces (gravity)

void Grid2D::step_forces(const Grid2D* prev, Grid2D* /*temp*/, real dt, real /*diff*/, real /*scale*/)
{
    // Solid immovable obstacles set all velocity inside and touching them to 0
    // (free-slip condition)
    // Carlson Thesis page 23

    vec2 f(0,-5*dt);

    FOR_INNER_CELLS(fdata,
    {
        vec2 u = f;
        int p0 = fdata[ind].type;
        if (p0 < 0)
            u.clear();
        else
        {
            // not an obstacle
            int p1;
            // X Axis
            p1 = fdata[ind+index(-1,0)].type;
            if (p1>=0 && p0+p1>0) // this face is now in the fluid
            {
                u[0] += prev->fdata[ind].u[0]; // was in the fluid in previous step
            }
            else //if (p1<0)
                u[0] = 0; // Obstacle
            // Y Axis
            p1 = fdata[ind+index(0,-1)].type;
            if (p1>=0 && p0+p1>0) // this face is now in the fluid
            {
                u[1] += prev->fdata[ind].u[1]; // was in the fluid in previous step
            }
            else //if (p1<0)
                u[1] = 0; // Obstacle
        }
        fdata[ind].u = u;
    });

    if (t > 0 && t < tend)
    {
        union
        {
            float f;
            unsigned int i;
        } tmp;
        tmp.f = 2*t;
        srand(tmp.i);
        vec2 v(4,0);
        int cx = 1*ny/4;
        int cy = 3*ny/4;
        real r = ny/8.0f;
        if (t < 1) r *= t;
        else if (t>tend-1) r*= tend-t;
        int ir = rceil(r)+5;

        for (int y=cy-ir; y<=cy+ir; y++)
            if ((unsigned)y<(unsigned)ny)
                for (int x=cx-ir; x<=cx+ir; x++)
                {
                    real d = sqrtf((real)((y-cy)*(y-cy)+(x-cx)*(x-cx))) - r;
                    int ind = index(x,y);
                    if (d < 0)
                    {
                        fdata[ind].u = v;
                        if (fdata[ind].type == PART_EMPTY)
                            fdata[ind].type = PART_FULL;
                    }
                    if (d < levelset[ind])
                        levelset[ind] = d;
                }
    }
}

//////////////////////////////////////////////////////////////////
//// Set boundary conditions for air/fluid faces

void Grid2D::step_surface(const Grid2D* /*prev*/, Grid2D* /*temp*/, real /*dt*/, real /*diff*/)
{
    // Boundary conditions are not trivial...
    // Carlson Thesis page 24-28
    enum
    {
        FACE_X0=1<<0,
        FACE_X1=1<<1,
        FACE_Y0=1<<2,
        FACE_Y1=1<<3,
    };

    FOR_SURFACE_CELLS(fdata,
    {
        if (fdata[ind].type>0)
        {
            // get face air bitmask
            int mask = 0;
            if (fdata[ind+index(-1,0)].type==0) mask|=FACE_X0;
            if (fdata[ind+index( 1,0)].type==0) mask|=FACE_X1;
            if (fdata[ind+index(0,-1)].type==0) mask|=FACE_Y0;
            if (fdata[ind+index(0, 1)].type==0) mask|=FACE_Y1;
            if (mask!=0)
            {
                real& x0 = fdata[ind].u[0];
                real& y0 = fdata[ind].u[1];
                real& x1 = fdata[ind+index( 1,0)].u[0];
                real& y1 = fdata[ind+index(0, 1)].u[1];
                // Continuity condition: x1-x0+y1-y0 = 0
                switch (mask)
                {
                    // 16 cases...

                    // 1 face: add the other values
                case FACE_X0: x0 = x1   +y1-y0; break;
                case FACE_X1: x1 = x0   +y0-y1; break;
                case FACE_Y0: y0 = x1-x0+y1   ; break;
                case FACE_Y1: y1 = x0-x1+y0   ; break;

                    // 2 faces: weaker condition x1=x0 y1=y0
                    // opposite faces -> do nothing
                case FACE_X0|FACE_X1: break;
                case FACE_Y0|FACE_Y1: break;
                    // other cases: copy opposite value and add half the difference of the remaining two faces
                case FACE_X0|FACE_Y0: x0 = x1;  y0 = y1; break;
                case FACE_X0|FACE_Y1: x0 = x1;  y1 = y0; break;
                case FACE_X1|FACE_Y0: x1 = x0;  y0 = y1; break;
                case FACE_X1|FACE_Y1: x1 = x0;  y1 = y0; break;

                    // 3 faces with one opposite pair: ignore the pair and resolve the remaining one
                case FACE_X0|FACE_Y0|FACE_Y1: x0 = x1   +y1-y0; break;
                case FACE_X1|FACE_Y0|FACE_Y1: x1 = x0   +y0-y1; break;
                case FACE_Y0|FACE_X0|FACE_X1: y0 = x1-x0+y1   ; break;
                case FACE_Y1|FACE_X0|FACE_X1: y1 = x0-x1+y0   ; break;

                    // 4 faces: can't do anything
                case FACE_X0|FACE_X1|FACE_Y0|FACE_Y1: break;

                    // Done!
                }
            }
        }
    });
}

//////////////////////////////////////////////////////////////////
//// Finally advance fluid velocity using Navier-Stockes equations

// This is the real fluid simulation...

void Grid2D::step_advect(const Grid2D* /*prev*/, Grid2D* temp, real dt, real /*diff*/)
{
    // Calculate advection using a semi-lagrangian technique
    // Stam

    // not much optimized for now...

    memset(temp->fdata,0,temp->ncell*sizeof(Cell));

    FOR_INNER_CELLS(temp->fdata,
    {
        // X Axis
        vec2 px( x-0.5f - dt*(fdata[ind].u[0]),
        y      - dt*0.25f*(fdata[ind].u[1]+fdata[ind+index(-1,0)].u[1]+fdata[ind+index(0,1)].u[1]+fdata[ind+index(-1,1)].u[1]));
        temp->fdata[ind].u[0] = interp<0>(px);
        // Y Axis
        vec2 py( x      - dt*0.25f*(fdata[ind].u[0]+fdata[ind+index(0,-1)].u[0]+fdata[ind+index(1,0)].u[0]+fdata[ind+index(1,-1)].u[0]),
        y-0.5f - dt*(fdata[ind].u[1]));
        temp->fdata[ind].u[1] = interp<1>(py);
    });
    // Result is now is temp
}


void Grid2D::step_diffuse(const Grid2D* /*prev*/, Grid2D* temp, real /*dt*/, real diff)
{
    // Calculate diffusion back to here
    // TODO: Check boundary conditions
    if (diff==0.0f)
    {
        for (int ind=0; ind<ncell; ind++)
            fdata[ind].u = temp->fdata[ind].u;
        return;
    }

    real a = diff;
    real inv_c = 1.0f / (1.0001f + 4*a);

    FOR_INNER_CELLS(fdata,
    {
        fdata[ind].u = (temp->fdata[ind].u +
        (temp->fdata[ind+index(-1,0)].u+temp->fdata[ind+index(1,0)].u+
        temp->fdata[ind+index(0,-1)].u+temp->fdata[ind+index(0,1)].u
        )*a)*inv_c;
    });
}


void Grid2D::step_project(const Grid2D* prev, Grid2D* temp, real dt, real /*diff*/)
{
    // Finally calculate projection to divergence free velocity

    // u_new = u~ - dt/P Dp where P = fluid density and p = pressure, Dp = (dp/dx,dp/dy)
    // D.u_new = 0  =>  -dx�D�p = -P/dt dx�D.u~
    //   where  -dx�D�p = 6p(i,j,k)-p(i-1,j,k)-p(i,j-1,k)-p(i,j,k-1)-p(i+1,j,k)-p(i,j+1,k)-p(i,j,k+1)
    //     and  -P/dt dx�D.u~ = -P/dt dx ( u~(i+1,j,k) - u~(i,j,k) + v~(i,j+1,k) - v~(i,j,k) + w~(i,j,k+1) - w~(i,j,k) )
    // Ap = b where A is a diagonal matrix plus neighbour coefficients at -1
    memset(temp->fdata,0,temp->ncell*sizeof(Cell));
    memset(temp->pressure,0,temp->ncell*sizeof(real));

    real* diag = (real*)temp->fdata;
    real* b = diag+ncell;
    real* r = b+ncell;
    real* g = r+ncell;
    real* q = temp->pressure;

    real a = -1.0f/dt;

    real b_norm2 = 0.0;

    //  int nbdiag[7]={0,0,0,0,0,0,0};

    FOR_INNER_CELLS(diag,
    {
        if (fdata[ind].type>0)
        {
            real d = 4; // count air/fluid neighbours
            d -= ((unsigned int)fdata[ind+index(-1,0)].type)>>31;
            d -= ((unsigned int)fdata[ind+index(0,-1)].type)>>31;
            d -= ((unsigned int)fdata[ind+index( 1,0)].type)>>31;
            d -= ((unsigned int)fdata[ind+index(0, 1)].type)>>31;
            //  nbdiag[(int)d]++;
            diag[ind] = d;
        }
    });

    FOR_INNER_CELLS(b,
    {
        if (fdata[ind].type>0)
        {
            real bi = a*(fdata[ind+index(1,0)].u[0]-fdata[ind].u[0] + fdata[ind+index(0,1)].u[1]-fdata[ind].u[1]);
            b[ind] = bi;
            b_norm2 += bi*bi;
        }
    });

    FOR_ALL_CELLS(pressure,
    {
        if (fdata[ind].type>0)
            pressure[ind] = prev->pressure[ind]; // use previous pressure as initial estimate
        else pressure[ind] = 0;
    });

    real err = 0.0;

    // r = b - Ax
    FOR_INNER_CELLS(r,
    {
        if (diag[ind] != 0)
        {
            r[ind] = b[ind] - (diag[ind]*pressure[ind]
            -pressure[ind+index(-1,0)]-pressure[ind+index(0,-1)]
            -pressure[ind+index( 1,0)]-pressure[ind+index(0, 1)]);
        }
    });

    FOR_ALL_CELLS(g,
    {
        g[ind] = r[ind]; // first direction is r
    });

    real min_err = 0.0001f*b_norm2;

    int step;
    for (step=0; step<100; step++)
    {
        real err_old = err;
        err = 0.0f;
        FOR_READ_INNER_CELLS(
        {
            err += r[ind]*r[ind];
        });

        if (err<=min_err) break;
        if (step>0)
        {
            real beta = err/err_old;
            // g = g*beta + r
            FOR_ALL_CELLS(g,
            {
                g[ind] = g[ind]*beta + r[ind];
            });
        }
        real g_q = 0.0;
        // q = Ag
        FOR_INNER_CELLS(q,
        {
            if (diag[ind] != 0)
            {
                real Ag = (diag[ind]*g[ind]
                -g[ind+index(-1,0)]-g[ind+index(0,-1)]
                -g[ind+index( 1,0)]-g[ind+index(0, 1)]);
                q[ind] = Ag;
                g_q += g[ind]*Ag;
            }
        });

        real alpha = err/g_q;

        FOR_ALL_CELLS(pressure,
        {
            pressure[ind] += alpha*g[ind];
            r[ind] -= alpha*q[ind];
        });
    }

    // Now apply pressure back to velocity
    a = dt;

    real max_speed = 0.5f/dt;

    //max_pressure = 0.0f;
    max_pressure = prev->max_pressure;

    FOR_INNER_CELLS(fdata,
    {
        if (fdata[ind].type>=PART_EMPTY)
        {
            if (fdata[ind+index(-1,0)].type>=PART_EMPTY)
            {
                fdata[ind].u[0] -= a*(pressure[ind] - pressure[ind+index(-1,0)]);
                // safety check
                if (fdata[ind].u[0] >  max_speed) fdata[ind].u[0] =  max_speed;
                else if (fdata[ind].u[0] < -max_speed) fdata[ind].u[0] = -max_speed;
            }
            if (fdata[ind+index(0,-1)].type>=PART_EMPTY)
            {
                fdata[ind].u[1] -= a*(pressure[ind] - pressure[ind+index(0,-1)]);
                if (fdata[ind].u[1] >  max_speed) fdata[ind].u[1] =  max_speed;
                else if (fdata[ind].u[1] < -max_speed) fdata[ind].u[1] = -max_speed;
            }
        }
    });
}

} // namespace eulerianfluid

} // namespace behaviormodel

} // namespace component

} // namespace sofa

