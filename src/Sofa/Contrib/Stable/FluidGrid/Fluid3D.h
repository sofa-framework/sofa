#ifndef SOFA_CONTRIB_FLUIDGRID3D_FLUID3D_H
#define SOFA_CONTRIB_FLUIDGRID3D_FLUID3D_H

#include "Grid3D.h"
#include <Sofa/Abstract/BehaviorModel.h>
#include <Sofa/Abstract/VisualModel.h>
#include <Sofa/Components/Common/Field.h>
#include <Sofa/Components/Common/DataField.h>
#include <Sofa/Components/ImplicitSurfaceMapping.h>

namespace Sofa
{

namespace Contrib
{

namespace FluidGrid
{

class Fluid3D : public Sofa::Abstract::BehaviorModel, public Sofa::Abstract::VisualModel
{
public:
    typedef Grid3D::real real;
    typedef Grid3D::vec3 vec3;

protected:
    int nx,ny,nz;
    real cellwidth;

    Grid3D* fluid;
    Grid3D* fnext;
    Grid3D* ftemp;

public:
    Sofa::Components::Common::Field<int> f_nx;
    Sofa::Components::Common::Field<int> f_ny;
    Sofa::Components::Common::Field<int> f_nz;
    Sofa::Components::Common::Field<real> f_cellwidth;
    Sofa::Components::Common::DataField<real> f_height;
    Sofa::Components::Common::DataField<vec3> f_dir;
    Sofa::Components::Common::DataField<real> f_tstart;
    Sofa::Components::Common::DataField<real> f_tstop;

    Fluid3D();
    virtual ~Fluid3D();

    int getNx() const { return f_nx.getValue(); }
    void setNx(int v) { f_nx.setValue(v);       }

    int getNy() const { return f_ny.getValue(); }
    void setNy(int v) { f_ny.setValue(v);       }

    int getNz() const { return f_nz.getValue(); }
    void setNz(int v) { f_nz.setValue(v);       }

    virtual void init();

    virtual void reset();

    virtual void updatePosition(double dt);

    virtual void draw();

    virtual void initTextures() {}

    virtual void update();

    virtual bool addBBox(double* minBBox, double* maxBBox);

protected:
    // marching cube

    struct Vertex
    {
        vec3 p;
        vec3 n;
    };

    struct Face
    {
        int p[3];
    };

    std::vector<Vertex> points;
    std::vector<Face> facets;

    /// For each cube, store the vertex indices on each 3 first edges, and the data value
    struct CubeData
    {
        int p[3];
    };

    // temporary storage for marching cube
    std::vector<CubeData> planes;
    //typename std::vector<CubeData>::iterator P0; /// Pointer to first plane
    //typename std::vector<CubeData>::iterator P1; /// Pointer to second plane

    template<int C>
    int addPoint(int x,int y,int z, real v0, real v1, real iso)
    {
        int p = points.size();
        vec3 pos = vec3((real)x,(real)y,(real)z);
        pos[C] -= (iso-v0)/(v1-v0);
        points.resize(p+1);
        points[p].p = pos * cellwidth;
        return p;
    }

    int addFace(int p1, int p2, int p3)
    {
        int nbp = points.size();
        if ((unsigned)p1<(unsigned)nbp &&
            (unsigned)p2<(unsigned)nbp &&
            (unsigned)p3<(unsigned)nbp)
        {
            int f = facets.size();
            facets.resize(f+1);
            facets[f].p[0] = p1;
            facets[f].p[1] = p2;
            facets[f].p[2] = p3;
            return f;
        }
        else
        {
            std::cerr << "ERROR: Invalid face "<<p1<<" "<<p2<<" "<<p3<<std::endl;
            return -1;
        }
    }

};

} // namespace FluidGrid

} // namespace Contrib

} // namespace Sofa

#endif
