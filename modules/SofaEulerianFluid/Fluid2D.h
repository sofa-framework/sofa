/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_BEHAVIORMODEL_EULERIANFLUID_FLUID2D_H
#define SOFA_COMPONENT_BEHAVIORMODEL_EULERIANFLUID_FLUID2D_H

#include <SofaEulerianFluid/Grid2D.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/objectmodel/Data.h>


namespace sofa
{

namespace component
{

namespace behaviormodel
{

namespace eulerianfluid
{

class SOFA_EULERIAN_FLUID_API Fluid2D : public sofa::core::BehaviorModel
{
public:
    SOFA_CLASS(Fluid2D, sofa::core::BehaviorModel);

    typedef Grid2D::real real;
    typedef Grid2D::vec2 vec2;
    typedef sofa::defaulttype::Vec<3,real> vec3;
protected:

    Grid2D* fluid;
    Grid2D* fnext;
    Grid2D* ftemp;

public:
    sofa::core::objectmodel::Data<int> f_nx;
    sofa::core::objectmodel::Data<int> f_ny;
    sofa::core::objectmodel::Data<real> f_cellwidth;
    sofa::core::objectmodel::Data<real> f_height;
    sofa::core::objectmodel::Data<vec2> f_dir;
    sofa::core::objectmodel::Data<real> f_tstart;
    sofa::core::objectmodel::Data<real> f_tstop;
protected:
    Fluid2D();
    virtual ~Fluid2D();
public:
    int getNx() const { return f_nx.getValue(); }
    void setNx(int v) { f_nx.setValue(v);       }

    int getNy() const { return f_ny.getValue(); }
    void setNy(int v) { f_ny.setValue(v);       }

    virtual void init();

    virtual void reset();

    virtual void updatePosition(double dt);

    virtual void draw(const core::visual::VisualParams* vparams);

    virtual void computeBBox(const core::ExecParams* /* params */);

    virtual void updateVisual();

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

    sofa::helper::vector<Vertex> points;
    sofa::helper::vector<Face> facets;

    /// For each cube, store the vertex indices on each 3 first edges, and the data value
    struct CubeData
    {
        int p[3];
    };

    // temporary storage for marching cube
    sofa::helper::vector<CubeData> planes;
    //typename sofa::helper::vector<CubeData>::iterator P0; /// Pointer to first plane
    //typename sofa::helper::vector<CubeData>::iterator P1; /// Pointer to second plane

    template<int C>
    int addPoint(int x,int y,int z, real v0, real v1, real iso)
    {
        int p = points.size();
        vec3 pos((real)x,(real)y,(real)z);
        pos[C] -= (iso-v0)/(v1-v0);
        points.resize(p+1);
        points[p].p = pos; // * cellwidth;
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
            serr << "ERROR: Invalid face "<<p1<<" "<<p2<<" "<<p3<<sendl;
            return -1;
        }
    }

};

} // namespace eulerianfluid

} // namespace behaviormodel

} // namespace component

} // namespace sofa

#endif
