/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
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
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_BEHAVIORMODEL_EULERIANFLUID_FLUID3D_H
#define SOFA_COMPONENT_BEHAVIORMODEL_EULERIANFLUID_FLUID3D_H

#include <sofa/component/behaviormodel/eulerianfluid/Grid3D.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/Field.h>
#include <sofa/core/objectmodel/DataField.h>
#include <sofa/component/mapping/ImplicitSurfaceMapping.h>

namespace sofa
{

namespace component
{

namespace behaviormodel
{

namespace eulerianfluid
{

class Fluid3D : public sofa::core::BehaviorModel, public sofa::core::VisualModel
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
    sofa::core::objectmodel::Field<int> f_nx;
    sofa::core::objectmodel::Field<int> f_ny;
    sofa::core::objectmodel::Field<int> f_nz;
    sofa::core::objectmodel::Field<real> f_cellwidth;
    sofa::core::objectmodel::DataField<vec3> f_center;
    sofa::core::objectmodel::DataField<real> f_height;
    sofa::core::objectmodel::DataField<vec3> f_dir;
    sofa::core::objectmodel::DataField<real> f_tstart;
    sofa::core::objectmodel::DataField<real> f_tstop;

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

    virtual void exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex);

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
            std::cerr << "ERROR: Invalid face "<<p1<<" "<<p2<<" "<<p3<<std::endl;
            return -1;
        }
    }

};

} // namespace eulerianfluid

} // namespace behaviormodel

} // namespace component

} // namespace sofa

#endif
