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
#ifndef SOFAEULERIANFLUID_FLUID2D_H
#define SOFAEULERIANFLUID_FLUID2D_H

#include <SofaEulerianFluid/config.h>

#include <SofaEulerianFluid/Grid2D.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/objectmodel/Data.h>


namespace sofaeulerianfluid
{
    
class SOFA_EULERIAN_FLUID_API Fluid2D : public sofa::core::BehaviorModel
{
public:
    SOFA_CLASS(Fluid2D, sofa::core::BehaviorModel);

    typedef Grid2D::real real;
    typedef Grid2D::vec2 vec2;
    typedef sofa::type::Vec<3,real> vec3;
protected:

    Grid2D* fluid;
    Grid2D* fnext;
    Grid2D* ftemp;

public:
    sofa::core::objectmodel::Data<int> f_nx; ///< grid size along x axis
    sofa::core::objectmodel::Data<int> f_ny; ///< grid size along y axis
    sofa::core::objectmodel::Data<real> f_cellwidth; ///< width of each cell
    sofa::core::objectmodel::Data<real> f_height; ///< initial fluid height
    sofa::core::objectmodel::Data<vec2> f_dir; ///< initial fluid surface normal
    sofa::core::objectmodel::Data<real> f_tstart; ///< starting time for fluid source
    sofa::core::objectmodel::Data<real> f_tstop; ///< stopping time for fluid source
protected:
    Fluid2D();
    ~Fluid2D() override;
public:
    int getNx() const { return f_nx.getValue(); }
    void setNx(int v) { f_nx.setValue(v);       }

    int getNy() const { return f_ny.getValue(); }
    void setNy(int v) { f_ny.setValue(v);       }

    void init() override;

    void reset() override;

    void doUpdatePosition(SReal dt) override;

    void draw(const sofa::core::visual::VisualParams* vparams) override;

    void computeBBox(const sofa::core::ExecParams* /* params */, bool onlyVisible=false) override;

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

    sofa::type::vector<Vertex> points;
    sofa::type::vector<Face> facets;

    /// For each cube, store the vertex indices on each 3 first edges, and the data value
    struct CubeData
    {
        int p[3];
    };

    // temporary storage for marching cube
    sofa::type::vector<CubeData> planes;
    //typename sofa::type::vector<CubeData>::iterator P0; /// Pointer to first plane
    //typename sofa::type::vector<CubeData>::iterator P1; /// Pointer to second plane

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
		std::size_t nbp = points.size();
        if ((unsigned)p1<(unsigned)nbp &&
            (unsigned)p2<(unsigned)nbp &&
            (unsigned)p3<(unsigned)nbp)
        {
			std::size_t f = facets.size();
            facets.resize(f+1);
            facets[f].p[0] = p1;
            facets[f].p[1] = p2;
            facets[f].p[2] = p3;
            return f;
        }
        else
        {
            msg_error() << "Invalid face "<<p1<<" "<<p2<<" "<<p3;
            return -1;
        }
    }

};

} // namespace sofaeulerianfluid

#endif
