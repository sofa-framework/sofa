/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONTAINER_IMPLICITSURFACECONTAINER_H
#define SOFA_COMPONENT_CONTAINER_IMPLICITSURFACECONTAINER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/tree/GNode.h>


namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;
using namespace sofa::simulation::tree;

////////////////// ///////////////


class ImplicitSurface : public virtual core::objectmodel::BaseObject
{

public:
    ImplicitSurface( ) { }
    virtual ~ImplicitSurface() { }
    virtual double getValue(defaulttype::Vec3d& pos) =0;
    virtual double getValue(defaulttype::Vec3d& pos, int& /*domain*/) { return getValue(pos); };  // the second parameter could be useful to identify a domain
    virtual defaulttype::Vec3d getGradient(defaulttype::Vec3d&, int domain=0);

    virtual bool computeSegIntersection(defaulttype::Vec3d& posInside, defaulttype::Vec3d& posOutside, defaulttype::Vec3d& intersecPos, int domain=0);
    virtual void projectPointonSurface(defaulttype::Vec3d& point, int domain=0);
    virtual bool projectPointonSurface2(defaulttype::Vec3d& point, int domain, defaulttype::Vec3d& dir); // TODO mettre les paramètres step=0.1 & countMax=30 en paramètre
    bool projectPointonSurface2(defaulttype::Vec3d& point, int i=0)
    {
        defaulttype::Vec3d dir = defaulttype::Vec3d(0,0,0);
        return projectPointonSurface2(point, i, dir);
    }
    virtual bool projectPointOutOfSurface(defaulttype::Vec3d& point, int domain, defaulttype::Vec3d& dir, double &dist_out);
    bool projectPointOutOfSurface(defaulttype::Vec3d& point, int domain=0)
    {
        defaulttype::Vec3d dir;
        double dist_out = 0.0;
        return projectPointOutOfSurface(point, domain, dir, dist_out);
    }
    virtual int getDomain(sofa::defaulttype::Vec3d &/*pos*/) { return 0; }
};


class SphereSurface  : public ImplicitSurface
{
public:
    SphereSurface()
        : inside(initData(&inside, false, "inside", "if true the constraint object is inside the sphere"))
        , radiusSphere(initData(&radiusSphere, 1.0, "radius", "Radius of the Sphere Surface"))
        , centerSphere(initData(&centerSphere, defaulttype::Vec3d(0.0,0.0,0.0), "center", "Position of the Sphere Surface"))
    {init();}

    ~SphereSurface() { }

    void init()
    {
        _inside = inside.getValue();
        _Center = centerSphere.getValue();
        _radius = radiusSphere.getValue();
    }

    void reinit() {init();}

    double getValue(defaulttype::Vec3d& Pos);
    //inline	double getValue(defaulttype::Vec3d& Pos, int&){return getValue(Pos);}
    //defaulttype::Vec3d getGradient(defaulttype::Vec3d &Pos);

    Data<bool> inside;
    Data<double> radiusSphere;
    Data<defaulttype::Vec3d> centerSphere;


private:

    defaulttype::Vec3d _Center;
    double _radius;
    bool _inside;


};
/////////////////////////////////////////////////////////


} // namespace container

} // namespace component

} // namespace sofa

#endif

