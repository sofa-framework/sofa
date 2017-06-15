/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_CONTAINER_IMPLICITSURFACECONTAINER_H
#define SOFA_COMPONENT_CONTAINER_IMPLICITSURFACECONTAINER_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/Node.h>


namespace sofa
{

namespace component
{

namespace container
{


////////////////// ///////////////


class SOFA_VOLUMETRIC_DATA_API ImplicitSurface : public virtual core::objectmodel::BaseObject
{

public:
    SOFA_CLASS(ImplicitSurface,core::objectmodel::BaseObject);
protected:
    ImplicitSurface( ) { }
    virtual ~ImplicitSurface() { }
	
private:
	ImplicitSurface(const ImplicitSurface& n) ;
	ImplicitSurface& operator=(const ImplicitSurface& n) ;
	
public:
    virtual int getDomain(sofa::defaulttype::Vec3d& /*pos*/, int /*ref_domain*/) {return -1;}

    virtual double getValue(defaulttype::Vec3d& pos)
    { int domain=-1; return getValue(pos,domain);}
    virtual double getValue(defaulttype::Vec3d& pos, int& domain) = 0;  ///< the second parameter is used to identify a domain

    virtual defaulttype::Vec3d getGradient(defaulttype::Vec3d& pos)
    {int domain=-1; return getGradient(pos,domain);}
    virtual defaulttype::Vec3d getGradient(defaulttype::Vec3d& pos, int& i);

    virtual void getValueAndGradient(defaulttype::Vec3d& pos, double &value, defaulttype::Vec3d& grad)
    {
      int domain=-1;
      return getValueAndGradient(pos,value,grad,domain);
    }
    virtual void getValueAndGradient(defaulttype::Vec3d& pos, double &value, defaulttype::Vec3d& grad, int& domain)
    {
      value = getValue(pos,domain);
      grad = getGradient(pos,domain);
    }

    virtual double getDistance(defaulttype::Vec3d& pos)
    {
      int domain=-1;
      return getDistance(pos,domain);
    }
    virtual double getDistance(defaulttype::Vec3d& pos, int& domain)
    {
      defaulttype::Vec3d grad;
      double value;
      getValueAndGradient(pos,value,grad,domain);
      return getDistance(pos,value,grad.norm());
    }

    virtual double getDistance(defaulttype::Vec3d& pos, double value, double grad_norm)
    {
      int domain=-1;
      return getDistance(pos, value, grad_norm, domain);
    }

    virtual double getDistance(defaulttype::Vec3d& /*pos*/, double value, double grad_norm, int &domain)
    { 
      (void)domain;
      // use Taubin's distance by default
      if (grad_norm < 1e-10) return value < 0 ? double(std::numeric_limits<long>::min()) : double(std::numeric_limits<long>::max());
      return value/grad_norm;
    }


    virtual bool computeSegIntersection(defaulttype::Vec3d& posInside, defaulttype::Vec3d& posOutside, defaulttype::Vec3d& intersecPos, int i=-1);
    virtual bool computeSegIntersection(defaulttype::Vec3d& posInside, double valInside, defaulttype::Vec3d& gradInside,
					defaulttype::Vec3d& posOutside, double valOutside, defaulttype::Vec3d& gradOutside,
					defaulttype::Vec3d& intersecPos, int i=-1)
    {
      (void)valInside;
      (void)gradInside;
      (void)valOutside;
      (void)gradOutside;
      return computeSegIntersection(posInside, posOutside, intersecPos, i);
    }
    virtual void projectPointonSurface(defaulttype::Vec3d& point, int i=-1);
    virtual void projectPointonSurface(defaulttype::Vec3d& point, double value, defaulttype::Vec3d& grad, int i=-1)
    {
      (void)value;
      (void)grad;
      projectPointonSurface(point, i);
    }
    virtual bool projectPointonSurface2(defaulttype::Vec3d& point, int i, defaulttype::Vec3d& dir); // TODO mettre les paramètres step=0.1 & countMax=30 en paramètre
    virtual bool projectPointOutOfSurface(defaulttype::Vec3d& point, int i, defaulttype::Vec3d& dir, double &dist_out);


    virtual bool projectPointonSurface2(defaulttype::Vec3d& point, int i=-1)
    {
        defaulttype::Vec3d dir = defaulttype::Vec3d(0,0,0);
        return projectPointonSurface2(point, i, dir);
    }

    virtual bool projectPointOutOfSurface(defaulttype::Vec3d& point, int i=-1)
    {
        defaulttype::Vec3d dir;
        double dist_out = 0.0;
        return projectPointOutOfSurface(point, i, dir, dist_out);
    }


};


class SphereSurface  : public ImplicitSurface
{
public:
    SOFA_CLASS(SphereSurface,ImplicitSurface);

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

  double getValue(defaulttype::Vec3d& Pos, int &domain);
  defaulttype::Vec3d getGradient(defaulttype::Vec3d &Pos, int& domain);
  void getValueAndGradient(defaulttype::Vec3d& Pos, double &value, defaulttype::Vec3d& grad, int& domain);
  virtual double getDistance(defaulttype::Vec3d& pos, int& domain);
  // The following function uses only either value or grad_norm (they are redundant)
  // - value is used is grad_norm < 0
  // - else grad_norm is used: for example, in that case dist = _radius - grad_norm/2 (with _inside=true)
  virtual double getDistance(defaulttype::Vec3d& pos, double value, double grad_norm, int &domain);


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

