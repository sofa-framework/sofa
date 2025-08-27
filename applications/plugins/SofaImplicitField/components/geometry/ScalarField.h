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
/******************************************************************************
*  Contributors:
*  - damien.marchal@univ-lille.fr
*  - olivier.goury@inria.fr
******************************************************************************/

#ifndef SOFAIMPLICITFIELD_COMPONENT_SCALARFIELD_H
#define SOFAIMPLICITFIELD_COMPONENT_SCALARFIELD_H
#include <SofaImplicitField/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Mat.h>

namespace sofa::component::geometry
{

namespace _scalarfield_
{

using sofa::core::objectmodel::BaseObject ;
using sofa::type::Vec3d ;
using sofa::type::Mat3x3 ;

////////////////// ///////////////
class SOFA_SOFAIMPLICITFIELD_API ScalarField : public BaseObject
{
public:
    SOFA_CLASS(ScalarField, BaseObject);

public:
    void init() override;

    /// Compute the gradient using a first order finite-difference scheme.
    /// This is of lower precision compared to analytical gradient computed by derivating
    /// the equations.
    Vec3d getGradientByFinitDifference(Vec3d& pos, int& domain) ;
    void getHessianByCentralFiniteDifference(const Vec3d& x, const double dx,
                                             Mat3x3& hessian);
    virtual int getDomain(Vec3d& pos, int domain) {
        SOFA_UNUSED(pos);
        SOFA_UNUSED(domain);
        return -1;
    }

    virtual double getValue(Vec3d& pos, int& domain) = 0;
    inline double getValue(Vec3d& pos) { int domain=-1; return getValue(pos,domain); }

    /// By default compute the gradient using a first order finite difference approache
    /// If you have analytical derivative don't hesitate to override this function.
    virtual Vec3d getGradient(Vec3d& pos, int& domain);
    inline Vec3d getGradient(Vec3d& pos) {int domain=-1; return getGradient(pos,domain); }
    virtual void getHessian(Vec3d &Pos, Mat3x3& h);

    /// Returns the value and the gradiant by evaluating one after an other.
    /// For some computation it is possible to implement more efficiently the computation
    /// By factoring the computing of the two...if you can do this please override this function.
    virtual void getValueAndGradient(Vec3d& pos, double &value, Vec3d& grad, int& domain) ;
    inline void getValueAndGradient(Vec3d& pos, double &value, Vec3d& grad)
    {
      int domain=-1;
      return getValueAndGradient(pos,value,grad,domain);
    }

    virtual bool computeSegIntersection(Vec3d& posInside, Vec3d& posOutside, Vec3d& intersecPos, int domain=-1);
    bool computeSegIntersection(Vec3d& posInside, double valInside, Vec3d& gradInside,
                    Vec3d& posOutside, double valOutside, Vec3d& gradOutside,
                    Vec3d& intersecPos, int domain=-1)
    {
      (void)valInside;
      (void)gradInside;
      (void)valOutside;
      (void)gradOutside;
      return computeSegIntersection(posInside, posOutside, intersecPos, domain);
    }

    virtual void projectPointonSurface(Vec3d& point, int i=-1);
    void projectPointonSurface(Vec3d& point, double value, Vec3d& grad, int domain=-1)
    {
      (void)value;
      (void)grad;
      projectPointonSurface(point, domain);
    }

    // TODO mettre les paramètres step=0.1 & countMax=30 en paramètre
    virtual bool projectPointonSurface2(Vec3d& point, int i, Vec3d& dir);
    bool projectPointonSurface2(Vec3d& point, int domain=-1)
    {
        Vec3d dir = Vec3d(0,0,0);
        return projectPointonSurface2(point, domain, dir);
    }

    virtual bool projectPointOutOfSurface(Vec3d& point, int i, Vec3d& dir, double &dist_out);
    bool projectPointOutOfSurface(Vec3d& point, int domain=-1)
    {
        Vec3d dir;
        double dist_out = 0.0;
        return projectPointOutOfSurface(point, domain, dir, dist_out);
    }


protected:
    Data< double > d_epsilon; ///< Tolerance when evaluating the gradient and/or the hessian of the implicit surface numerically
    ScalarField( )
        : d_epsilon(initData(&d_epsilon,0.00001,"epsilon","Tolerance when evaluating the gradient and/or the hessian of the implicit surface numerically"))
    {
    }
    ~ScalarField() override { }

private:
    ScalarField(const ScalarField& n) ;
    ScalarField& operator=(const ScalarField& n) ;
};


} /// namespace _scalarfield_

using _scalarfield_::ScalarField ;

} /// namespace sofa::component::geometry

#endif

