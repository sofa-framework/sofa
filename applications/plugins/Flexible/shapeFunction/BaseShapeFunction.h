/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifndef SOFA_CORE_BEHAVIOR_BaseShapeFunction_H
#define SOFA_CORE_BEHAVIOR_BaseShapeFunction_H

#include "../initFlexible.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>

namespace sofa
{
namespace core
{
namespace behavior
{

using defaulttype::Vec;
using defaulttype::Mat;
using helper::vector;

/** A shape function $w_i(x)$ encodes the influence of a parent node $x_i$ over a child node $x$.
  It is used to map displacements/velocities/forces, but any other quantities in general.
  Child nodes depend at most on on nbRef parents (use of fixed sizes for efficiency)
  In general, it is a partition of unity : $sum_i w_i(x)=1$
  When $w_i(x_j)=0, i!=j$ and  $w_i(x_i)=1$, shape functions are interpolating. Otherwise they are approximating.
  In first order finite elements, shape functions are barycentric coordinates.
  */
template <class TShapeFunctionTypes>
class BaseShapeFunction : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(BaseShapeFunction, TShapeFunctionTypes) , objectmodel::BaseObject);

    typedef TShapeFunctionTypes ShapeFunctionTypes;
    typedef typename ShapeFunctionTypes::Real Real;
    enum {material_dimensions=ShapeFunctionTypes::material_dimensions};

    /** @name types */
    //@{
    typedef vector<unsigned int> VRef;
    typedef vector<Real> VReal;
    typedef Vec<material_dimensions,Real> Coord;                          ///< Material coordinate: parameters of a point in the object (1 for a wire, 2 for a hull, 3 for a volumetric object)
    typedef Vec<material_dimensions,Real> Gradient;                       ///< gradient of a scalar value in material space
    typedef vector<Gradient> VGradient;
    typedef Mat<material_dimensions,material_dimensions,Real> Hessian;    ///< hessian (second derivative) of a scalar value in material space
    typedef vector<Hessian> VHessian;
    //@}

    /** @name data */
    //@{
    Data<unsigned int > f_nbRef; ///< maximum number of parents per child
    Data<vector<Coord> > f_position;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const BaseShapeFunction<ShapeFunctionTypes>* = NULL) { return ShapeFunctionTypes::Name(); }

    /// compute shape function values (and their first and second derivatives) at a given child position
    /// this is the main function to be reimplemented
    virtual void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)=0;

    /// wrappers
    void computeShapeFunction(const vector<Coord>& childPosition, vector<VRef>& ref, vector<VReal>& w, vector<VGradient>& dw,vector<VHessian>& ddw)
    {
        unsigned int nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);   dw.resize(nb);  ddw.resize(nb);
        for(unsigned i=0; i<nb; i++) computeShapeFunction(childPosition[i],ref[i],w[i],&dw[i],&ddw[i]);
    }

    void computeShapeFunction(const vector<Coord>& childPosition, vector<VRef>& ref, vector<VReal>& w, vector<VGradient>& dw)
    {
        unsigned int nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);   dw.resize(nb);
        for(unsigned i=0; i<nb; i++) computeShapeFunction(childPosition[i],ref[i],w[i],&dw[i]);
    }

    void computeShapeFunction(const vector<Coord>& childPosition, vector<VRef>& ref, vector<VReal>& w)
    {
        unsigned int nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);
        for(unsigned i=0; i<nb; i++) computeShapeFunction(childPosition[i],ref[i],w[i]);
    }

    /// used to make a partition of unity: $sum_i w_i(x)=1$ and adjust derivatives accordingly
    void normalize(VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)
    {
        unsigned int nbRef=w.size();
        Real sum_w=0;
        Gradient sum_dw;
        Hessian sum_ddw;

        // Compute norm
        for (unsigned int j = 0; j < nbRef && w[j]>0.; j++) sum_w += w[j];
        if(dw)
        {
            for (unsigned int j = 0; j < nbRef && w[j]>0.; j++) sum_dw += (*dw)[j];
            if(ddw) for (unsigned int j = 0; j < nbRef && w[j]>0.; j++) sum_ddw += (*ddw)[j];
        }

        // Normalize
        if(sum_w)
            for (unsigned int j = 0; j < nbRef && w[j]>0.; j++)
            {
                Real wn=w[j]/sum_w;
                if(dw)
                {
                    Gradient dwn=((*dw)[j] - sum_dw*wn)/sum_w;
                    if(ddw) for(unsigned int o=0; o<material_dimensions; o++) for(unsigned int p=0; p<material_dimensions; p++) (*ddw)[j][o][p]=((*ddw)[j][o][p] - wn*sum_ddw[o][p] - sum_dw[o]*dwn[p] - sum_dw[p]*dwn[o])/sum_w;
                    (*dw)[j]=dwn;
                }
                w[j]=wn;
            }
    }

    /// wrappers
    void normalize(vector<VReal>& w, vector<VGradient>& dw,vector<VHessian>& ddw)    {        for(unsigned i=0; i<w.size(); i++) normalize(w[i],&dw[i],&ddw[i]);    }
    void normalize(vector<VReal>& w, vector<VGradient>& dw)    {        for(unsigned i=0; i<w.size(); i++) normalize(w[i],&dw[i]);    }
    void normalize(vector<VReal>& w)    {        for(unsigned i=0; i<w.size(); i++) normalize(w[i]);    }

protected:
    BaseShapeFunction()
        : f_nbRef(initData(&f_nbRef,(unsigned int)4,"nbRef", "maximum number of parents per child"))
        , f_position(initData(&f_position,"position", "position of parent nodes"))
    {}

    virtual ~BaseShapeFunction() {}
};


template <int material_dimensions_, class Real_>
struct ShapeFunctionTypes
{
    typedef Real_ Real;
    static const int material_dimensions=material_dimensions_ ;  ///< number of node dimensions (1 for a wire, 2 for a hull, 3 for a volumetric object)
    static const char* Name();
};

typedef ShapeFunctionTypes<3,float> ShapeFunction3f;
typedef ShapeFunctionTypes<3,double> ShapeFunction3d;
template<> inline const char* ShapeFunction3d::Name() { return "ShapeFunction3d"; }
template<> inline const char* ShapeFunction3f::Name() { return "ShapeFunction3f"; }


}
}
}


#endif
