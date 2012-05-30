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
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/MatSym.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/core/behavior/BaseMechanicalState.h>


namespace sofa
{
namespace core
{
namespace behavior
{

using defaulttype::StdVectorTypes;
using defaulttype::Vec;
using defaulttype::Mat;
using defaulttype::MatSym;
using helper::vector;

/** Compute interpolation weights and their derivatives.
  Interpolation is defined across a material space as \f$ x_j = \sum_i w_{ij} x_i \f$, where the x are material coordinates (3 dimensions for a volumetris solid, 2 for a surface, 1 for a line, independently of the dimension of the space they are moving in).
  Shape function \f$ w_{ij}(x_j) \f$ encodes the influence of a parent node at \f$ x_i \f$ an a child node at \f$ x_j \f$.
  It is used to map displacements/velocities/forces, but any other quantities in general.
  For efficiency, child nodes depend at most on nbRef parents.
  In general, it is a partition of unity : \f$ sum_i w_{ij}(x)=1 \f$ everywhere.
  When \f$ w_i(x_j)=0, i!=j \f$ and  \f$ w_i(x_i)=1 \f$, shape functions are called interpolating. Otherwise they are called approximating.
  In first order finite elements, the shape functions are the barycentric coordinates.
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
    typedef vector<Coord> VCoord;
    typedef Vec<material_dimensions,Real> Gradient;                       ///< Gradient of a scalar value in material space
    typedef vector<Gradient> VGradient;
    typedef Mat<material_dimensions,material_dimensions,Real> Hessian;    ///< Hessian (second derivative) of a scalar value in material space
    typedef vector<Hessian> VHessian;
    //@}

    /** @name data */
    //@{
    Data<unsigned int > f_nbRef;      ///< maximum number of parents per child
    Data< VCoord > f_position;  ///< material coordinates of the parent nodes
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const BaseShapeFunction<ShapeFunctionTypes>* = NULL) { return ShapeFunctionTypes::Name(); }


    virtual void init()
    {
        if(!f_position.isSet())
            // material positions are not given, so we compute them based on the current spatial positions
        {
            BaseMechanicalState* state = NULL;
            this->getContext()->get(state,core::objectmodel::BaseContext::Local);
            if(!state) { serr<<"state not found"<< sendl; return; }
            else
            {
                helper::WriteAccessor<Data<vector<Coord> > > pos(this->f_position);
                pos.resize(state->getSize());
                for(unsigned int i=0; i<pos.size(); ++i)
                {
                    StdVectorTypes<Coord,Coord>::set( pos[i], state->getPX(i),state->getPY(i),state->getPZ(i) );
//                    pos[i]=Coord(state->getPX(i),state->getPY(i),state->getPZ(i));
//                std::cout<<"pts: "<<state->getPX(0)<<", "<<state->getPY(0)<<", "<<state->getPZ(0);
                }
            }
        }

    }

    /// compute shape function values (and their first and second derivatives) at a given child position
    /// this is the main function to be reimplemented
    virtual void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)=0;

    /// wrappers
    void computeShapeFunction(const vector<Coord>& childPosition, vector<VRef>& ref, vector<VReal>& w, vector<VGradient>& dw,vector<VHessian>& ddw)
    {
        unsigned int nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);   dw.resize(nb);  ddw.resize(nb);
        for(unsigned i=0; i<nb; i++)            computeShapeFunction(childPosition[i],ref[i],w[i],&dw[i],&ddw[i]);
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
        for (unsigned int j = 0; j < nbRef; j++) sum_w += w[j];
        if(dw)
        {
            for (unsigned int j = 0; j < nbRef; j++) sum_dw += (*dw)[j];
            if(ddw) for (unsigned int j = 0; j < nbRef; j++) sum_ddw += (*ddw)[j];
        }

        // Normalize
        if(sum_w)
            for (unsigned int j = 0; j < nbRef; j++)
            {
                Real wn=w[j]/sum_w;
                if(dw)
                {
                    Gradient dwn=((*dw)[j] - sum_dw*wn)/sum_w;
                    if(ddw) for(unsigned int o=0; o<material_dimensions; o++) for(unsigned int p=0; p<material_dimensions; p++) (*ddw)[j](o,p)=((*ddw)[j](o,p) - wn*sum_ddw(o,p) - sum_dw[o]*dwn[p] - sum_dw[p]*dwn[o])/sum_w;
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
    {
    }

    virtual ~BaseShapeFunction() {}
};


template <int material_dimensions_, class Real_>
struct ShapeFunctionTypes
{
    typedef Real_ Real;
    static const int material_dimensions=material_dimensions_ ;  ///< number of node dimensions (1 for a wire, 2 for a hull, 3 for a volumetric object)
    static const char* Name();
};

typedef ShapeFunctionTypes<2,float>  ShapeFunction2f;
typedef ShapeFunctionTypes<2,double> ShapeFunction2d;
typedef ShapeFunctionTypes<3,float>  ShapeFunction3f;
typedef ShapeFunctionTypes<3,double> ShapeFunction3d;

#ifdef SOFA_FLOAT
typedef ShapeFunction2f ShapeFunction2;
typedef ShapeFunction3f ShapeFunction3;
#else
typedef ShapeFunction2d ShapeFunction2;
typedef ShapeFunction3d ShapeFunction3;
#endif

template<> inline const char* ShapeFunction2d::Name() { return "ShapeFunction2d"; }
template<> inline const char* ShapeFunction2f::Name() { return "ShapeFunction2f"; }
template<> inline const char* ShapeFunction3d::Name() { return "ShapeFunction3d"; }
template<> inline const char* ShapeFunction3f::Name() { return "ShapeFunction3f"; }


}
}
}


#endif
