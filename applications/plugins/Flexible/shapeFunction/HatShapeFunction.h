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
#ifndef FLEXIBLE_HatShapeFunction_H
#define FLEXIBLE_HatShapeFunction_H

#include <Flexible/config.h>
#include "BaseShapeFunction.h"
#include <sofa/helper/OptionsGroup.h>
#include <limits>

namespace sofa
{
namespace component
{
namespace shapefunction
{

/**
Compactly supported hat shape function followed by normalization
  */

template<typename TShapeFunctionTypes>
struct HatShapeFunctionInternalData
{
};


template <class ShapeFunctionTypes_>
class HatShapeFunction : public core::behavior::BaseShapeFunction<ShapeFunctionTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HatShapeFunction, ShapeFunctionTypes_) , SOFA_TEMPLATE(core::behavior::BaseShapeFunction, ShapeFunctionTypes_));
    typedef core::behavior::BaseShapeFunction<ShapeFunctionTypes_> Inherit;

    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VCoord VCoord;
    enum {spatial_dimensions=Inherit::spatial_dimensions};
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::VHessian VHessian;
	typedef typename Inherit::VRef VRef;
	typedef typename Inherit::Cell Cell;
    typedef typename Inherit::Hessian Hessian;
    typedef typename Inherit::VecVRef VecVRef;
    typedef typename Inherit::VecVReal VecVReal;
    typedef typename Inherit::VecVGradient VecVGradient;
    typedef typename Inherit::VecVHessian VecVHessian;
    typedef HatShapeFunctionInternalData<ShapeFunctionTypes_> InternalData;

    typedef helper::vector<double> ParamTypes;
    typedef helper::ReadAccessor<Data< ParamTypes > > raParam;

    Data<helper::OptionsGroup> method;
    Data< ParamTypes > param;

    virtual void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell /*cell*/=-1)
    {
        helper::ReadAccessor<Data<VCoord > > parent(this->f_position);
        unsigned int nbp=parent.size(),nbRef=this->f_nbRef.getValue();
        raParam prm(this->param);

        // get the nbRef closest parents
        ref.resize(nbRef); ref.fill(0);
        w.resize(nbRef); w.fill(std::numeric_limits<Real>::max());
        if(dw) dw->resize(nbRef);
        if(ddw) ddw->resize(nbRef);

        for (unsigned int j=0; j<nbp; j++ )
        {
            Coord u=childPosition-parent[j];
            Real d=u.norm();
            unsigned int m=0; while (m!=nbRef && w[m]<d) m++;
            if(m!=nbRef)
            {
                for (unsigned int k=nbRef-1; k>m; k--)
                {
                    w[k]=w[k-1];
                    ref[k]=ref[k-1];
                }
                w[m]=d;
                ref[m]=j;
            }
        }

        // compute weight
        switch(this->method.getValue().getSelectedId())
        {
        case 0:
        {
            Real R=1;    if(prm.size())   R=(Real)prm[0];
            Real p=2;    if(prm.size()>1)   p=(Real)prm[1];
            Real n=3;    if(prm.size()>2)   n=(Real)prm[2];

            for (unsigned int j=0; j<nbRef; j++ )
            {
                // max ( 0, w = (1-(d/R)^p)^n )
                Real d=w[j];
                Real w1= 1. - pow(d/R,p);
                w[j]=pow(w1,n);
                if(w[j]<0) { w[j]=0; if(dw) (*dw)[j].fill(0); if(ddw) (*ddw)[j].clear();}
                else
                {
                    if(dw)
                    {
                        Coord u=childPosition-parent[ref[j]];
                        Real w2 =  p*n*pow(d,p-2)*pow(w1,n-1)/pow(R,p);
                        (*dw)[j] = - u * w2; // dw = - p.n.d^(p-2)(1-(d/R)^p)^(n-1)/R^p (x-x_i)
                        if(ddw)
                        {
                            // ddw = dw.I + p.n.d^(p-4)(1-(d/R)^p)^(n-1)/R^p.( p.(n-1).d^p/(1-(d/R)^p)/R^p - p + 2)  .(x-x_i).(x-x_i)^T
                            Real w4= w2/(d*d) * ( p - 2 - p*(n-1)*pow(d,2*p)/(w1*pow(R,p)) );
                            for(int k=0; k<Hessian::nbLines; k++) (*ddw)[j](k,k)= - w2;
                            for(int k=0; k<Hessian::nbLines; k++) for(int m=0; m<Hessian::nbCols; m++) (*ddw)[j](k,m)+=u[k]*u[m]*w4;
                        }
                    }
                }
            }
        }
            break;

        default:
            break;
        }

        // normalize
        this->normalize(w,dw,ddw);
    }

protected:
    HatShapeFunction()
        :Inherit()
        , method ( initData ( &method,"method","method" ) )
        , param ( initData ( &param,"param","param" ) )

    {
        helper::OptionsGroup methodo(1	,"0 - max[0,(1-(dist/R)^p)^n], params=(R,p=2,n=3)" );
        methodo.setSelectedItem(0);
        method.setValue(methodo);
    }

    virtual ~HatShapeFunction()
    {

    }
};


}
}
}


#endif
