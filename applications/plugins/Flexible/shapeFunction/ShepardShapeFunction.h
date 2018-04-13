/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef FLEXIBLE_ShepardShapeFunction_H
#define FLEXIBLE_ShepardShapeFunction_H

#include <Flexible/config.h>
#include "BaseShapeFunction.h"
#include <limits>

namespace sofa
{
namespace component
{
namespace shapefunction
{

/**
Shepard shape function (=inverse distance weights) is defined as w_i(x)=1/d(x,x_i)^power followed by normalization
http://en.wikipedia.org/wiki/Inverse_distance_weighting
  */

template<typename TShapeFunctionTypes>
struct ShepardShapeFunctionInternalData
{
};


template <class ShapeFunctionTypes_>
class ShepardShapeFunction : public core::behavior::BaseShapeFunction<ShapeFunctionTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ShepardShapeFunction, ShapeFunctionTypes_) , SOFA_TEMPLATE(core::behavior::BaseShapeFunction, ShapeFunctionTypes_));
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
    typedef ShepardShapeFunctionInternalData<ShapeFunctionTypes_> InternalData;

    Data<Real> power; ///< power of the inverse distance

    virtual void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell /*cell*/=-1)
    {
		helper::ReadAccessor<Data<VCoord > > parent(this->f_position);
        unsigned int nbp=parent.size(),nbRef=this->f_nbRef.getValue();
		Real pw=this->power.getValue();

        // get the nbRef closest parents
        ref.resize(nbRef); ref.fill(0);
        w.resize(nbRef); w.fill(0);
        if(dw) dw->resize(nbRef);
        if(ddw) ddw->resize(nbRef);

        for (unsigned int j=0; j<nbp; j++ )
        {
            Coord u=childPosition-parent[j];
            Real W=pow(u.norm(),pw);
            if(W!=0) W=1./W; else W=std::numeric_limits<Real>::max()/2.; // divide by two to avoid out of bound problems during normalization
            unsigned int m=0; while (m!=nbRef && w[m]>W) m++;
            if(m!=nbRef)
            {
                for (unsigned int k=nbRef-1; k>m; k--)
                {
                    w[k]=w[k-1];
                    ref[k]=ref[k-1];
                }
                w[m]=W;
                ref[m]=j;
            }
        }

        // compute weight gradients
        if(dw)
            for (unsigned int j=0; j<nbRef; j++ )
            {
                (*dw)[j].fill(0);
                if(ddw) (*ddw)[j].clear();
                if (w[j])
                {
                    Coord u=childPosition-parent[ref[j]];
					Real u2=u.norm2();
                    Real w2=(u2) ? (pw * w[j] / u2) : 0.;
                    (*dw)[j] = - u * w2; // dw = - pw.(x-x_i)/d(x,x_i)^(power+2)
                    if(ddw)
                    {
                        // ddw = - pw.I/d(x,x_i)^(power+2) + pw.(pw+2).(x-x_i).(x-x_i)^T/d(x,x_i)^(power+4)
                        Real u4=u2*u2;
                        Real w4= (u4) ? (pw * (pw+2.) * w[j] / u4) : 0.;
                        for(int k=0; k<Hessian::nbLines; k++) (*ddw)[j](k,k)= - w2;
                        for(int k=0; k<Hessian::nbLines; k++) for(int m=0; m<Hessian::nbCols; m++) (*ddw)[j](k,m)+=u[k]*u[m]*w4;
                    }
                }
            }


        // normalize
		this->normalize(w,dw,ddw);
    }

protected:
    ShepardShapeFunction()
        :Inherit()
        , power(initData(&power,(Real)2.0, "power", "power of the inverse distance"))

    {
    }

    virtual ~ShepardShapeFunction()
    {

    }
};


}
}
}


#endif
