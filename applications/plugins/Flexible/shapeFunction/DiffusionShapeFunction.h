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
#ifndef FLEXIBLE_DiffusionShapeFunction_H
#define FLEXIBLE_DiffusionShapeFunction_H

#include "../initFlexible.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include "../shapeFunction/BaseImageShapeFunction.h"
#include "../types/PolynomialBasis.h"

#include <image/ImageTypes.h>
#include <image/ImageAlgorithms.h>

#include <sofa/helper/OptionsGroup.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

#define ISOTROPIC 0
#define ANISOTROPIC 1

namespace sofa
{
namespace component
{
namespace shapefunction
{

using core::behavior::BaseShapeFunction;
using defaulttype::Mat;
using defaulttype::Vec;

/**
Shape functions computed using heat diffusion in images
  */

template <class ShapeFunctionTypes_,class ImageTypes_>
class DiffusionShapeFunction : public BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DiffusionShapeFunction, ShapeFunctionTypes_,ImageTypes_) , SOFA_TEMPLATE2(BaseImageShapeFunction, ShapeFunctionTypes_,ImageTypes_));
    typedef BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_> Inherit;

    /** @name  Shape function types */
    //@{
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    //@}

    /** @name  Image data */
    //@{
    typedef ImageTypes_ ImageTypes;
    typedef typename Inherit::T T;
    typedef typename Inherit::imCoord imCoord;
    typedef typename Inherit::raImage raImage;

    typedef typename Inherit::raTransform raTransform;

    typedef typename Inherit::DistT DistT;
    typedef typename Inherit::DistTypes DistTypes;
    typedef typename Inherit::waDist waDist;

    typedef typename Inherit::IndT IndT;
    typedef typename Inherit::IndTypes IndTypes;
    typedef typename Inherit::waInd waInd;
    //@}

    /** @name  Options */
    //@{
    Data<helper::OptionsGroup> method;
    Data<bool> biasDistances;
    Data<bool> useDijkstra;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const DiffusionShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    virtual void init()
    {
        Inherit::init();

//        helper::ReadAccessor<Data<vector<Coord> > > parent(this->f_position);
//        if(!parent.size()) { serr<<"Parent nodes not found"<<sendl; return; }

//        // get tranform and image at time t
//        raImage in(this->image);
//        raTransform inT(this->transform);
//        if(!in->getCImgList().size())  { serr<<"Image not found"<<sendl; return; }
//        const CImg<T>& inimg = in->getCImg(0);  // suppose time=0
//        const CImg<T>* biasFactor=biasDistances.getValue()?&inimg:NULL;
//        const Vec<3,Real>& voxelsize=this->transform.getValue().getScale();

//        // init voronoi and distances
//        imCoord dim = in->getDimensions(); dim[3]=dim[4]=1;
//        waInd vorData(this->f_voronoi); vorData->setDimensions(dim);
//        CImg<IndT>& voronoi = vorData->getCImg(); voronoi.fill(0);

//        waDist distData(this->f_distances);         distData->setDimensions(dim);
//        CImg<DistT>& dist = distData->getCImg(); dist.fill(-1);
//        cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) dist(x,y,z)=cimg::type<DistT>::max();
//        }


    }

protected:
    DiffusionShapeFunction()
        :Inherit()
        , method ( initData ( &method,"method","method (param)" ) )
        , biasDistances(initData(&biasDistances,false,"bias","Bias distances using inverse pixel values"))
        , useDijkstra(initData(&useDijkstra,true,"useDijkstra","Use Dijkstra for geodesic distance computation (use fastmarching otherwise)"))
    {
        helper::OptionsGroup methodOptions(3,"0 - Isotropic"
                ,"1 - Anisotropic"
                                          );
        methodOptions.setSelectedItem(ISOTROPIC);
        method.setValue(methodOptions);

        method.setGroup("parameters");
        biasDistances.setGroup("parameters");
        useDijkstra.setGroup("parameters");
    }

    virtual ~DiffusionShapeFunction()
    {

    }
};


}
}
}


#endif

