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
#ifndef SOFA_IMAGE_IMAGEFILTER_H
#define SOFA_IMAGE_IMAGEFILTER_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/SVector.h>

#include <sofa/component/component.h>

#define NONE 0
#define BLURDERICHE 1
#define BLURMEDIAN 2
#define BLURBILATERAL 3
#define BLURANISOTROPIC 4
#define DERICHE 5
#define CROP 6
#define RESIZE 7
#define TRIM 8
#define DILATE 9
#define ERODE 10
#define NOISE 11
#define QUANTIZE 12
#define THRESHOLD 13
#define LAPLACIAN 14
#define STENSOR 15
#define DISTANCE 16


namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;
using namespace helper;

/**
 * This class computes a filtered image
 */


template <class _ImageTypes>
class ImageFilter : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageFilter,_ImageTypes),Inherited);

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;

    typedef SReal Real;
    typedef ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    typedef vector<double> ParamTypes;
    typedef helper::ReadAccessor<Data< ParamTypes > > raParam;

    Data<OptionsGroup> filter;
    Data< ParamTypes > param;

    Data< ImageTypes > inputImage;
    Data< TransformType > inputTransform;

    Data< ImageTypes > outputImage;
    Data< TransformType > outputTransform;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageFilter<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    ImageFilter()    :   Inherited()
        , filter ( initData ( &filter,"filter","Filter" ) )
        , param ( initData ( &param,"param","Parameters" ) )
        , inputImage(initData(&inputImage,ImageTypes(),"inputImage",""))
        , inputTransform(initData(&inputTransform,TransformType(),"inputTransform",""))
        , outputImage(initData(&outputImage,ImageTypes(),"outputImage",""))
        , outputTransform(initData(&outputTransform,TransformType(),"outputTransform",""))
    {
        inputImage.setReadOnly(true);
        inputTransform.setReadOnly(true);
        outputImage.setReadOnly(true);
        outputTransform.setReadOnly(true);
        helper::OptionsGroup filterOptions(17	,"0 - None"
                ,"1 - Blur ( sigma )"
                ,"2 - Blur Median ( n )"
                ,"4 - Blur Bilateral ( sigma_s, sigma_r)"
                ,"5 - Blur Anisotropic ( amplitude )"
                ,"3 - Deriche ( sigma , order , axis )"
                ,"6 - Crop ( xmin , ymin , zmin , xmax , ymax , zmax)"
                ,"7 - Resize( dimx , dimy , dimz , no interp.|nearest neighb.|moving av.|linear|grid|bicubic|lanzcos)"
                ,"8 - Trim ( tmin , tmax )"
                ,"9 - Dilate ( size )"
                ,"10 - Erode ( size )"
                ,"11 - Noise ( sigma , gaussian|uniform|Salt&Pepper|Poisson|Rician )"
                ,"12 - Quantize ( nbLevels )"
                ,"13 - Threshold ( value )"
                ,"14 - Laplacian"
                ,"15 - Structure tensors ( scheme )"
                ,"16 - Distance ( value )" //, const float sizex=1, const float sizey=1, const float sizez=1
                                          );
        filterOptions.setSelectedItem(NONE);
        filter.setValue(filterOptions);
    }

    virtual ~ImageFilter() {}

    virtual void init()
    {
        addInput(&inputImage);
        addInput(&inputTransform);
        addOutput(&outputImage);
        addOutput(&outputTransform);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        bool updateImage = this->inputImage.isDirty();	// change of input image -> update output image
        bool updateTransform = this->inputTransform.isDirty();	// change of input transform -> update output transform
        if(!updateImage && !updateTransform) {updateImage=true; updateTransform=true;}  // change of parameters -> update all

        cleanDirty();
        raParam p(this->param);
        raImage in(this->inputImage);
        raTransform inT(this->inputTransform);
        waImage out(this->outputImage);
        waTransform outT(this->outputTransform);

        CImgList<T>& img = out->getCImgList();
        if(updateImage) img.assign(in->getCImgList());	// copy
        if(updateTransform) outT->operator=(inT);	// copy

        switch(this->filter.getValue().getSelectedId())
        {
        case BLURDERICHE:
            if(updateImage)
            {
                float sigma=0; if(p.size()) sigma=(float)p[0];
                cimglist_for(img,l) img(l).blur(sigma);
            }
            break;
        case BLURMEDIAN:
            if(updateImage)
            {
                unsigned int n=0; if(p.size()) n=(unsigned int)p[0];
                cimglist_for(img,l) img(l).blur_median (n);
            }
            break;
        case BLURBILATERAL:
            if(updateImage)
            {
                float sigma_s=0;  if(p.size()) sigma_s=(float)p[0];
                float sigma_r=0; if(p.size()>1) sigma_r=(float)p[1];
                cimglist_for(img,l) img(l).blur_bilateral (sigma_s,sigma_r);
            }
            break;
        case BLURANISOTROPIC:
            if(updateImage)
            {
                float amplitude=0; if(p.size()) amplitude=(float)p[0];
                cimglist_for(img,l) img(l).blur_anisotropic (amplitude);
            }
            break;
        case DERICHE:
            if(updateImage)
            {
                float sigma=0;  if(p.size()) sigma=(float)p[0];
                unsigned int order=0; if(p.size()>1) order=(unsigned int)p[1];
                char axis='x';  if(p.size()>2) { if((int)p[2]==1) axis='y'; else if((int)p[2]==2) axis='z'; }
                cimglist_for(img,l) img(l).deriche (sigma,order,axis);
            }
            break;
        case CROP:
            if(updateImage || updateTransform)
            {
                unsigned int xmin=0; if(p.size())   xmin=(unsigned int)p[0];
                unsigned int ymin=0; if(p.size()>1) ymin=(unsigned int)p[1];
                unsigned int zmin=0; if(p.size()>2) zmin=(unsigned int)p[2];
                unsigned int xmax=in->getDimensions()[0]-1; if(p.size()>3) xmax=(unsigned int)p[3];
                unsigned int ymax=in->getDimensions()[1]-1; if(p.size()>4) ymax=(unsigned int)p[4];
                unsigned int zmax=in->getDimensions()[2]-1; if(p.size()>5) zmax=(unsigned int)p[5];
                if(updateImage) cimglist_for(img,l) img(l).crop(xmin,ymin,zmin,0,xmax,ymax,zmax,in->getDimensions()[3]-1);
                if(updateTransform)
                {
                    outT->getTranslation()=outT->fromImage( Coord((Real)xmin,(Real)ymin,(Real)zmin) );
                    outT->setCamPos((Real)out->getDimensions()[0]/2.0,(Real)out->getDimensions()[1]/2.0);
                }
            }
            break;
        case RESIZE:
            if(updateImage || updateTransform)
            {
                unsigned int dimx=in->getDimensions()[0]; if(p.size())   dimx=(unsigned int)p[0];
                unsigned int dimy=in->getDimensions()[1]; if(p.size()>1) dimy=(unsigned int)p[1];
                unsigned int dimz=in->getDimensions()[2]; if(p.size()>2) dimz=(unsigned int)p[2];
                unsigned int interpolation=1; if(p.size()>3) interpolation=(unsigned int)p[3];
                if(updateImage) cimglist_for(img,l) img(l).resize(dimx,dimy,dimz,-100,interpolation);
                if(updateTransform)
                    if(interpolation)
                    {
                        if(dimx!=1) outT->getScale()[0]*=((Real)in->getDimensions()[0]-1.0)/((Real)dimx-1.0);
                        if(dimy!=1) outT->getScale()[1]*=((Real)in->getDimensions()[1]-1.0)/((Real)dimy-1.0);
                        if(dimz!=1) outT->getScale()[2]*=((Real)in->getDimensions()[2]-1.0)/((Real)dimz-1.0);
                        outT->setCamPos((Real)out->getDimensions()[0]/2.0,(Real)out->getDimensions()[1]/2.0);
                    }
            }

            break;
        case TRIM:
            if(updateImage || updateTransform)
            {
                unsigned int tmin=0; if(p.size())   tmin=(unsigned int)p[0];
                unsigned int tmax=in->getDimensions()[4]-1; if(p.size()>1) tmax=(unsigned int)p[1];
                if(updateImage) if(tmax<in->getDimensions()[4]-1) img.remove(tmax+1,in->getDimensions()[4]-1);
                if(updateImage) if(tmin>0) img.remove(0,tmin-1);
                if(updateTransform) outT->getOffsetT()=outT->fromImage((Real)tmin);
            }
            break;
        case DILATE:
            if(updateImage)
            {
                unsigned int size=0; if(p.size()) size=(unsigned int)p[0];
                cimglist_for(img,l) img(l).dilate (size);
            }
            break;
        case ERODE:
            if(updateImage)
            {
                unsigned int size=0; if(p.size()) size=(unsigned int)p[0];
                cimglist_for(img,l) img(l).erode (size);
            }
            break;
        case NOISE:
            if(updateImage)
            {
                float sigma=0;  if(p.size()) sigma=(float)p[0];
                unsigned int noisetype=0; if(p.size()>1) noisetype=(unsigned int)p[1];
                cimglist_for(img,l) img(l).noise (sigma,noisetype);
            }
            break;
        case QUANTIZE:
            if(updateImage)
            {
                unsigned int nblevels=0; if(p.size()) nblevels=(unsigned int)p[0];
                cimglist_for(img,l) img(l).quantize (nblevels);
            }
            break;
        case THRESHOLD:
            if(updateImage)
            {
                T value=0; if(p.size()) value=(T)p[0];
                cimglist_for(img,l) img(l).threshold (value);
            }
            break;
        case LAPLACIAN:
            if(updateImage)
            {
                cimglist_for(img,l) img(l).laplacian ();
            }
            break;
        case STENSOR:
            if(updateImage)
            {
                unsigned int scheme=1; if(p.size()) scheme=(unsigned int)p[0];
                cimglist_for(img,l) img(l).structure_tensors (scheme);
            }
            break;
        case DISTANCE:
            if(updateImage || updateTransform)
            {
                T value=0; if(p.size()) value=(T)p[0];
                float factor=100; // hack to avoid cimg crash with small distances..
                float sizex=(float)inT->getScale()[0]*factor;
                float sizey=(float)inT->getScale()[1]*factor;
                float sizez=(float)inT->getScale()[2]*factor;
                cimglist_for(img,l) {img(l).distance ( value , sizex , sizey , sizez);  }
            }
            break;

        default:
            break;
        }
    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGEFILTER_H
