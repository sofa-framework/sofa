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
#ifndef SOFA_IMAGE_IMAGEFILTER_H
#define SOFA_IMAGE_IMAGEFILTER_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

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
#define GRADIENT 17
#define HESSIAN 18
#define NORMALIZE 19
#define RESAMPLE 20
#define SELECTCHANNEL 21
#define SKELETON 22
#define MEANDIFFUSION 23
#define FILLHOLES 24
#define CONNECTEDCOMPONENTS 25
#define LARGESTCONNECTEDCOMPONENT 26
#define MIRROR 27
#define SHARPEN 28
#define EXPAND 29


namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class computes a filtered image
 */


template <class _InImageTypes,class _OutImageTypes>
class ImageFilter : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE2(ImageFilter,_InImageTypes,_OutImageTypes),Inherited);

    typedef _InImageTypes InImageTypes;
    typedef typename InImageTypes::T Ti;
    typedef typename InImageTypes::imCoord imCoordi;
    typedef helper::ReadAccessor<Data< InImageTypes > > raImagei;

    typedef _OutImageTypes OutImageTypes;
    typedef typename OutImageTypes::T To;
    typedef typename OutImageTypes::imCoord imCoordo;
    typedef helper::WriteOnlyAccessor<Data< OutImageTypes > > waImageo;

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteOnlyAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    typedef helper::vector<double> ParamTypes;
    typedef helper::ReadAccessor<Data< ParamTypes > > raParam;

    Data<helper::OptionsGroup> filter; ///< Filter
    Data< ParamTypes > param; ///< Parameters

    Data< InImageTypes > inputImage;
    Data< TransformType > inputTransform;

    Data< OutImageTypes > outputImage;
    Data< TransformType > outputTransform;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const ImageFilter<InImageTypes,OutImageTypes>* = NULL) { return InImageTypes::Name()+std::string(",")+OutImageTypes::Name(); }

    ImageFilter()    :   Inherited()
      , filter ( initData ( &filter,"filter","Filter" ) )
      , param ( initData ( &param,"param","Parameters" ) )
      , inputImage(initData(&inputImage,InImageTypes(),"inputImage",""))
      , inputTransform(initData(&inputTransform,TransformType(),"inputTransform",""))
      , outputImage(initData(&outputImage,OutImageTypes(),"outputImage",""))
      , outputTransform(initData(&outputTransform,TransformType(),"outputTransform",""))
    {
        inputImage.setReadOnly(true);
        inputTransform.setReadOnly(true);
        outputImage.setReadOnly(true);
        outputTransform.setReadOnly(true);
        helper::OptionsGroup filterOptions(30	,"0 - None"
                                           ,"1 - Blur ( sigma )"
                                           ,"2 - Blur Median ( n )"
                                           ,"3 - Blur Bilateral ( sigma_s, sigma_r)"
                                           ,"4 - Blur Anisotropic ( amplitude )"
                                           ,"5 - Deriche ( sigma , order , axis )"
                                           ,"6 - Crop ( xmin , ymin , zmin , xmax , ymax , zmax)"
                                           ,"7 - Resize( dimx , dimy , dimz , no interp.|nearest neighb.|moving av.|linear|grid|bicubic|lanzcos)"
                                           ,"8 - Trim ( tmin , tmax )"
                                           ,"9 - Dilate ( size )"
                                           ,"10 - Erode ( size )"
                                           ,"11 - Noise ( sigma , gaussian|uniform|Salt&Pepper|Poisson|Rician )"
                                           ,"12 - Quantize ( nbLevels )"
                                           ,"13 - Threshold ( min , max )"
                                           ,"14 - Laplacian"
                                           ,"15 - Structure tensors ( scheme )"
                                           ,"16 - Distance ( value, scale )"
                                           ,"17 - Gradient ( axis x | y | z | magnitude)"
                                           ,"18 - Hessian (axis1 , axis2) "
                                           ,"19 - Normalize ( out_min, out_max , in_min, in_max)"
                                           ,"20 - Resample ( ox , oy , oz , dimx , dimy , dimz , dx , dy , dz  , nearest neighb.|linear|cubic)"
                                           ,"21 - SelectChannels ( c0, c1 )"
                                           ,"22 - Skeleton from distance map"
                                           ,"23 - Mean Diffusion ( max iterations=0 (0->until convergence), fixed boundaries=1, exclude outside=1, threshold=eps )"
                                           ,"24 - Fill Holes (inval=0, outval=1)"
                                           ,"25 - Label connected components (tolerance=0)"
                                           ,"26 - Largest connected component (tolerance=0)"
                                           ,"27 - Mirror (axis=0)"
                                           ,"28 - Sharpen (sigma)"
                                           ,"29 - Expand ( size )"
                                           );
        filterOptions.setSelectedItem(NONE);
        filter.setValue(filterOptions);
    }

    virtual ~ImageFilter() {}

    virtual void init() override
	{
        addInput(&inputImage);
        addInput(&inputTransform);
        addOutput(&outputImage);
        addOutput(&outputTransform);
        setDirtyValue();
    }

    virtual void reinit() override { update(); }

protected:

    virtual void update() override
    {
        bool updateImage = this->inputImage.isDirty();	// change of input image -> update output image
        bool updateTransform = this->inputTransform.isDirty();	// change of input transform -> update output transform
        if(!updateImage && !updateTransform) {updateImage=true; updateTransform=true;}  // change of parameters -> update all

        raParam p(this->param);
        raTransform inT(this->inputTransform);
        raImagei in(this->inputImage);

        cleanDirty();

        waImageo out(this->outputImage);
        waTransform outT(this->outputTransform);

		if(in->isEmpty()) return;

        const cimg_library::CImgList<Ti>& inimg = in->getCImgList();
        cimg_library::CImgList<To>& img = out->getCImgList();
        if(updateImage) img.assign(inimg);	// copy
        if(updateTransform) outT->operator=(inT);	// copy

        switch(this->filter.getValue().getSelectedId())
        {
        case BLURDERICHE:
            if(updateImage)
            {
                float sigma=0; if(p.size()) sigma=(float)p[0];
                cimglist_for(img,l) img(l)=inimg(l).get_blur(sigma);
            }
            break;
        case BLURMEDIAN:
            if(updateImage)
            {
                unsigned int n=0; if(p.size()) n=(unsigned int)p[0];
                cimglist_for(img,l) img(l)=inimg(l).get_blur_median (n);
            }
            break;
        case BLURBILATERAL:
            if(updateImage)
            {
                float sigma_s=0;  if(p.size()) sigma_s=(float)p[0];
                float sigma_r=0; if(p.size()>1) sigma_r=(float)p[1];
                cimglist_for(img,l) img(l)=inimg(l).get_blur_bilateral (inimg(l), sigma_s,sigma_r);
            }
            break;
        case BLURANISOTROPIC:
            if(updateImage)
            {
                float amplitude=0; if(p.size()) amplitude=(float)p[0];
                cimglist_for(img,l) img(l)=inimg(l).get_blur_anisotropic (amplitude);
            }
            break;
        case DERICHE:
            if(updateImage)
            {
                float sigma=0;  if(p.size()) sigma=(float)p[0];
                unsigned int order=0; if(p.size()>1) order=(unsigned int)p[1];
                char axis='x';  if(p.size()>2) { if((int)p[2]==1) axis='y'; else if((int)p[2]==2) axis='z'; }
                cimglist_for(img,l) img(l)=inimg(l).get_deriche (sigma,order,axis);
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
                if(updateImage) cimglist_for(img,l) img(l)=inimg(l).get_crop(xmin,ymin,zmin,0,xmax,ymax,zmax,in->getDimensions()[3]-1);
                if(updateTransform)
                {
                    outT->getTranslation()=outT->fromImage( Coord((Real)xmin,(Real)ymin,(Real)zmin) );
                    outT->setCamPos((Real)(out->getDimensions()[0]-1)/2.0,(Real)(out->getDimensions()[1]-1)/2.0);
                }
            }
            break;
        case EXPAND:
            if(updateImage || updateTransform)
            {
                unsigned int size=0; if(p.size())   size=(unsigned int)p[0];
                unsigned int dim[3]= {in->getDimensions()[0],in->getDimensions()[1],in->getDimensions()[2]};
                if(updateImage) cimglist_for(img,l) img(l)=inimg(l).get_resize(dim[0]+2*size,dim[1]+2*size,dim[2]+2*size,-100,0,0,0.5,0.5,0.5,0);
                if(updateTransform)
                {
                    outT->getTranslation()-=outT->getScale()*size;
                    outT->setCamPos((Real)(out->getDimensions()[0]-1)/2.0,(Real)(out->getDimensions()[1]-1)/2.0);
                }
            }
            break;
        case RESIZE:
            if(updateImage || updateTransform)
            {
                unsigned int dim[3]= {in->getDimensions()[0],in->getDimensions()[1],in->getDimensions()[2]};
                if(p.size())   dim[0]=(unsigned int)p[0];
                if(p.size()>1) dim[1]=(unsigned int)p[1];
                if(p.size()>2) dim[2]=(unsigned int)p[2];
                unsigned int interpolation=1; if(p.size()>3) interpolation=(unsigned int)p[3];
                if(updateImage) cimglist_for(img,l) img(l)=inimg(l).get_resize(dim[0],dim[1],dim[2],-100,interpolation);
                if(updateTransform)
                    if(interpolation)
                    {
                        for(unsigned int k=0; k<3; k++)
                        {
                            if(dim[k]!=1 && in->getDimensions()[k]!=1) outT->getScale()[k]*=((Real)in->getDimensions()[k]-1.0)/((Real)dim[k]-1.0); // keep same origin
                            else // dim =1 -> keep same thickness and translate origin
                            {
                                outT->getTranslation()[k]+=(((Real)in->getDimensions()[k])/((Real)dim[k])-1.0)*0.5*outT->getScale()[k];
                                outT->getScale()[k]*=((Real)in->getDimensions()[k])/((Real)dim[k]);
                            }
                        }
                        outT->setCamPos((Real)(out->getDimensions()[0]-1)/2.0,(Real)(out->getDimensions()[1]-1)/2.0);
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
                cimglist_for(img,l) img(l)=inimg(l).get_dilate (size);
            }
            break;
        case ERODE:
            if(updateImage)
            {
                unsigned int size=0; if(p.size()) size=(unsigned int)p[0];
                cimglist_for(img,l) img(l)=inimg(l).get_erode (size);
            }
            break;
        case NOISE:
            if(updateImage)
            {
                float sigma=0;  if(p.size()) sigma=(float)p[0];
                unsigned int noisetype=0; if(p.size()>1) noisetype=(unsigned int)p[1];
                cimglist_for(img,l) img(l)=inimg(l).get_noise (sigma,noisetype);
            }
            break;
        case QUANTIZE:
            if(updateImage)
            {
                unsigned int nblevels=0; if(p.size()) nblevels=(unsigned int)p[0];
                cimglist_for(img,l) img(l)=inimg(l).get_quantize (nblevels);
            }
            break;
        case THRESHOLD:
            if(updateImage)
            {
                Ti valuemin=cimg_library::cimg::type<Ti>::min(); if(p.size()) valuemin=(Ti)p[0];
                Ti valuemax=cimg_library::cimg::type<Ti>::max(); if(p.size()>1) valuemax=(Ti)p[1];

                cimglist_for(img,l)
                        cimg_forXYZ(img(l),x,y,z)
                {
                    if(inimg(l)(x,y,z)>=valuemin && inimg(l)(x,y,z)<=valuemax) img(l)(x,y,z)=(To)1;
                    else img(l)(x,y,z)=(To)0;
                }
            }
            break;
        case LAPLACIAN:
            if(updateImage)
            {
                cimglist_for(img,l) img(l)=inimg(l).get_laplacian ();
            }
            break;
        case STENSOR:
            if(updateImage)
            {
                unsigned int scheme=1; if(p.size()) scheme=(unsigned int)p[0];
                cimglist_for(img,l) img(l)=inimg(l).get_structure_tensors (scheme);
            }
            break;
        case DISTANCE:
            if(updateImage || updateTransform)
            {
                Ti value=0; if(p.size()) value=(Ti)p[0];
                float scale=1; if(p.size()>1) scale=(float)p[1];
                float sizex=(float)inT->getScale()[0]*scale;
                float sizey=(float)inT->getScale()[1]*scale;
                float sizez=(float)inT->getScale()[2]*scale;
                cimg_library::CImg<float> metric_distance(2,2,2,1,0);
                metric_distance(1,0,0)=sizex;
                metric_distance(0,1,0)=sizey;
                metric_distance(0,0,1)=sizez;
                metric_distance(1,1,0)=sqrt(sizex*sizex+sizey*sizey);
                metric_distance(1,0,1)=sqrt(sizex*sizex+sizez*sizez);
                metric_distance(0,1,1)=sqrt(sizey*sizey+sizez*sizez);
                metric_distance(1,1,1)=sqrt(sizex*sizex+sizey*sizey+sizez*sizez);
                cimglist_for(img,l) {img(l)=inimg(l).get_distance ( value , metric_distance);  }
            }
            break;
        case GRADIENT:
            if(updateImage || updateTransform)
            {
                char axis='a';  if(p.size()) { if((int)p[0]==0) axis='x'; else if((int)p[0]==1) axis='y'; else if((int)p[0]==2) axis='z'; }

                CImg_3x3x3(I,To);
                cimglist_for(img,l)
                {
                    To *ptrd = img(l)._data;
                    // Central finite differences.
                    if(axis=='x') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)      *(ptrd++) = (Incc - Ipcc)*(To)0.5/(To)inT->getScale()[0];
                    else if(axis=='y') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To) *(ptrd++) = (Icnc - Icpc)*(To)0.5/(To)inT->getScale()[1];
                    else if(axis=='z') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To) *(ptrd++) = (Iccn - Iccp)*(To)0.5/(To)inT->getScale()[2];
                    else  cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)
                    {
                        To ix = (Incc - Ipcc)*(To)0.5/(To)inT->getScale()[0];
                        To iy = (Icnc - Icpc)*(To)0.5/(To)inT->getScale()[1];
                        To iz = (Iccn - Iccp)*(To)0.5/(To)inT->getScale()[2];
                        *(ptrd++) = (To)sqrt( (SReal) ix*ix+iy*iy+iz*iz);
                    }
                }
            }
            break;
        case HESSIAN:
            if(updateImage || updateTransform)
            {
                char axis1='x';  if(p.size()) { if((int)p[0]==1) axis1='y'; else if((int)p[0]==2) axis1='z'; }
                char axis2='x';  if(p.size()>1) { if((int)p[1]==1) axis2='y'; else if((int)p[1]==2) axis2='z'; }
                if (axis1>axis2) cimg_library::cimg::swap(axis1,axis2);
                CImg_3x3x3(I,To);
                cimglist_for(img,l)
                {
                    To *ptrd = img(l)._data;
                    // Central finite differences.
                    if(axis1=='x' && axis2=='x') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)  *(ptrd++) = (Ipcc + Incc - 2*Iccc)              /(To)(inT->getScale()[0]*inT->getScale()[0]);
                    else if(axis1=='x' && axis2=='y') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)  *(ptrd++) = (Ippc + Innc - Ipnc - Inpc)*(To)0.25/(To)(inT->getScale()[0]*inT->getScale()[1]);
                    else if(axis1=='x' && axis2=='z') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)  *(ptrd++) = (Ipcp + Incn - Ipcn - Incp)*(To)0.25/(To)(inT->getScale()[0]*inT->getScale()[2]);
                    else if(axis1=='y' && axis2=='y') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)  *(ptrd++) = (Icpc + Icnc - 2*Iccc)              /(To)(inT->getScale()[1]*inT->getScale()[1]);
                    else if(axis1=='y' && axis2=='z') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)  *(ptrd++) = (Icpp + Icnn - Icpn - Icnp)*(To)0.25/(To)(inT->getScale()[1]*inT->getScale()[2]);
                    else if(axis1=='z' && axis2=='z') cimg_forC(inimg(l),c) cimg_for3x3x3(inimg(l),x,y,z,c,I,To)  *(ptrd++) = (Iccn + Iccp - 2*Iccc)              /(To)(inT->getScale()[2]*inT->getScale()[2]);
                }
            }
            break;

        case NORMALIZE:
            if(updateImage)
            {
                To o1=cimg_library::cimg::type<To>::min();    if(p.size())   o1=(To)p[0];
                To o2=cimg_library::cimg::type<To>::max();    if(p.size()>1) o2=(To)p[1];
                Ti i1=cimg_library::cimg::type<Ti>::min();    if(p.size()>2) i1=(Ti)p[2];
                Ti i2=cimg_library::cimg::type<Ti>::max();    if(p.size()>3) i2=(Ti)p[3];
                cimglist_for(img,l) {img(l)=inimg(l).get_cut(i1 , i2).get_normalize( (Ti)o1, (Ti)o2); }
            }
            break;
        case RESAMPLE:
            if(updateImage || updateTransform)
            {
                Coord origin = inT->getTranslation();
                if(p.size()>0) origin[0]=(Real)p[0];
                if(p.size()>1) origin[1]=(Real)p[1];
                if(p.size()>2) origin[2]=(Real)p[2];
                unsigned int dimx=in->getDimensions()[0]; if(p.size()>3) dimx=(unsigned int)p[3];
                unsigned int dimy=in->getDimensions()[1]; if(p.size()>4) dimy=(unsigned int)p[4];
                unsigned int dimz=in->getDimensions()[2]; if(p.size()>5) dimz=(unsigned int)p[5];
                Coord scale = inT->getScale();
                if(p.size()>6) scale[0]=(Real)p[6];
                if(p.size()>7) scale[1]=(Real)p[7];
                if(p.size()>8) scale[2]=(Real)p[8];
                unsigned int interpolation=1; if(p.size()>9) interpolation=(unsigned int)p[9];

                outT->getTranslation() = origin;
                outT->getScale() = scale;
                outT->getRotation() = Coord();
                if(dimz!=1) outT->isPerspective()=0;
                else outT->setCamPos((Real)(out->getDimensions()[0]-1)/2.0,(Real)(out->getDimensions()[1]-1)/2.0);
                outT->update();


                unsigned int nbc=in->getDimensions()[3];
                Ti OutValue=(Ti)0.;
                cimglist_for(img,l)
                {
                    img(l).resize(dimx,dimy,dimz,nbc);
                    cimg_forXYZ(img(l),x,y,z)
                    {
                        Coord p=inT->toImage(outT->fromImage(Coord(x,y,z)));
                        if(p[0]<-0.5 || p[1]<-0.5 || p[2]<-0.5 || p[0]>inimg(l).width()-0.5 || p[1]>inimg(l).height()-0.5 || p[2]>inimg(l).depth()-0.5)
                            for(unsigned int k=0; k<nbc; k++) img(l)(x,y,z,k) = OutValue;
                        else
                        {
                            if(interpolation==0) for(unsigned int k=0; k<nbc; k++) img(l)(x,y,z,k) = (To) inimg(l).atXYZ(sofa::helper::round((double)p[0]),sofa::helper::round((double)p[1]),sofa::helper::round((double)p[2]),k);
                            else if(interpolation==1) for(unsigned int k=0; k<nbc; k++) img(l)(x,y,z,k) = (To) inimg(l).linear_atXYZ(p[0],p[1],p[2],k,OutValue);
                            else if(interpolation==2) for(unsigned int k=0; k<nbc; k++) img(l)(x,y,z,k) = (To) inimg(l).cubic_atXYZ(p[0],p[1],p[2],k,OutValue,cimg_library::cimg::type<Ti>::min(),cimg_library::cimg::type<Ti>::max());
                        }
                    }
                }

            }
            break;
        case SELECTCHANNEL:
            if(updateImage)
            {
                unsigned int c0=0;    if(p.size())   c0=(unsigned int)p[0];
                unsigned int c1=c0;    if(p.size()>1)   c1=(unsigned int)p[1];
                cimglist_for(img,l) {img(l)=inimg(l).get_channels(c0,c1);  }
            }
            break;

        case SKELETON:
            if(updateImage)
            {
                bool curve=true;            //if(p.size())   curve=(bool)p[0];
                float thresh = -0.3f;       //if(p.size()>1)   thresh=(float)p[1];
                float dlt1 = 2, dlt2 = 1;         //if(p.size()>2)   dlt2=(float)p[2];

                cimglist_for(img,l)
                {
                    const cimg_library::CImgList<Real> grad = inimg(l).get_gradient("xyz");
                    cimg_library::CImg<Real> flux = inimg(l).get_flux(grad,1,1);
                    if (dlt2) // correction proposed by Torsello 03
                    {
                        cimg_library::CImg<Real> logdensity = inimg(l).get_logdensity(inimg(l),grad,flux,dlt1);
                        flux = inimg(l).get_corrected_flux(logdensity,grad,flux,dlt2);
                    }
                    img(l) = inimg(l).get_skeleton(flux,inimg(l),curve,thresh);
                }
            }
            break;

        case MEANDIFFUSION:
            if(updateImage)
            {
                typename InImageTypes::imCoord dim = in->getDimensions();

                // a mask to know which pixel to compute
                cimg_library::CImg<bool> mask;
                mask.assign( dim[0], dim[1], dim[2], 1 );

                unsigned int maxDiffusionIterations=0; if(p.size()) maxDiffusionIterations=(unsigned int)p[0];
                bool fixedBoundaries=true; if(p.size()>1) fixedBoundaries=(p[1]!=0);
                bool excludeOutside=true; if(p.size()>2) excludeOutside=(p[2]!=0);
                To threshold=std::numeric_limits<To>::epsilon(); if(p.size()>3) threshold=(To)p[3];

                cimg_library::CImg<To> imTmp;

                cimglist_for(inimg,l)
                {
                    // fill mask
                    if( fixedBoundaries )
                    {
                        mask.assign( img(l) ); // create boolean mask by copy and automatic cast

                        if( excludeOutside )
                        {
                            bool fillColor = true;
                            mask.draw_fill(0,0,0,&fillColor);
                        }
                    }

                    imTmp.assign( img(l) ); // copy

                    bool change = true;
                    unsigned i;

                    for ( i = 0 ; change && ( maxDiffusionIterations==0 || i < maxDiffusionIterations ) ; ++i )
                    {
                        change = false;

                        cimg_forXYZ(mask,x,y,z)
                        {
                            if( mask(x,y,z) == false ) // to compute
                            {
                                SReal mean = (SReal)0.0;
                                unsigned int nb = 0;
                                for(int xx=x-1;xx<=x+1;++xx)
                                    for(int yy=y-1;yy<=y+1;++yy)
                                        for(int zz=z-1;zz<=z+1;++zz)
                                        {
                                            if( xx >= 0 && xx<mask.width() &&
                                                    yy >= 0 && yy<mask.height() &&
                                                    zz >= 0 && zz<mask.depth() )
                                            {
                                                ++nb;
                                                mean+=(SReal)img(l)(xx,yy,zz);
                                            }
                                        }
                                mean /= (SReal)nb;

                                assert( nb!=0 );


                                imTmp(x, y, z) = (To)mean;

                                if( !helper::isEqual( (To)mean, img(l)(x, y, z), threshold ) )
                                {
                                    change = true;
                                }
                            }
                        }

                        if( change ) img(l).swap(imTmp);
                    }

                    if(this->f_printLog.getValue()) std::cout<<SOFA_CLASS_METHOD<<"MEANDIFFUSION ("<<this->getName()<<"): "<<i<<" diffusion iterations on "<<l<<"-th image"<<std::endl;

                }

            }
            break;


        case FILLHOLES:
            if(updateImage)
            {
                To inval=(To)1.;    if(p.size()) inval=(To)p[0];
                To outval=(To)0;    if(p.size()>1)   outval=(To)p[1];

                cimglist_for(img,l)
                {
                    cimg_library::CImg<unsigned char> im = inimg(l);
                    cimg_foroff(im,off) if( im[off]!=0 ) im[off]=1;
                    unsigned char fillColor = (unsigned char)2;
                    im.draw_fill(0,0,0,&fillColor); // flood fill from voxel (0,0,0)
                    cimg_foroff(im,off) if( im[off]==2 ) img(l)[off]=outval; else img(l)[off]=inval;
                }
            }
            break;

        case CONNECTEDCOMPONENTS:
            if(updateImage)
            {
                float tol=0;    if(p.size()) tol=(float)p[0];

                cimglist_for(img,l)
                {
                    img(l).label(false,tol);
                }
            }
            break;


        case LARGESTCONNECTEDCOMPONENT:
            if(updateImage)
            {
                float tol=0;    if(p.size()) tol=(float)p[0];

                cimglist_for(img,l)
                {
                    typedef unsigned int Tlabel;
                    cimg_library::CImg<Tlabel> im = inimg(l);
                    im.label(false,tol);
                    //histo
                    std::map<Tlabel,unsigned long> histo;
                    cimg_foroff(im,off) histo[im[off]]++;
                    Tlabel val=0; unsigned long mx=0;
                    // get max size
                    for (typename std::map<Tlabel,unsigned long>::iterator it=histo.begin(); it!=histo.end(); ++it) if(it->second>=mx && it->first!=(Tlabel)0.) { mx=it->second; val=it->first; }
                    // mask input
                    cimg_foroff(im,off) if(im[off]==val) img(l)[off]=(To)inimg(l)[off]; else     img(l)[off]=(To)0.;
                }
            }
            break;

        case MIRROR:
            if(updateImage)
            {
                unsigned int axis=0; if(p.size()) axis=(unsigned int)p[0];
                if(axis==0) cimglist_for(img,l) img(l)=inimg(l).get_mirror ('x');
                else if(axis==0) cimglist_for(img,l) img(l)=inimg(l).get_mirror ('y');
                else cimglist_for(img,l) img(l)=inimg(l).get_mirror ('z');
            }
            break;

        case SHARPEN:
            if(updateImage)
            {
                float sigma=0; if(p.size()) sigma=(float)p[0];
                cimglist_for(img,l) img(l)=inimg(l).get_sharpen(sigma);
            }
            break;

        default:
            break;
        }

        if (updateTransform) outT->update(); // update internal data

    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGEFILTER_H
