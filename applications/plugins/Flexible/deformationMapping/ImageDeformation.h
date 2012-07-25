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
#ifndef SOFA_Flexible_ImageDeformation_H
#define SOFA_Flexible_ImageDeformation_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

#include <sofa/component/component.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include "BaseDeformationMapping.h"

#include <omp.h>

#define FORWARD_MAPPING 0
#define BACKWARD_MAPPING 1


namespace sofa
{
namespace component
{
namespace engine
{

using helper::vector;
using helper::round;
using defaulttype::Vec;
using namespace cimg_library;

/**
 * This class deforms an image based on an existing deformation mapping using forward or inverse mapping
 * output image dimensions and transformation need to be specified
 */


template <class _ImageTypes>
class ImageDeformation : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageDeformation,_ImageTypes),Inherited);

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    static const int spatial_dimensions=3; // used in point mapper

    Data<helper::OptionsGroup> deformationMethod; ///< forward, backward
    Data< Vec<3,unsigned int> > dimensions;

    typedef vector<double> ParamTypes;
    typedef helper::ReadAccessor<Data< ParamTypes > > raParam;
    Data< ParamTypes > param;

    Data< ImageTypes > inputImage;
    Data< TransformType > inputTransform;

    Data< ImageTypes > outputImage;
    Data< TransformType > outputTransform;


    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageDeformation<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    ImageDeformation()    :   Inherited()
        , deformationMethod ( initData ( &deformationMethod,"deformationMethod","Deformation method" ) )
        , dimensions ( initData ( &dimensions,Vec<3,unsigned int>(1,1,1),"dimensions","output image dimensions" ) )
        , param ( initData ( &param,"param","Parameters" ) )
        , inputImage(initData(&inputImage,ImageTypes(),"inputImage",""))
        , inputTransform(initData(&inputTransform,TransformType(),"inputTransform",""))
        , outputImage(initData(&outputImage,ImageTypes(),"outputImage",""))
        , outputTransform(initData(&outputTransform,TransformType(),"outputTransform",""))
        , deformationMapping(NULL)
        , time((unsigned int)0)
    {
        // f_listening.setValue(true); // listening -> update at each time step. Disabled by default
        inputImage.setReadOnly(true);
        inputTransform.setReadOnly(true);
        outputImage.setReadOnly(true);

        helper::OptionsGroup DefoOptions(2,"1 - Forward deformation (hole filling=true)",
                "2 - Backward deformation (Nearest|Linear|Cubic interpolation , tolerance=1e-5)");
        DefoOptions.setSelectedItem(FORWARD_MAPPING);
        deformationMethod.setValue(DefoOptions);
    }

    virtual ~ImageDeformation() {}

    virtual void init()
    {
        addInput(&inputImage);
        addInput(&inputTransform);
        addInput(&dimensions);
        addInput(&outputTransform);

        addOutput(&outputImage);
        setDirtyValue();

        this->getContext()->get( deformationMapping, core::objectmodel::BaseContext::Local);
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        cleanDirty();

        if(!deformationMapping) return;

        raParam params(this->param);
        raImage in(this->inputImage);
        raTransform inT(this->inputTransform);
        raTransform outT(this->outputTransform);

        if(!in->getCImgList().size()) return;
        const CImg<T>& img = in->getCImg(this->time);
        imCoord dim = in->getDimensions();

        waImage out(this->outputImage);
        imCoord outDim;
        outDim[0]=this->dimensions.getValue()[0];
        outDim[1]=this->dimensions.getValue()[1];
        outDim[2]=this->dimensions.getValue()[2];
        outDim[3]=dim[3]; // channels
        outDim[4]=1;      // time (current)
        out->setDimensions(outDim);
        CImg<T>& outImg = out->getCImg(0);

        switch(this->deformationMethod.getValue().getSelectedId())
        {
        case FORWARD_MAPPING:
        {
            outImg.fill(0);

            bool holefilling=true;  if(params.size()) holefilling=(bool)params[0];

            if(!holefilling) // paste transformed voxels into output image : fastest but sparse
            {
                #pragma omp parallel for
                for(int z=0; z<img.depth(); z++)
                    for(int y=0; y<img.height(); y++)
                        for(int x=0; x<img.width(); x++)
                        {
                            Coord p;  deformationMapping->ForwardMapping(p,inT->fromImage(Coord(x,y,z)));
                            if(p[0] || p[1] || p[2])  // discard non mapped points
                            {
                                Coord po = outT->toImage(p);
                                p[0]=round(po[0]); p[1]=round(po[1]); p[2]=round(po[2]);
                                if(p[0]>=0) if(p[1]>=0) if(p[2]>=0) if(p[0]<outImg.width()) if(p[1]<outImg.height()) if(p[2]<outImg.depth())
                                                        cimg_forC(img,c) outImg(p[0],p[1],p[2],c) = img(x,y,z,c);
                            }
                        }
            }
            else  // paste transformed voxels inside bounding boxes of voxel corners
            {
                // create floating point image of transformed voxel corners
                CImg<Real> flt(dim[0]+1,dim[1]+1,dim[2]+1,3);
                #pragma omp parallel for
                for(int z=0; z<=img.depth(); z++)
                    for(int y=0; y<=img.height(); y++)
                        for(int x=0; x<=img.width(); x++)
                        {
                            Coord p;  deformationMapping->ForwardMapping(p,inT->fromImage(Coord(x-0.5,y-0.5,z-0.5)));
                            Coord po;
                            if(p[0] || p[1] || p[2]) po = outT->toImage(p); // discard non mapped points
                            else po=p;
                            for(unsigned int d=0; d<3; d++) flt(x,y,z,d)=po[d];
                        }

                // paste values
                CImg<Real> accu(outDim[0],outDim[1],outDim[2],outDim[3]);    accu.fill(0);
                CImg<unsigned int> count(outDim[0],outDim[1],outDim[2],1);   count.fill(0);

                cimg_forXYZ(img,x,y,z)
                {
                    Real BB[3][2];
                    Coord pn;
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x,y,z,d);       if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { BB[d][0]=BB[d][1]=pn[d]; }
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x+1,y,z,d);     if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[d])  BB[d][0]=pn[d];  if(BB[d][1]<pn[d])  BB[d][1]=pn[d]; }
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x,y+1,z,d);     if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[d])  BB[d][0]=pn[d];  if(BB[d][1]<pn[d])  BB[d][1]=pn[d]; }
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x+1,y+1,z,d);   if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[d])  BB[d][0]=pn[d];  if(BB[d][1]<pn[d])  BB[d][1]=pn[d]; }
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x,y,z+1,d);     if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[d])  BB[d][0]=pn[d];  if(BB[d][1]<pn[d])  BB[d][1]=pn[d]; }
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x+1,y,z+1,d);   if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[d])  BB[d][0]=pn[d];  if(BB[d][1]<pn[d])  BB[d][1]=pn[d]; }
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x,y+1,z+1,d);   if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[d])  BB[d][0]=pn[d];  if(BB[d][1]<pn[d])  BB[d][1]=pn[d]; }
                    for(unsigned int d=0; d<3; d++) pn[d]=flt(x+1,y+1,z+1,d); if(!pn[0] && !pn[1] && !pn[2]) continue;    for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[d])  BB[d][0]=pn[d];  if(BB[d][1]<pn[d])  BB[d][1]=pn[d]; }

                    for(unsigned int d=0; d<3; d++) { BB[d][0]=floor(BB[d][0]);  BB[d][1]=ceil(BB[d][1]); }
                    for(int zo=BB[2][0]; zo<=BB[2][1]; zo++)  if(zo>=0) if(zo<outImg.depth())
                                for(int yo=BB[1][0]; yo<=BB[1][1]; yo++)  if(yo>=0) if(yo<outImg.height())
                                            for(int xo=BB[0][0]; xo<=BB[0][1]; xo++) if(xo>=0) if(xo<outImg.width())
                                                    {
                                                        cimg_forC(img,c) accu(xo,yo,zo,c) += (Real)img(x,y,z,c);
                                                        count(xo,yo,zo) ++;
                                                    }
                }
                #pragma omp parallel for
                for(int z=0; z<outImg.depth(); z++)
                    for(int y=0; y<outImg.height(); y++)
                        for(int x=0; x<outImg.width(); x++)
                            if(count(x,y,z))
                                cimg_forC(outImg,c)
                            {
                                Real v = (Real) accu(x,y,z,c) / (Real)count(x,y,z);
                                outImg(x,y,z,c) = (T)v;
                            }
            }
        }
        break;

        case BACKWARD_MAPPING:
        {
            outImg.fill(0);

            int interpolation=1;  if(params.size()) interpolation=(int)params[0];
            Real tolerance=1e-5;  if(params.size()>1) tolerance=(Real)params[1];
            bool usekdtree=1;
            unsigned int nbMaxIt=10;

            if(usekdtree) {Coord p,p0,q; deformationMapping-> getClosestMappedPoint(p, p0, q, usekdtree); } // first, update kd tree to avoid conflict during parallelization

            #pragma omp parallel for
            for(int z=0; z<outImg.depth(); z++)
                for(int y=0; y<outImg.height(); y++)
                    for(int x=0; x<outImg.width(); x++)
                    {
                        Coord p=outT->fromImage(Coord(x,y,z)),p0,q;
                        deformationMapping-> getClosestMappedPoint(p, p0, q, usekdtree);
                        deformationMapping->BackwardMapping(p0,p,tolerance,nbMaxIt);
                        Coord pi;
                        if(p0[0] || p0[1] || p0[2]) // discard non mapped points
                        {
                            pi = inT->toImage(p0);
                            if(pi[0]>=0) if(pi[1]>=0) if(pi[2]>=0) if(pi[0]<img.width()) if(pi[1]<img.height()) if(pi[2]<img.depth())
                                                {
                                                    if(interpolation==0)        cimg_forC(img,c) outImg(x,y,z,c) = img.atXYZ(round((double)pi[0]),round((double)pi[1]),round((double)pi[2]),c);
                                                    else if(interpolation==2)   cimg_forC(img,c) outImg(x,y,z,c) = img.cubic_atXYZ(pi[0],pi[1],pi[2],c,0,cimg::type<T>::min(),cimg::type<T>::max());
                                                    else                        cimg_forC(img,c) outImg(x,y,z,c) = img.linear_atXYZ(pi[0],pi[1],pi[2],c,0);
                                                }
                        }
                    }
        }
        break;

        default:
            break;
        }
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            raImage in(this->inputImage);
            raTransform inT(this->inputTransform);

            // get current time modulo dimt
            const unsigned int dimt=in->getDimensions()[4];
            if(!dimt) return;
            Real t=inT->toImage(this->getContext()->getTime()) ;
            t-=(Real)((int)((int)t/dimt)*dimt);
            t=(t-floor(t)>0.5)?ceil(t):floor(t); // nearest
            if(t<0) t=0.0; else if(t>=(Real)dimt) t=(Real)dimt-1.0; // clamp

            if(this->time!=(unsigned int)t) { this->time=(unsigned int)t;  }

            update(); // update at each time step (deformation has changed)
        }
    }

    mapping::BasePointMapper<spatial_dimensions,Real>* deformationMapping;
    unsigned int time;
};


} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_Flexible_ImageDeformation_H
