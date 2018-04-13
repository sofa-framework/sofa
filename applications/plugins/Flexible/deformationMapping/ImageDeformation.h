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
#ifndef SOFA_Flexible_ImageDeformation_H
#define SOFA_Flexible_ImageDeformation_H

#include <image/ImageTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>


#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include "BaseDeformationMapping.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <Eigen/Dense>

#define FORWARD_MAPPING 0
#define BACKWARD_MAPPING 1


namespace sofa
{
namespace component
{
namespace engine
{


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
    typedef helper::WriteOnlyAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteOnlyAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    static const int spatial_dimensions=3; // used in point mapper

    Data<helper::OptionsGroup> deformationMethod; ///< forward, backward
    Data<helper::OptionsGroup> interpolation; ///< Nearest,Linear,Cubic
    Data<bool> weightByVolumeChange; ///< for images representing densities, weight intensities according to the local volume variation
    Data< defaulttype::Vec<3,unsigned int> > dimensions; ///< output image dimensions

    typedef helper::vector<double> ParamTypes;
    typedef helper::ReadAccessor<Data< ParamTypes > > raParam;
    Data< ParamTypes > param; ///< Parameters

    Data< ImageTypes > inputImage;
    Data< TransformType > inputTransform;

    Data< ImageTypes > outputImage;
    Data< TransformType > outputTransform;


    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageDeformation<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    ImageDeformation()    :   Inherited()
        , deformationMethod ( initData ( &deformationMethod,"deformationMethod","" ) )
        , interpolation ( initData ( &interpolation,"interpolation","" ) )
        , weightByVolumeChange ( initData ( &weightByVolumeChange,false,"weightByVolumeChange","for images representing densities, weight intensities according to the local volume variation" ) )
        , dimensions ( initData ( &dimensions,defaulttype::Vec<3,unsigned int>(1,1,1),"dimensions","output image dimensions" ) )
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
                "2 - Backward deformation (tolerance=1e-5)");
        DefoOptions.setSelectedItem(FORWARD_MAPPING);
        deformationMethod.setValue(DefoOptions);

        helper::OptionsGroup interpOptions(3,"Nearest", "Linear", "Cubic");
        interpOptions.setSelectedItem(1);
        interpolation.setValue(interpOptions);
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
        if( !deformationMapping ) serr<<"No deformation mapping found"<<sendl;
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        if(!deformationMapping) return;

        raParam params(this->param);
        raImage in(this->inputImage);
        raTransform inT(this->inputTransform);
        raTransform outT(this->outputTransform);

        if(in->isEmpty()) return;
        const cimg_library::CImg<T>& img = in->getCImg(this->time);
        imCoord dim = in->getDimensions();

        waImage out(this->outputImage);
        imCoord outDim;
        outDim[0]=this->dimensions.getValue()[0];
        outDim[1]=this->dimensions.getValue()[1];
        outDim[2]=this->dimensions.getValue()[2];
        outDim[3]=dim[3]; // channels
        outDim[4]=1;      // time (current)
        out->setDimensions(outDim);
        cimg_library::CImg<T>& outImg = out->getCImg(0);

        unsigned int interp=this->interpolation.getValue().getSelectedId();

        switch(this->deformationMethod.getValue().getSelectedId())
        {
        case FORWARD_MAPPING:
        {
            outImg.fill(0);

            bool holefilling=true;  if(params.size()) holefilling=(bool)params[0];

            if(!holefilling) // paste transformed voxels into output image : fastest but sparse
            {
                if(weightByVolumeChange.getValue()) {serr<<"weightByVolumeChange not supported!"<<sendl;}
#ifdef _OPENMP
                #pragma omp parallel for
#endif
                for(int z=0; z<img.depth(); z++)
                    for(int y=0; y<img.height(); y++)
                        for(int x=0; x<img.width(); x++)
                        {
                            Coord p;  deformationMapping->ForwardMapping(p,inT->fromImage(Coord(x,y,z)));
                            if(p[0] || p[1] || p[2])  // discard non mapped points
                            {
                                Coord po = outT->toImage(p);
                                p[0]=sofa::helper::round(po[0]); p[1]=sofa::helper::round(po[1]); p[2]=sofa::helper::round(po[2]);
                                if(p[0]>=0) if(p[1]>=0) if(p[2]>=0) if(p[0]<outImg.width()) if(p[1]<outImg.height()) if(p[2]<outImg.depth())
                                                        cimg_forC(img,c) outImg(p[0],p[1],p[2],c) = img(x,y,z,c);
                            }
                        }
            }
            else  // paste transformed voxels inside bounding boxes of voxel corners
            {
                Real tolerance=1e-15; // tolerance for trilinear weights computation

                // create floating point image of transformed voxel corners
                cimg_library::CImg<Real> flt(dim[0]+1,dim[1]+1,dim[2]+1,3);

#ifdef _OPENMP
                #pragma omp parallel for
#endif
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
                Real dv0 = 1; if(weightByVolumeChange.getValue()) dv0 = inT->getScale()[0]*inT->getScale()[1]*inT->getScale()[2]/(outT->getScale()[0]*outT->getScale()[1]*outT->getScale()[2]);

#ifdef _OPENMP
                #pragma omp parallel for
#endif
                cimg_forXYZ(img,x,y,z)
                {
                    // get deformed voxel (x,y,z)
                    Coord pn[8]; for(unsigned int d=0; d<3; d++) { pn[0][d]=flt(x,y,z,d); pn[1][d]=flt(x+1,y,z,d); pn[2][d]=flt(x+1,y,z+1,d); pn[3][d]=flt(x,y,z+1,d); pn[4][d]=flt(x,y+1,z,d); pn[5][d]=flt(x+1,y+1,z,d); pn[6][d]=flt(x+1,y+1,z+1,d); pn[7][d]=flt(x,y+1,z+1,d); }

                    Real dv = dv0; if(weightByVolumeChange.getValue()) dv /= computeHexaVolume(pn); // local volume change supposing that voxels are cubes

                    // compute bounding box
                    Real BB[3][2] = { {-std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()}, {-std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()}, {-std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()} };
                    bool valid=true;
                    for(unsigned int i=0; i<8; i++)
                    {
                        if(!pn[i][0] && !pn[i][1] && !pn[i][2]) valid=false;
                        for(unsigned int d=0; d<3; d++) { if(BB[d][0]>pn[i][d] || i==0)  BB[d][0]=pn[i][d];  if(BB[d][1]<pn[i][d] || i==0)  BB[d][1]=pn[i][d]; }
                    }
                    if(!valid) continue;

                    // fill
                    for(unsigned int d=0; d<3; d++) { BB[d][0]=ceil(BB[d][0]);  BB[d][1]=floor(BB[d][1]); }
                    Coord w;
                    for(int zo=BB[2][0]; zo<=BB[2][1]; zo++)  if(zo>=0) if(zo<outImg.depth())
                                for(int yo=BB[1][0]; yo<=BB[1][1]; yo++)  if(yo>=0) if(yo<outImg.height())
                                            for(int xo=BB[0][0]; xo<=BB[0][1]; xo++) if(xo>=0) if(xo<outImg.width())
                                                    {
                                                        Coord p(xo,yo,zo);
                                                        computeTrilinearWeights(w, pn, p, tolerance);

                                                        if(w[0]>=-tolerance) if(w[0]<=1+tolerance)
                                                                if(w[1]>=-tolerance) if(w[1]<=1+tolerance)
                                                                        if(w[2]>=-tolerance) if(w[2]<=1+tolerance)
                                                                            {
                                                                                Coord pi(x+w[0]-0.5,y+w[1]-0.5,z+w[2]-0.5);
                                                                                if(interp==0)        cimg_forC(img,c) outImg(xo,yo,zo,c) =  (T)(dv*img.atXYZ(sofa::helper::round((double)pi[0]),sofa::helper::round((double)pi[1]),sofa::helper::round((double)pi[2]),c));
                                                                                else if(interp==2)   cimg_forC(img,c) outImg(xo,yo,zo,c) =  (T)(dv*img.cubic_atXYZ(pi[0],pi[1],pi[2],c,0,cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max()));
                                                                                else                        cimg_forC(img,c) outImg(xo,yo,zo,c) =  (T)(dv*img.linear_atXYZ(pi[0],pi[1],pi[2],c,0));
                                                                            }
                                                    }
                }
            }
        }
        break;

        case BACKWARD_MAPPING:
        {
            outImg.fill(0);

            Real tolerance=1e-5;  if(params.size()) tolerance=(Real)params[0];
            bool usekdtree=1;
            unsigned int nbMaxIt=10;

            if(weightByVolumeChange.getValue()) {serr<<"weightByVolumeChange not supported!"<<sendl;}

            if(usekdtree) {Coord p,p0,q; deformationMapping-> getClosestMappedPoint(p, p0, q, usekdtree); } // first, update kd tree to avoid conflict during parallelization

#ifdef _OPENMP
            #pragma omp parallel for
#endif
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
                                                    if(interp==0)        cimg_forC(img,c) outImg(x,y,z,c) = img.atXYZ(sofa::helper::round((double)pi[0]),sofa::helper::round((double)pi[1]),sofa::helper::round((double)pi[2]),c);
                                                    else if(interp==2)   cimg_forC(img,c) outImg(x,y,z,c) = (T)img.cubic_atXYZ(pi[0],pi[1],pi[2],c,0,cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max()); // warning cast for non-floating types
                                                    else                        cimg_forC(img,c) outImg(x,y,z,c) = (T)img.linear_atXYZ(pi[0],pi[1],pi[2],c,0); // warning cast for non-floating types
                                                }
                        }
                    }
        }
        break;

        default:
            break;
        }

        cleanDirty();
    }


    /// computes w such that x= p0*(1-wx)*(1-wz)*(1-wy) + p1*wx*(1-wz)*(1-wy) + p2*wx*wz*(1-wy) + p3*(1-wx)*wz*(1-wy) + p4*(1-wx)*(1-wz)*wy + p5*wx*(1-wz)*wy + p6*wx*wz*wy + p7*(1-wx)*wz*wy
    /// using Newton method
    void computeTrilinearWeights(Coord &w, const Coord p[8], const Coord &x,const Real &tolerance)
    {
        w[0]=0.5; w[1]=0.5; w[2]=0.5; // initial guess
        static const unsigned int MAXIT=20;
        static const Real MIN_DETERMINANT = 1.0e-100;
        unsigned int it=0;
        while( it < MAXIT)
        {
            Coord g(1.-w[0],1.-w[1],1.-w[2]);
            Coord f = p[0]*g[0]*g[2]*g[1] + p[1]*w[0]*g[2]*g[1] + p[2]*w[0]*w[2]*g[1] + p[3]*g[0]*w[2]*g[1] + p[4]*g[0]*g[2]*w[1] + p[5]*w[0]*g[2]*w[1] + p[6]*w[0]*w[2]*w[1] + p[7]*g[0]*w[2]*w[1] - x; // function to minimize
            if(f.norm2()<tolerance) {  return; }

            defaulttype::Mat<3,3,Real> df;
            df[0] = - p[0]*g[2]*g[1] + p[1]*g[2]*g[1] + p[2]*w[2]*g[1] - p[3]*w[2]*g[1] - p[4]*g[2]*w[1] + p[5]*g[2]*w[1] + p[6]*w[2]*w[1] - p[7]*w[2]*w[1];
            df[1] = - p[0]*g[0]*g[2] - p[1]*w[0]*g[2] - p[2]*w[0]*w[2] - p[3]*g[0]*w[2] + p[4]*g[0]*g[2] + p[5]*w[0]*g[2] + p[6]*w[0]*w[2] + p[7]*g[0]*w[2];
            df[2] = - p[0]*g[0]*g[1] - p[1]*w[0]*g[1] + p[2]*w[0]*g[1] + p[3]*g[0]*g[1] - p[4]*g[0]*w[1] - p[5]*w[0]*w[1] + p[6]*w[0]*w[1] + p[7]*g[0]*w[1];

            Real det=determinant(df);
            if ( -MIN_DETERMINANT<=det && det<=MIN_DETERMINANT) { return; }
            defaulttype::Mat<3,3,Real> dfinv;
            dfinv(0,0)= (df(1,1)*df(2,2) - df(2,1)*df(1,2))/det;
            dfinv(0,1)= (df(1,2)*df(2,0) - df(2,2)*df(1,0))/det;
            dfinv(0,2)= (df(1,0)*df(2,1) - df(2,0)*df(1,1))/det;
            dfinv(1,0)= (df(2,1)*df(0,2) - df(0,1)*df(2,2))/det;
            dfinv(1,1)= (df(2,2)*df(0,0) - df(0,2)*df(2,0))/det;
            dfinv(1,2)= (df(2,0)*df(0,1) - df(0,0)*df(2,1))/det;
            dfinv(2,0)= (df(0,1)*df(1,2) - df(1,1)*df(0,2))/det;
            dfinv(2,1)= (df(0,2)*df(1,0) - df(1,2)*df(0,0))/det;
            dfinv(2,2)= (df(0,0)*df(1,1) - df(1,0)*df(0,1))/det;

            w -= dfinv*f;
            it++;
        }
    }

    /// computes the signed volume of an hexahedron
    /// using hex subdivision into 5 tets -> fast but not exact (no curved quads)
    Real computeHexaVolume(const Coord p[8])
    {
        return computeTetVolume(p[0],p[2],p[7],p[3]) + computeTetVolume(p[0],p[1],p[5],p[2]) + computeTetVolume(p[2],p[5],p[7],p[6]) + computeTetVolume(p[0],p[5],p[7],p[2]) + computeTetVolume(p[0],p[5],p[4],p[7]);
    }

    /// computes the signed volume of an tetrahedron
    /// positive volume -> D in the direction of ABxAC
    Real computeTetVolume(const Coord &a,const Coord &b,const Coord &c,const Coord &d)
    {
        return dot(d-a,cross(b-a,c-a))/6.;
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
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
