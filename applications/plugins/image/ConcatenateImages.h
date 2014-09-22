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
#ifndef SOFA_IMAGE_CONCATENATEIMAGES_H
#define SOFA_IMAGE_CONCATENATEIMAGES_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

#include <sofa/component/component.h>

#ifdef USING_OMP_PRAGMAS
    #include <omp.h>
#endif

#define INTERPOL_NEAREST 0
#define INTERPOL_LINEAR 1
#define INTERPOL_CUBIC 2

namespace sofa
{

namespace component
{

namespace engine
{

using helper::vector;
using defaulttype::Vec;
using defaulttype::Vector3;
using defaulttype::Mat;
using cimg_library::CImg;
using cimg_library::CImgList;

/**
 * This class merges N images with 1 channel into one with Nbref channel , Nbref determined from input images
 */


template <class _ImageTypes>
class ConcatenateImages : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ConcatenateImages,_ImageTypes),Inherited);

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;

#ifdef BUILD_ALL_IMAGE_TYPES
    typedef defaulttype::ImageI ImageIndiceType;
#else
    typedef defaulttype::ImageD ImageIndiceType;
#endif
    typedef typename ImageIndiceType::T Tindices;
    typedef helper::WriteAccessor<Data< ImageIndiceType > > waImageIndice;

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    Data<helper::OptionsGroup> Interpolation;
    Data<unsigned int> nbImages;

    helper::vector<Data<ImageTypes>*> inputImages;
    helper::vector<Data<TransformType>*> inputTransforms;

    Data<ImageTypes> weightImage;
    Data<ImageIndiceType> indiceImage;
//    Data<ImageTypes> indiceImage;
    Data<TransformType> transform;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ConcatenateImages<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    ConcatenateImages()    :   Inherited()
        , Interpolation( initData ( &Interpolation,"interpolation","Interpolation method." ) )
        , nbImages ( initData ( &nbImages,(unsigned int)0,"nbImages","number of images to merge" ) )
        , weightImage(initData(&weightImage,ImageTypes(),"weightImage","weightImage"))
        , indiceImage(initData(&indiceImage,ImageIndiceType(),"indiceImage","indiceImage"))
        , transform(initData(&transform,TransformType(),"transform","Transform"))
    {
        createInputImagesData();
        weightImage.setReadOnly(true);
        transform.setReadOnly(true);
        this->addAlias(&weightImage, "outputweightImage");
        this->addAlias(&transform, "outputTransform");
        this->addAlias(&indiceImage,"outputindiceImage");


        helper::OptionsGroup InterpolationOptions(3,"Nearest", "Linear", "Cubic");
        InterpolationOptions.setSelectedItem(INTERPOL_LINEAR);
        Interpolation.setValue(InterpolationOptions);
    }

    virtual ~ConcatenateImages()
    {
        deleteInputDataVector(inputImages);
        deleteInputDataVector(inputTransforms);
    }

    virtual void init()
    {
        addInput(&nbImages);
        createInputImagesData();

        addOutput(&weightImage);
        addOutput(&indiceImage);
        addOutput(&transform);

        setDirtyValue();
    }

    virtual void reinit()
    {
        createInputImagesData();
        update();
    }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        const char* p = arg->getAttribute(nbImages.getName().c_str());
        if (p)
        {
            std::string nbStr = p;
            sout << "parse: setting nbImages="<<nbStr<<sendl;
            nbImages.read(nbStr);
            createInputImagesData();
        }
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        std::map<std::string,std::string*>::const_iterator it = str.find(nbImages.getName());
        if (it != str.end() && it->second)
        {
            std::string nbStr = *it->second;
            sout << "parseFields: setting nbImages="<<nbStr<<sendl;
            nbImages.read(nbStr);
            createInputImagesData();
        }
        Inherit1::parseFields(str);
    }


protected:

    template<class t>
    void createInputDataVector(unsigned int nb, helper::vector< Data<t>* >& vf, std::string name, std::string help)
    {
        vf.reserve(nb);
        for (unsigned int i=vf.size(); i<nb; i++)
        {
            std::ostringstream oname; oname << name << (1+i); std::string name_i = oname.str();

            Data<t>* d = new Data<t>();
            d->setName(name_i);
            d->setHelpMsg(help.c_str());
            d->setReadOnly(true);

            vf.push_back(d);
            this->addData(d);
            this->addInput(d);
        }
    }
    template<class t>
    void deleteInputDataVector(helper::vector< Data<t>* >& vf)
    {
        for (unsigned int i=0; i<vf.size(); ++i)
        {
            this->delInput(vf[i]);
            delete vf[i];
        }
        vf.clear();
    }

    void createInputImagesData(int nb=-1)
    {
        unsigned int n = (nb < 0) ? nbImages.getValue() : (unsigned int)nb;

        createInputDataVector(n, inputImages, "image", "image");
        createInputDataVector(n, inputTransforms, "transform", "transform");
        if (n != nbImages.getValue())
            nbImages.setValue(n);
    }


    struct pttype  // to handle overlaps, we need to record some values and positions for each image
    {
        vector<double> vals;
        Coord u;
        vector<unsigned> indices;
    };


    // get the max number of channel
    unsigned getNbref(raImage outImage,raTransform outT, imCoord dim){

        unsigned Nbref = 0;
        const CImgList<T>& img = outImage->getCImgList();
        unsigned int nb = nbImages.getValue();

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif

        cimg_forXYZ(img(0),x,y,z) //space

        {
            Coord p = outT->fromImage(Coord(x,y,z)); //coordinate of voxel (x,y,z) in world space
            vector<struct pttype> pts;

            for(unsigned int j=0; j<nb; j++) // store values at p from input images
            {
                raImage in(this->inputImages[j]);
                const CImgList<T>& inImg = in->getCImgList();
                const imCoord indim=in->getDimensions();

                raTransform inT(this->inputTransforms[j]);
                Coord inp=inT->toImage(p); //corresponding voxel in image j
                if(inp[0]>=0 && inp[1]>=0 && inp[2]>=0 && inp[0]<=indim[0]-1 && inp[1]<=indim[1]-1 && inp[2]<=indim[2]-1)
                {
                    struct pttype pt;
                    if(Interpolation.getValue().getSelectedId()==INTERPOL_NEAREST)
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            //pt.vals.push_back(double());
                            pt.vals.push_back((double)inImg(t).atXYZ(sofa::helper::round((double)inp[0]),sofa::helper::round((double)inp[1]),sofa::helper::round((double)inp[2]),0));
                        }
                    else if(Interpolation.getValue().getSelectedId()==INTERPOL_LINEAR)
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            //pt.vals.push_back(double());
                            pt.vals.push_back((double)inImg(t).linear_atXYZ(inp[0],inp[1],inp[2],0));
                        }
                    else
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            //pt.vals.push_back(double());
                            pt.vals.push_back((double)inImg(t).cubic_atXYZ(inp[0],inp[1],inp[2],0));
                        }
                    pt.u=Coord( ( inp[0]< indim[0]-inp[0]-1)? inp[0]: indim[0]-inp[0]-1 ,
                            ( inp[1]< indim[1]-inp[1]-1)? inp[1]: indim[1]-inp[1]-1 ,
                            ( inp[2]< indim[2]-inp[2]-1)? inp[2]: indim[2]-inp[2]-1 ); // distance from border

                    bool isnotnull=false;
                    for(unsigned int t=0; t<pt.vals.size(); t++)  if(pt.vals[t]>(T)0) isnotnull=true;
                    if(isnotnull) pts.push_back(pt);

                }
            }
            if (pts.size()>Nbref) Nbref= pts.size();

        }
        return Nbref;
    }

    virtual void update()
    {
        cleanDirty();
        createInputImagesData();

        unsigned int nb = nbImages.getValue();
        if(!nb) return;

        Vec<2,Coord> BB = this->getBB(0);//bounding box of the output image
        Coord minScale;
        for (unsigned int j = 0 ; j < this->getScale(0).size(); j++)
            minScale[j] = fabs(this->getScale(0)[j]);
        for(unsigned int j=1; j<nb; j++)
        {
            Vec<2,Coord> bb = this->getBB(j);
            for(unsigned int k=0; k<bb[0].size(); k++)
            {
                //BB is axis-aligned
                if(BB[0][k]>bb[0][k]) BB[0][k]=bb[0][k];
                if(BB[1][k]<bb[1][k]) BB[1][k]=bb[1][k];
            }
            for(unsigned int k=0; k<3; k++)
                if( minScale[k] > fabs(this->getScale(j)[k]) )
                    minScale[k] = fabs(this->getScale(j)[k]);
        }

        // transform = translated version of inputTransforms[0] with minimum voxel size
        raTransform inT0(this->inputTransforms[0]);
        waTransform outT(this->transform);
        outT->operator=(inT0);
        outT->getRotation()=Coord(); //reset rotation because output image is axis aligned
        outT->getTranslation()=BB[0];
        outT->getScale()=minScale;
        outT->update(); //update internal quaternion depending on the rotation

        // set image
        raImage in0(this->inputImages[0]);
        if(in0->isEmpty()) return;

        imCoord dim=in0->getDimensions();
        dim[ImageTypes::DIMENSION_X]=fabs(BB[1][0] - BB[0][0]) / fabs(outT->getScale()[0]);
        dim[ImageTypes::DIMENSION_Y]=fabs(BB[1][1] - BB[0][1]) / fabs(outT->getScale()[1]);
        dim[ImageTypes::DIMENSION_Z]=fabs(BB[1][2] - BB[0][2]) / fabs(outT->getScale()[2]);

        waImage out(this->weightImage);
        waImageIndice outIndices(this->indiceImage);
        out->clear();
        outIndices->clear();

        unsigned Nbref=0;

        dim[3] = nb;
        out->setDimensions(dim);
        outIndices->setDimensions(dim);

        CImgList<T>& img = out->getCImgList();
        CImgList<Tindices>& imgIndices = outIndices->getCImgList();


        raImage outRead= this->weightImage;
        raTransform outTRead = this->transform;
        Nbref= getNbref(outRead,outTRead,dim);

        dim[3]=Nbref;
        out->setDimensions(dim);
        outIndices->setDimensions(dim);

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        cimg_forXYZ(img(0),x,y,z) //space
        {
            for(unsigned int t=0; t<dim[4]; t++) for(unsigned int k=0; k<dim[3]; k++){ img(t)(x,y,z,k) = (T)-1; imgIndices(t)(x,y,z,k)=(Tindices)-1;}

            Coord p = outT->fromImage(Coord(x,y,z)); //coordinate of voxel (x,y,z) in world space
            vector<struct pttype> pts;
            for(unsigned int j=0; j<nb; j++) // store values at p from input images
            {
                raImage in(this->inputImages[j]);
                const CImgList<T>& inImg = in->getCImgList();
                const imCoord indim=in->getDimensions();

                raTransform inT(this->inputTransforms[j]);
                Coord inp=inT->toImage(p); //corresponding voxel in image j
                if(inp[0]>=0 && inp[1]>=0 && inp[2]>=0 && inp[0]<=indim[0]-1 && inp[1]<=indim[1]-1 && inp[2]<=indim[2]-1)
                {
                    struct pttype pt;
                    if(Interpolation.getValue().getSelectedId()==INTERPOL_NEAREST)
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            //pt.vals.push_back(double());
                            pt.vals.push_back((double)inImg(t).atXYZ(sofa::helper::round((double)inp[0]),sofa::helper::round((double)inp[1]),sofa::helper::round((double)inp[2]),0));
                            pt.indices.push_back(j);
                        }
                    else if(Interpolation.getValue().getSelectedId()==INTERPOL_LINEAR)
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            //pt.vals.push_back(double());
                            pt.vals.push_back((double)inImg(t).linear_atXYZ(inp[0],inp[1],inp[2],0));
                            pt.indices.push_back(j);
                        }
                    else
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            //pt.vals.push_back(double());
                            pt.vals.push_back((double)inImg(t).cubic_atXYZ(inp[0],inp[1],inp[2],0));
                            pt.indices.push_back(j);
                        }
                    pt.u=Coord( ( inp[0]< indim[0]-inp[0]-1)? inp[0]: indim[0]-inp[0]-1 ,
                            ( inp[1]< indim[1]-inp[1]-1)? inp[1]: indim[1]-inp[1]-1 ,
                            ( inp[2]< indim[2]-inp[2]-1)? inp[2]: indim[2]-inp[2]-1 ); // distance from border

                    bool isnotnull=false;
                    for(unsigned int t=0; t<pt.vals.size(); t++)  if(pt.vals[t]>(T)0) isnotnull=true;
                    if(isnotnull) pts.push_back(pt);

                }
            }
            unsigned int nbp=pts.size();
            if(nbp==0) continue;
            else if(nbp==1){
                for(unsigned int t=0; t<pts[0].vals.size(); t++)  if((T)pts[0].vals[t]!=(T)0) {
                    img(t)(x,y,z,0) = (T)pts[0].vals[t];
                    imgIndices(t)(x,y,z,0)= (Tindices)pts[0].indices[t];
                }
            }
            else if(nbp>1)
            {
                unsigned int nbt=pts[0].vals.size();
                for (unsigned int j=0; j<nbp && j<Nbref;j++ ) for(unsigned int t=0; t<nbt; t++) {
                    img(t)(x,y,z,j)= pts[j].vals[t];
                    imgIndices(t)(x,y,z,j)= (Tindices)pts[j].indices[t];
                }
            }
        }

        sout << "Created merged image from " << nb << " input images." << sendl;
    }

    Vec<2,Coord> getBB(unsigned int i) // get image corners
    {
        Vec<2,Coord> BB;
        raImage rimage(this->inputImages[i]);
        raTransform rtransform(this->inputTransforms[i]);

        const imCoord dim= rimage->getDimensions();
        Vec<8,Coord> p;
        p[0]=Vector3(0,0,0);
        p[1]=Vector3(dim[0]-1,0,0);
        p[2]=Vector3(0,dim[1]-1,0);
        p[3]=Vector3(dim[0]-1,dim[1]-1,0);
        p[4]=Vector3(0,0,dim[2]-1);
        p[5]=Vector3(dim[0]-1,0,dim[2]-1);
        p[6]=Vector3(0,dim[1]-1,dim[2]-1);
        p[7]=Vector3(dim[0]-1,dim[1]-1,dim[2]-1);

        Coord tp=rtransform->fromImage(p[0]);
        BB[0]=tp;
        BB[1]=tp;
        for(unsigned int j=1; j<8; j++)
        {
            tp=rtransform->fromImage(p[j]);
            for(unsigned int k=0; k<tp.size(); k++)
            {
                if(BB[0][k]>tp[k]) BB[0][k]=tp[k];
                if(BB[1][k]<tp[k]) BB[1][k]=tp[k];
            }
        }
        return BB;
    }

    Coord getScale(unsigned int i)
    {
        Coord scale;
        raTransform rtransform(this->inputTransforms[i]);
        scale=rtransform->getScale();
        return scale;
    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_ConcatenateImages_H
