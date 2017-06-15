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
#ifndef SOFA_IMAGE_MERGEIMAGES_H
#define SOFA_IMAGE_MERGEIMAGES_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/vectorData.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#define AVERAGE 0
#define ORDER 1
#define ALPHABLEND 2
#define SEPARATE 3
#define ADDITIVE 4
#define INTERSECT 5

#define INTERPOLATION_NEAREST 0
#define INTERPOLATION_LINEAR 1
#define INTERPOLATION_CUBIC 2

namespace sofa
{

namespace component
{

namespace engine
{


/**
 * This class merges images into one
 */


template <class _ImageTypes>
class MergeImages : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MergeImages,_ImageTypes),Inherited);

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

    Data<helper::OptionsGroup> overlap;
    Data<helper::OptionsGroup> Interpolation;
    Data<unsigned int> nbImages;

    helper::vectorData<ImageTypes> inputImages;
    helper::vectorData<TransformType> inputTransforms;

    Data<ImageTypes> image;
    Data<TransformType> transform;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const MergeImages<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    MergeImages()    :   Inherited()
        , overlap ( initData ( &overlap,"overlap","method for handling overlapping regions" ) )
        , Interpolation( initData ( &Interpolation,"interpolation","Interpolation method." ) )
        , nbImages ( initData ( &nbImages,(unsigned int)0,"nbImages","number of images to merge" ) )
        , inputImages(this, "image", "input image")
        , inputTransforms(this, "transform", "input transform")
        , image(initData(&image,ImageTypes(),"image","Image"))
        , transform(initData(&transform,TransformType(),"transform","Transform"))
    {
        inputImages.resize(nbImages.getValue());
        inputTransforms.resize(nbImages.getValue());
        image.setReadOnly(true);
        transform.setReadOnly(true);
        this->addAlias(&image, "outputImage");
        this->addAlias(&transform, "outputTransform");

        helper::OptionsGroup overlapOptions(6	,"0 - Average pixels"
                ,"1 - Use image order as priority"
                ,"2 - Alpha blending according to distance from border"
                ,"3 - Take farthest pixel from border"
                ,"4 - Add pixels of each images"
                ,"5 - Set overlapping pixels of the first image to zero (only if the corresponding pixel in the other images different to zero)"
                                           );
        overlapOptions.setSelectedItem(ALPHABLEND);
        overlap.setValue(overlapOptions);

        helper::OptionsGroup InterpolationOptions(3,"Nearest", "Linear", "Cubic");
        InterpolationOptions.setSelectedItem(INTERPOLATION_LINEAR);
        Interpolation.setValue(InterpolationOptions);
    }

    virtual ~MergeImages()
    { }

    virtual void init()
    {
        addInput(&nbImages);
        inputImages.resize(nbImages.getValue());
        inputTransforms.resize(nbImages.getValue());

        addOutput(&image);
        addOutput(&transform);

        setDirtyValue();
    }

    virtual void reinit()
    {
        inputImages.resize(nbImages.getValue());
        inputTransforms.resize(nbImages.getValue());
        update();
    }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        inputImages.parseSizeData(arg, nbImages);
        inputTransforms.parseSizeData(arg, nbImages);
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        inputImages.parseFieldsSizeData(str, nbImages);
        inputTransforms.parseFieldsSizeData(str, nbImages);
        Inherit1::parseFields(str);
    }


protected:

    struct pttype  // to handle overlaps, we need to record some values and positions for each image
    {
        helper::vector<helper::vector<double> > vals;
        Coord u;
    };

    virtual void update()
    {
        unsigned int nb = nbImages.getValue();
        inputImages.resize(nb);
        inputTransforms.resize(nb);
        if(!nb) return;

        defaulttype::Vec<2,Coord> BB = this->getBB(0);//bounding box of the output image
        Coord minScale;
        for (unsigned int j = 0 ; j < this->getScale(0).size(); j++)
            minScale[j] = fabs(this->getScale(0)[j]);
        for(unsigned int j=1; j<nb; j++)
        {
            defaulttype::Vec<2,Coord> bb = this->getBB(j);
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

        waImage out(this->image);
        out->clear();
        out->setDimensions(dim);

        unsigned int overlp = this->overlap.getValue().getSelectedId();

        cimg_library::CImgList<T>& img = out->getCImgList();


#ifdef _OPENMP
        #pragma omp parallel for
#endif
        cimg_forXYZ(img(0),x,y,z) //space
        {
            for(unsigned int t=0; t<dim[4]; t++) for(unsigned int k=0; k<dim[3]; k++) img(t)(x,y,z,k) = (T)0;

            Coord p = outT->fromImage(Coord(x,y,z)); //coordinate of voxel (x,y,z) in world space
            helper::vector<struct pttype> pts;
            for(unsigned int j=0; j<nb; j++) // store values at p from input images
            {
                raImage in(this->inputImages[j]);
                const cimg_library::CImgList<T>& inImg = in->getCImgList();
                const imCoord indim=in->getDimensions();

                raTransform inT(this->inputTransforms[j]);
                Coord inp=inT->toImage(p); //corresponding voxel in image j
                if(inp[0]>=0 && inp[1]>=0 && inp[2]>=0 && inp[0]<=indim[0]-1 && inp[1]<=indim[1]-1 && inp[2]<=indim[2]-1)
                {
                    struct pttype pt;
                    if(Interpolation.getValue().getSelectedId()==INTERPOLATION_NEAREST)
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            pt.vals.push_back(helper::vector<double>());
                            for(unsigned int k=0; k<indim[3] && k<dim[3]; k++) // channels
                                pt.vals[t].push_back((double)inImg(t).atXYZ(sofa::helper::round((double)inp[0]),sofa::helper::round((double)inp[1]),sofa::helper::round((double)inp[2]),k));
                        }
                    else if(Interpolation.getValue().getSelectedId()==INTERPOLATION_LINEAR)
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            pt.vals.push_back(helper::vector<double>());
                            for(unsigned int k=0; k<indim[3] && k<dim[3]; k++) // channels
                                pt.vals[t].push_back((double)inImg(t).linear_atXYZ(inp[0],inp[1],inp[2],k));
                        }
                    else
                        for(unsigned int t=0; t<indim[4] && t<dim[4]; t++) // time
                        {
                            pt.vals.push_back(helper::vector<double>());
                            for(unsigned int k=0; k<indim[3] && k<dim[3]; k++) // channels
                                pt.vals[t].push_back((double)inImg(t).cubic_atXYZ(inp[0],inp[1],inp[2],k));

                        }
                    pt.u=Coord( ( inp[0]< indim[0]-inp[0]-1)? inp[0]: indim[0]-inp[0]-1 ,
                            ( inp[1]< indim[1]-inp[1]-1)? inp[1]: indim[1]-inp[1]-1 ,
                            ( inp[2]< indim[2]-inp[2]-1)? inp[2]: indim[2]-inp[2]-1 ); // distance from border

                    bool isnotnull=false;
                    for(unsigned int t=0; t<pt.vals.size(); t++) for(unsigned int k=0; k<pt.vals[t].size(); k++) if(pt.vals[t][k]!=(T)0) isnotnull=true;
                    if(isnotnull) pts.push_back(pt);

                }
            }
            unsigned int nbp=pts.size();
            if(nbp==0) continue;
            else if(nbp==1) {                
                    for(unsigned int t=0; t<pts[0].vals.size(); t++) for(unsigned int k=0; k<pts[0].vals[t].size(); k++) if((T)pts[0].vals[t][k]!=(T)0) img(t)(x,y,z,k) = (T)pts[0].vals[t][k];
            }
            else if(nbp>1)
            {                
                unsigned int nbt=pts[0].vals.size();
                unsigned int nbc=pts[0].vals[0].size();
                if(overlp==AVERAGE)
                {
                    for(unsigned int j=1; j<nbp; j++) for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) pts[0].vals[t][k] += pts[j].vals[t][k];
                    for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) img(t)(x,y,z,k) = (T)(pts[0].vals[t][k]/(double)nbp);
                }
                else if(overlp==ORDER)
                {
                    for(int j=nbp-1; j>=0; j--) for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) if((T)pts[j].vals[t][k]!=(T)0) img(t)(x,y,z,k) = (T)pts[j].vals[t][k];
                }
                else if(overlp==ALPHABLEND)
                {
                   unsigned int dir=0; if(pts[1].u[1]!=pts[0].u[1]) dir=1; if(pts[1].u[2]!=pts[0].u[2]) dir=2; // blending direction = direction where distance to border is different
                   double count=pts[0].u[dir]; for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) pts[0].vals[t][k]*=pts[0].u[dir];
                   for(unsigned int j=1; j<nbp; j++) { count+=pts[j].u[dir]; for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) pts[0].vals[t][k] += pts[j].vals[t][k]*pts[j].u[dir]; }
                   for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) img(t)(x,y,z,k) = (T)(pts[0].vals[t][k]/count);
                }
                else if(overlp==SEPARATE)
                {
                    for(unsigned int j=1; j<nbp; j++) if(pts[j].u[0]>pts[0].u[0] || pts[j].u[1]>pts[0].u[1] || pts[j].u[2]>pts[0].u[2]) { pts[0].u= pts[j].u; for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) pts[0].vals[t][k] = pts[j].vals[t][k]; }
                    for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) img(t)(x,y,z,k) = (T)pts[0].vals[t][k];
                }
                else if(overlp==ADDITIVE)
                {
                    for(unsigned int j=1; j<nbp; j++) for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) pts[0].vals[t][k] += pts[j].vals[t][k];
                    for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) img(t)(x,y,z,k) = (T)(pts[0].vals[t][k]);
                }
                else if(overlp==INTERSECT)
                {
                    for(unsigned int j=1; j<nbp; j++) for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) if (pts[0].vals[t][k] && pts[j].vals[t][k]) pts[0].vals[t][k] = (T)0.0;
                    for(unsigned int t=0; t<nbt; t++) for(unsigned int k=0; k<nbc; k++) img(t)(x,y,z,k) = (T)(pts[0].vals[t][k]);
                }

            }
        }

        sout << "Created merged image from " << nb << " input images." << sendl;
        cleanDirty();
    }

    defaulttype::Vec<2,Coord> getBB(unsigned int i) // get image corners
    {
        defaulttype::Vec<2,Coord> BB;
        raImage rimage(this->inputImages[i]);
        raTransform rtransform(this->inputTransforms[i]);

        const imCoord dim= rimage->getDimensions();
        defaulttype::Vec<8,Coord> p;
        p[0]=defaulttype::Vector3(0,0,0);
        p[1]=defaulttype::Vector3(dim[0]-1,0,0);
        p[2]=defaulttype::Vector3(0,dim[1]-1,0);
        p[3]=defaulttype::Vector3(dim[0]-1,dim[1]-1,0);
        p[4]=defaulttype::Vector3(0,0,dim[2]-1);
        p[5]=defaulttype::Vector3(dim[0]-1,0,dim[2]-1);
        p[6]=defaulttype::Vector3(0,dim[1]-1,dim[2]-1);
        p[7]=defaulttype::Vector3(dim[0]-1,dim[1]-1,dim[2]-1);

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

#endif // SOFA_IMAGE_MergeImages_H
