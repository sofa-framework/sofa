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
#ifndef SOFA_IMAGE_ImageDataDisplay_H
#define SOFA_IMAGE_ImageDataDisplay_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/SVector.h>


namespace sofa
{

namespace component
{

namespace engine
{


/**
 * This class Store custom data in an image
 */


template <class _InImageTypes,class _OutImageTypes>
class ImageDataDisplay : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE2(ImageDataDisplay,_InImageTypes,_OutImageTypes),Inherited);

    typedef _InImageTypes InImageTypes;
    typedef typename InImageTypes::T Ti;
    typedef typename InImageTypes::imCoord imCoordi;
    typedef helper::ReadAccessor<Data< InImageTypes > > raImagei;

    typedef _OutImageTypes OutImageTypes;
    typedef typename OutImageTypes::T To;
    typedef typename OutImageTypes::imCoord imCoordo;
    typedef helper::WriteOnlyAccessor<Data< OutImageTypes > > waImageo;

    Data< InImageTypes > inputImage;
    Data< OutImageTypes > outputImage;
    Data<helper::SVector<helper::SVector<To> > > VoxelData; ///< Data associed to each non null input voxel

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const ImageDataDisplay<InImageTypes,OutImageTypes>* = NULL) { return InImageTypes::Name()+std::string(",")+OutImageTypes::Name(); }

    ImageDataDisplay()    :   Inherited()
      , inputImage(initData(&inputImage,InImageTypes(),"inputImage",""))
      , outputImage(initData(&outputImage,OutImageTypes(),"outputImage",""))
      , VoxelData(initData(&VoxelData,"VoxelData","Data associed to each non null input voxel"))
    {
        inputImage.setReadOnly(true);
        outputImage.setReadOnly(true);
    }

    virtual ~ImageDataDisplay() {}

    virtual void init() override
    {
        addInput(&VoxelData);
        addInput(&inputImage);
        addOutput(&outputImage);
        setDirtyValue();
    }

    virtual void reinit() override { update(); }

protected:

    virtual void update() override
    {
        const helper::SVector<helper::SVector<To> >& dat = this->VoxelData.getValue();
        raImagei in(this->inputImage);

        cleanDirty();

        waImageo out(this->outputImage);
        imCoordi dim = in->getDimensions();
        dim[InImageTypes::DIMENSION_T] = 1;
        dim[InImageTypes::DIMENSION_S] = dat.size()?dat[0].size():1;
        out->setDimensions(dim);

        const cimg_library::CImg<Ti>& inImg=in->getCImg();
        cimg_library::CImg<To>& outImg=out->getCImg();
        outImg.fill(0);

        unsigned int count=0;
        cimg_forXYZ(outImg,x,y,z)
                if(inImg(x,y,z))
                if(count<dat.size())
        {
            cimg_forC(outImg,c) outImg(x,y,z,c)=dat[count][c];
            count++;
        }
    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_ImageDataDisplay_H
