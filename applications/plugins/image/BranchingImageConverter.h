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
#ifndef SOFA_IMAGE_BRANCHINGIMAGECONVERTER_H
#define SOFA_IMAGE_BRANCHINGIMAGECONVERTER_H

#include "initImage.h"
#include "ImageTypes.h"
#include "BranchingImage.h"
#include <sofa/core/DataEngine.h>


namespace sofa
{

namespace component
{

namespace engine
{


/// convert a flat image to a branching image
template <class _T>
class ImageToBranchingImageEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageToBranchingImageEngine,_T),Inherited);

    typedef _T T;
    typedef defaulttype::Image<T> ImageTypes;
    typedef defaulttype::BranchingImage<T> BranchingImageTypes;

    Data<ImageTypes> inputImage;
    Data<BranchingImageTypes> branchingImage;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageToBranchingImageEngine<T>* = NULL) { return ImageTypes::Name()+std::string(",")+BranchingImageTypes::Name(); }

    ImageToBranchingImageEngine()    :   Inherited()
        , inputImage(initData(&inputImage,ImageTypes(),"inputImage","Image"))
        , branchingImage(initData(&branchingImage,BranchingImageTypes(),"branchingImage","BranchingImage"))
    {
        branchingImage.setReadOnly(true);
        this->addAlias(&inputImage, "image");
        this->addAlias(&branchingImage, "outputBranchingImage");
    }

    virtual ~ImageToBranchingImageEngine()
    {
    }

    virtual void init()
    {
        addInput(&inputImage);
        addOutput(&branchingImage);
        setDirtyValue();
    }

    virtual void reinit()
    {
        update();
    }


protected:

    virtual void update()
    {
        cleanDirty();

//        std::cerr<<"ImageToBranchingImageEngine::update "<<inputImage.getValue().getDimensions()<<std::endl;

        BranchingImageTypes &si = *branchingImage.beginEdit();
        si = inputImage.getValue(); // operator= is performing the conversion

        if( f_printLog.getValue() ) std::cerr<<"ImageToBranchingImageEngine::update - conversion finished ("<<inputImage.getValue().approximativeSizeInBytes()<<" Bytes -> "<<si.approximativeSizeInBytes()<<" Bytes -> x"<<si.approximativeSizeInBytes()/(float)inputImage.getValue().approximativeSizeInBytes()<<")\n";

        branchingImage.endEdit();
    }

};





/// convert a branching image to a flat image
template <class _T>
class BranchingImageToImageEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(BranchingImageToImageEngine,_T),Inherited);

    typedef _T T;
    typedef defaulttype::Image<T> ImageTypes;
    typedef defaulttype::BranchingImage<T> BranchingImageTypes;

    Data<ImageTypes> image;
    Data<BranchingImageTypes> inputBranchingImage;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const BranchingImageToImageEngine<T>* = NULL) { return BranchingImageTypes::Name()+std::string(",")+ImageTypes::Name(); }

    BranchingImageToImageEngine()    :   Inherited()
        , image(initData(&image,ImageTypes(),"image","output Image"))
        , inputBranchingImage(initData(&inputBranchingImage,BranchingImageTypes(),"inputBranchingImage","input BranchingImage"))
    {
        inputBranchingImage.setReadOnly(true);
        this->addAlias(&inputBranchingImage, "branchingImage");
        this->addAlias(&image, "outputImage");
    }

    virtual ~BranchingImageToImageEngine()
    {
    }

    virtual void init()
    {
        addInput(&inputBranchingImage);
        addOutput(&image);
        setDirtyValue();
    }

    virtual void reinit()
    {
        update();
    }


protected:

    virtual void update()
    {
        cleanDirty();

        ImageTypes &img = *image.beginEdit();
        inputBranchingImage.getValue().toImage( img );

        if( f_printLog.getValue() ) std::cerr<<"BranchingImageToImageEngine::update - conversion finished ("<<inputBranchingImage.getValue().approximativeSizeInBytes()<<" Bytes -> "<<img.approximativeSizeInBytes()<<" Bytes -> x"<<img.approximativeSizeInBytes()/(float)inputBranchingImage.getValue().approximativeSizeInBytes()<<")\n";

        image.endEdit();
    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_BRANCHINGIMAGECONVERTER_H
