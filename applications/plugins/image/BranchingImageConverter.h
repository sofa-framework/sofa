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



/// convert a flat image to a branching image with an optional coarsening
/// no coarsening (coarseningLevels==0) returns a branching image without any ramification
/// only coarsening can create ramifications
/// the coarse size will be equal to inputsize/2^coarseningLevels (each level subdivides the size x/y/z by 2)
/// This implementation improves the algorithm presented in: Nesme, Kry, Jeřábková, Faure, "Preserving Topology and Elasticity for Embedded Deformable Models", Siggraph09
/// because the coarse configuration is directly computed from the fine resolution.
template <class T>
class ImageToBranchingImageConverter : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageToBranchingImageConverter,T),Inherited);


    typedef SReal Real;

    typedef defaulttype::Image<T> ImageTypes;
    typedef defaulttype::BranchingImage<T> BranchingImageTypes;
    typedef defaulttype::ImageLPTransform<Real> TransformType;



    Data<ImageTypes> inputImage;
    Data<BranchingImageTypes> outputBranchingImage;
    Data<unsigned> coarseningLevels;
    Data<unsigned> superimpositionType;

    Data<TransformType> inputTransform, outputTransform;


    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName(const ImageToBranchingImageConverter<T>* = NULL) { return ImageTypes::Name()+std::string(",")+BranchingImageTypes::Name(); }

    ImageToBranchingImageConverter()    :   Inherited()
        , inputImage(initData(&inputImage,ImageTypes(),"inputImage","Image to coarsen"))
        , outputBranchingImage(initData(&outputBranchingImage,BranchingImageTypes(),"outputBranchingImage","Coarsened BranchingImage"))
        , coarseningLevels(initData(&coarseningLevels,(unsigned)0,"coarseningLevels","How many coarsenings (subdividing x/y/z by 2)?"))
        , superimpositionType(initData(&superimpositionType,(unsigned)0,"superimpositionType","Which value for superimposed voxels? (0->copy (default), 1->ratio on volume, 2->divided value on superimposed number)"))
        , inputTransform(initData(&inputTransform,TransformType(),"inputTransform","Input Transform"))
        , outputTransform(initData(&outputTransform,TransformType(),"outputTransform","Output Transform"))
    {
        inputImage.setReadOnly(true);
        outputBranchingImage.setReadOnly(true);
        this->addAlias(&inputImage, "image");
        this->addAlias(&outputBranchingImage, "branchingImage");
        this->addAlias(&outputTransform, "transform");
    }

    virtual ~ImageToBranchingImageConverter()
    {
    }

    virtual void init()
    {
        addInput(&inputImage);
        addInput(&coarseningLevels);
        addInput(&superimpositionType);
        addInput(&inputTransform);
        addOutput(&outputBranchingImage);
        addOutput(&outputTransform);
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

        BranchingImageTypes &output = *outputBranchingImage.beginEdit();
        const ImageTypes &input = inputImage.getValue();
        TransformType& transform = *outputTransform.beginEdit();

        transform = inputTransform.getValue();

        if( coarseningLevels.getValue()<1 )
        {
            output = input; // operator= is performing the conversion without modifying the resolution, so with flat neighbourood
        }
        else
        {
            unsigned nb = pow( 2, coarseningLevels.getValue() ); // nb fine voxels in a coarse voxel in each direction x/y/z

            TransformType::Coord& scale = transform.getScale();
            scale *= nb;

            const typename ImageTypes::imCoord inputDimension = input.getDimensions();
            typename BranchingImageTypes::Dimension outputDimension = inputDimension;
            outputDimension[BranchingImageTypes::DIMENSION_X] = ceil( outputDimension[BranchingImageTypes::DIMENSION_X]/(float)nb );
            outputDimension[BranchingImageTypes::DIMENSION_Y] = ceil( outputDimension[BranchingImageTypes::DIMENSION_Y]/(float)nb );
            outputDimension[BranchingImageTypes::DIMENSION_Z] = ceil( outputDimension[BranchingImageTypes::DIMENSION_Z]/(float)nb );

            output.setDimension( outputDimension );

            for( unsigned t=0 ; t<outputDimension[BranchingImageTypes::DIMENSION_T] ; ++t )
            {
                const typename ImageTypes::CImgT& input_t = input.getCImg(t);
                typename BranchingImageTypes::BranchingImage3D& output_t = output.imgList[t];

                // a global labeling image, where labels have sens only in small blocks corresponding to the coarse voxels
                cimg_library::CImg<unsigned long> labelImage;
                labelImage.resize( input_t.width(), input_t.height(), input_t.depth(), 1  );


                // FIND SUPERIMPOSED VOXELS

                unsigned index1d = 0;
                for( unsigned z=0 ; z<outputDimension[BranchingImageTypes::DIMENSION_Z] ; ++z )
                for( unsigned y=0 ; y<outputDimension[BranchingImageTypes::DIMENSION_Y] ; ++y )
                for( unsigned x=0 ; x<outputDimension[BranchingImageTypes::DIMENSION_X] ; ++x )
                {
                    unsigned finex = x*nb, finexend = std::min( (x+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_X] ) - 1;
                    unsigned finey = y*nb, fineyend = std::min( (y+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Y] ) - 1;
                    unsigned finez = z*nb, finezend = std::min( (z+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Z] ) - 1;

                    // the sub-image storing all fine voxels included in the coarse voxel
                    typename ImageTypes::CImgT subImage = input_t.get_crop( finex, finey, finez, finexend, fineyend, finezend );

                    // convert the subimage to a bool image with only 1 canal
                    cimg_library::CImg<bool> subBinaryImage = subImage.get_resize( subImage.width(), subImage.height(), subImage.depth(), 1, 3 );

                    // connected component labeling
                    cimg_library::CImg<unsigned long> subLabelImage = subBinaryImage.get_label( false );

                    unsigned nbLabels = subLabelImage.max();

                    // when all voxels are filled, there is only one label==0, replace it by 1
                    if( !nbLabels && subBinaryImage(0,0,0) )
                    {
                        nbLabels=1;
                        subLabelImage.fill(1);
                    }
                    else // some are empty, some are not -> enforce the label 0 to empty voxels
                    {
                        unsigned label0;
                        cimg_foroff( subLabelImage, off ) if( !subBinaryImage(off) ) { label0=subLabelImage(off); break; }
                        cimg_foroff( subLabelImage, off ) if( !subBinaryImage(off) ) subLabelImage(off)=0; else if( !subLabelImage(off) ) subLabelImage(off)=label0;
                    }

                    // a superimposed voxel per independant component
                    output_t[index1d].resize( nbLabels, outputDimension[BranchingImageTypes::DIMENSION_S] );

                    // TODO put the superimposed value based on superimpositionType

                    // copy the block (corresponding to the coarse voxel) in the global label image
                    cimg_library::copySubImage( labelImage, subLabelImage, finex, finey, finez );

                    ++index1d;
                }

                // FIND CONNECTIVITY

                index1d = 0;
                for( unsigned z=0 ; z<outputDimension[BranchingImageTypes::DIMENSION_Z] ; ++z )
                for( unsigned y=0 ; y<outputDimension[BranchingImageTypes::DIMENSION_Y] ; ++y )
                for( unsigned x=0 ; x<outputDimension[BranchingImageTypes::DIMENSION_X] ; ++x )
                {
                    unsigned finex = x*nb, finexend = std::min( (x+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_X] ) - 1;
                    unsigned finey = y*nb, fineyend = std::min( (y+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Y] ) - 1;
                    unsigned finez = z*nb, finezend = std::min( (z+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Z] ) - 1;

                    // for all fine voxels included in the coarse voxel
                    for( unsigned fz=finez ; fz<=finezend ; ++fz )
                    for( unsigned fy=finey ; fy<=fineyend ; ++fy )
                    for( unsigned fx=finex ; fx<=finexend ; ++fx )
                    {
                        unsigned coarseOffset = labelImage(fx,fy,fz); // position of coarse voxel including the fine voxel(fx,fy,fz) in the superimposed voxels

                        if( coarseOffset ) // not empty
                        {
                            coarseOffset -= 1; // the label 1 go to the offset/position 0 in the superimposed vector

                            // look at all fine neighbours of the fine voxel
                            if( fx>0 )
                            {
                                 unsigned neighbourCoarseOffset = labelImage(fx-1,fy,fz);
                                 if( neighbourCoarseOffset ) // neighbour is not empty
                                 {
                                     neighbourCoarseOffset -= 1;
                                     if( fx-1 < finex ) // both are not in the same coarse voxel
                                     {
                                         const unsigned neighbourCoarseIndex = output.getNeighbourIndex( BranchingImageTypes::LEFT, index1d );
                                        // connect index grossier de vf et de n
                                         output_t[index1d][coarseOffset].addNeighbour( BranchingImageTypes::LEFT, neighbourCoarseOffset );
                                         output_t[neighbourCoarseIndex][neighbourCoarseOffset].addNeighbour( BranchingImageTypes::RIGHT, coarseOffset );
                                     }
                                 }
                            }
                            if( fy>0 )
                            {
                                 unsigned neighbourCoarseOffset = labelImage(fx,fy-1,fz);
                                 if( neighbourCoarseOffset ) // neighbour is not empty
                                 {
                                     neighbourCoarseOffset -= 1;
                                     if( fy-1 < finey ) // both are not in the same coarse voxel
                                     {
                                         const unsigned neighbourCoarseIndex = output.getNeighbourIndex( BranchingImageTypes::BOTTOM, index1d );
                                        // connect index grossier de vf et de n
                                         output_t[index1d][coarseOffset].addNeighbour( BranchingImageTypes::BOTTOM, neighbourCoarseOffset );
                                         output_t[neighbourCoarseIndex][neighbourCoarseOffset].addNeighbour( BranchingImageTypes::TOP, coarseOffset );
                                     }
                                 }
                            }
                            if( fz>0 )
                            {
                                 unsigned neighbourCoarseOffset = labelImage(fx,fy,fz-1);
                                 if( neighbourCoarseOffset ) // neighbour is not empty
                                 {
                                     neighbourCoarseOffset -= 1;
                                     if( fz-1 < finez ) // both are not in the same coarse voxel
                                     {
                                         const unsigned neighbourCoarseIndex = output.getNeighbourIndex( BranchingImageTypes::BACK, index1d );
                                        // connect index grossier de vf et de n
                                         output_t[index1d][coarseOffset].addNeighbour( BranchingImageTypes::BACK, neighbourCoarseOffset );
                                         output_t[neighbourCoarseIndex][neighbourCoarseOffset].addNeighbour( BranchingImageTypes::FRONT, coarseOffset );
                                     }
                                 }
                            }
                        }
                    }
                    ++index1d;
                }
            } // for t
        }

        outputTransform.endEdit();
        outputBranchingImage.endEdit();


        if( f_printLog.getValue() ) std::cerr<<"BranchingImageCoarsener::update - coarsening finished\n";
    }

};




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





/// convert a branching image to a flat image
template <class _T>
class BranchingImageToImageConverter : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(BranchingImageToImageConverter,_T),Inherited);

    typedef _T T;
    typedef defaulttype::Image<T> ImageTypes;
    typedef defaulttype::BranchingImage<T> BranchingImageTypes;

    Data<ImageTypes> image;
    Data<BranchingImageTypes> inputBranchingImage;
    Data<unsigned> conversionType;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const BranchingImageToImageConverter<T>* = NULL) { return BranchingImageTypes::Name()+std::string(",")+ImageTypes::Name(); }

    BranchingImageToImageConverter()    :   Inherited()
        , image(initData(&image,ImageTypes(),"image","output Image"))
        , inputBranchingImage(initData(&inputBranchingImage,BranchingImageTypes(),"inputBranchingImage","input BranchingImage"))
        , conversionType(initData(&conversionType,(unsigned)2,"conversionType","0->first voxel, 1->mean, 2->nb superimposed voxels (default)"))
    {
        inputBranchingImage.setReadOnly(true);
        this->addAlias(&inputBranchingImage, "branchingImage");
        this->addAlias(&image, "outputImage");
    }

    virtual ~BranchingImageToImageConverter()
    {
    }

    virtual void init()
    {
        addInput(&inputBranchingImage);
        addInput(&conversionType);
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
        inputBranchingImage.getValue().toImage( img, conversionType.getValue() );

        if( f_printLog.getValue() ) std::cerr<<"BranchingImageToImageEngine::update - conversion finished ("<<inputBranchingImage.getValue().approximativeSizeInBytes()<<" Bytes -> "<<img.approximativeSizeInBytes()<<" Bytes -> x"<<img.approximativeSizeInBytes()/(float)inputBranchingImage.getValue().approximativeSizeInBytes()<<")\n";

        image.endEdit();
    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_BRANCHINGIMAGECONVERTER_H
