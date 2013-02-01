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


/// coarsening a flat Image in a branching image
/// the coarse size will be equal to inputsize/2^nbLevels (each level subdivides the size x/y/z by 2)
/// Nesme, Kry, Jeřábková, Faure, "Preserving Topology and Elasticity for Embedded Deformable Models", Siggraph09
template <class T>
class BranchingImageCoarsener : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(BranchingImageCoarsener,T),Inherited);

    typedef defaulttype::Image<T> ImageTypes;
    typedef defaulttype::BranchingImage<T> BranchingImageTypes;

    Data<ImageTypes> inputImage;
    Data<BranchingImageTypes> outputBranchingImage;
    Data<unsigned> nbLevels;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const BranchingImageCoarsener<T>* = NULL) { return ImageTypes::Name()+std::string(",")+BranchingImageTypes::Name(); }

    BranchingImageCoarsener()    :   Inherited()
        , inputImage(initData(&inputImage,ImageTypes(),"inputImage","Image to coarsen"))
        , outputBranchingImage(initData(&outputBranchingImage,BranchingImageTypes(),"outputBranchingImage","Coarsened BranchingImage"))
        , nbLevels(initData(&nbLevels,(unsigned)1,"nbLevels","How many coarsenings (subdividing x/y/z by 2)?"))
    {
        inputImage.setReadOnly(true);
        outputBranchingImage.setReadOnly(true);
        this->addAlias(&inputImage, "image");
        this->addAlias(&outputBranchingImage, "branchingImage");
    }

    virtual ~BranchingImageCoarsener()
    {
    }

    virtual void init()
    {
        addInput(&inputImage);
        addOutput(&outputBranchingImage);
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

        unsigned nb = pow( 2, nbLevels.getValue() ); // nb fine voxels in a coarse voxel in each direction x/y/z

        const typename ImageTypes::imCoord inputDimension = input.getDimensions();
        typename BranchingImageTypes::Dimension outputDimension = inputDimension;
        outputDimension[BranchingImageTypes::DIMENSION_X] /= nb;
        outputDimension[BranchingImageTypes::DIMENSION_Y] /= nb;
        outputDimension[BranchingImageTypes::DIMENSION_Z] /= nb;

        output.setDimension( outputDimension );

        for( unsigned t=0 ; t<outputDimension[BranchingImageTypes::DIMENSION_T] ; ++t )
        {
            const typename ImageTypes::CImgT& input_t = input.getCImg(t);
            typename BranchingImageTypes::BranchingImage3D& output_t = output.imgList[t];

            // a global labeling image, where labels have sens only in small blocks corresponding to the coarse voxels
            cimg_library::CImg<unsigned long> labelImage;
            labelImage.resize( input_t.width(), input_t.height(), input_t.depth(), 1  );

            unsigned index1d = 0;
            for( unsigned z=0 ; z<outputDimension[BranchingImageTypes::DIMENSION_Z] ; ++z )
            for( unsigned y=0 ; y<outputDimension[BranchingImageTypes::DIMENSION_Y] ; ++y )
            for( unsigned x=0 ; x<outputDimension[BranchingImageTypes::DIMENSION_X] ; ++x )
            {
                unsigned finex = x*nb, finexend = std::min( (x+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_X] ) - 1;
                unsigned finey = y*nb, fineyend = std::min( (y+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Y] ) - 1;
                unsigned finez = z*nb, finezend = std::min( (z+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Z] ) - 1;

                typename ImageTypes::CImgT subImage = input_t.get_crop( finex, finey, finez, finexend, fineyend, finezend );

                // convert the subimage to a bool image with only 1 canal
                cimg_library::CImg<bool> subBinaryImage = subImage.resize( subImage.width(), subImage.height(), subImage.depth(), 1, 3 );

                // Connected components labeling
                cimg_library::CImg<unsigned long> subLabelImage = subBinaryImage.get_label( false );

                // pour chaque composante connexe, creer un voxel grossier superposé
                output_t[index1d].resize( subLabelImage.max() );

                // copy the block (corresponding to the coarse voxel) in the global label image
                labelImage.crop( finex, finey, finez, finexend, fineyend, finezend ) = subLabelImage;

                ++index1d;
            }

            index1d = 0;
            for( unsigned z=0 ; z<outputDimension[BranchingImageTypes::DIMENSION_Z] ; ++z )
            for( unsigned y=0 ; y<outputDimension[BranchingImageTypes::DIMENSION_Y] ; ++y )
            for( unsigned x=0 ; x<outputDimension[BranchingImageTypes::DIMENSION_X] ; ++x )
            {
                unsigned finex = x*nb, finexend = std::min( (x+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_X] ) - 1;
                unsigned finey = y*nb, fineyend = std::min( (y+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Y] ) - 1;
                unsigned finez = z*nb, finezend = std::min( (z+1)*nb, inputDimension[BranchingImageTypes::DIMENSION_Z] ) - 1;

                // trouver connectivité
                //      pour tous voxels fins
                for( unsigned fz=finez ; fz<=finezend ; ++fz )
                for( unsigned fy=finey ; fy<=fineyend ; ++fy )
                for( unsigned fx=finex ; fx<=finexend ; ++fx )
                {
                    unsigned coarseOffset = labelImage(fx,fy,fz); // position of coarse voxel including the fine voxel(fx,fy,fz) in the superimposed voxels

                    if( coarseOffset ) // not empty
                    {
                    //          foreach n=voisins(vf) (ds input, l'image plate entière)
                        if( fx>0 )
                        {
                             unsigned neighbourCoarseOffset = labelImage(fx-1,fy,fz);
                             if( neighbourCoarseOffset ) // neighbour is not empty
                             {
                                 unsigned neighbourCoarseIndex = output.getNeighbourIndex( BranchingImageTypes::LEFT, index1d );
                                 if( neighbourCoarseIndex != index1d ) // both are not in the same coarse voxel
                                 {
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
                                 unsigned neighbourCoarseIndex = output.getNeighbourIndex( BranchingImageTypes::BOTTOM, index1d );
                                 if( neighbourCoarseIndex != index1d ) // both are not in the same coarse voxel
                                 {
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
                                 unsigned neighbourCoarseIndex = output.getNeighbourIndex( BranchingImageTypes::BACK, index1d );
                                 if( neighbourCoarseIndex != index1d ) // both are not in the same coarse voxel
                                 {
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



        if( f_printLog.getValue() ) std::cerr<<"BranchingImageCoarsener::update - coarsening finished\n";

        outputBranchingImage.endEdit();
    }

};





} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_BRANCHINGIMAGECONVERTER_H
