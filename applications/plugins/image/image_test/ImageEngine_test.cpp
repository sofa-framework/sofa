/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/Data.h>
#include <SceneCreator/SceneCreator.h>
//Including Simulation
#include <sofa/simulation/graph/DAGSimulation.h>

#include <SofaTest/Sofa_test.h>
#include <SofaTest/DataEngine_test.h>
#include <image/ImageContainer.h>
#include <image/ImageViewer.h>
#include "TestImageEngine.h"

namespace sofa {

/**  Test suite for engine data image.
 * Create a simple scene with an engine which computes an output image from an input image at each time step.
 * Visualize the output image of the engine with ImageViewer.
 * The input image of ImageViewer is then linked to the ouput image of the engine.
 * Copy on Write option is true.
 * Note: the function draw of ImageViewer is acutally not called in this test (it works with the gui).
  */
struct ImageEngine_test : public Sofa_test<>
{

    // Root of the scene graph
    simulation::Node::SPtr root;


    // Unload scene
    void TearDown()
    {
//        std::cerr << "start TearDown" << std::endl;
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
//        std::cerr << "end TearDown" << std::endl;
    }




    // Test link
    void testDataLink()
    {

//#define __DISPLAY__

        typedef defaulttype::Image<unsigned char> Image;

        core::objectmodel::Data< Image > data1;
        core::objectmodel::Data< Image > data2;


        // Set data1
        Image::CImgT img( (std::string(IMAGETEST_SCENES_DIR) + "/lena.jpg").c_str() );

#ifdef __DISPLAY__
        img.display("loaded image");
#endif

        data1.setValue(img);

#ifdef __DISPLAY__
        data1.getValue().getCImg().display("data1");
#endif


        // Set data link
        sofa::modeling::setDataLink(&data1,&data2);
        data1.getValue();

#ifdef __DISPLAY__
        data2.getValue().getCImg().display("data2");
#endif


        // Check that data values are the same
        ASSERT_EQ(data1.getValue(),data2.getValue());

        // Check if pointers are equal
        if(&data1.getValue()!= &data2.getValue())
        {
            ADD_FAILURE() << "Data Link duplicates the datas ! " << std::endl;
        }

        // Change value of data1
        helper::WriteAccessor<Data< Image > > w1(data1);
        w1->getCImg(0).fill(0);

#ifdef __DISPLAY__
        data1.getValue().getCImg().display("data1 after clear");
#endif

        // Check that data values are the same
        ASSERT_EQ(data1.getValue(),data2.getValue());

#ifdef __DISPLAY__
        data2.getValue().getCImg().display("data2 after clear");
#endif

        // Check if pointers are still equal
        if(&data1.getValue()!= &data2.getValue())
        {
            ADD_FAILURE() << "Data Link duplicates the datas ! " << std::endl;
        }





//        // Change value of data2
////        data2.getValue();
//        helper::WriteAccessor<Data< Image > > w2(data2);
//        w2->getCImg(0).fill(255,0,0);

//#ifdef __DISPLAY__
//        data2.getValue().getCImg().display("data2 after modif");
//        data1.getValue().getCImg().display("data1 after modif");
//#endif

//        std::cerr<<&data1.getValue()<<" "<<&data2.getValue()<<" "<<(data1.getValue()==data2.getValue())<<std::endl;

//        w1->getCImg(0).load((std::string(IMAGETEST_SCENES_DIR) + "/lena.jpg").c_str());

//        std::cerr<<&data1.getValue()<<" "<<&data2.getValue()<<" "<<(data1.getValue()==data2.getValue())<<std::endl;


//#ifdef __DISPLAY__
//        data1.getValue().getCImg().display("data1 end");
//        data2.getValue().getCImg().display("data2 end");
//#endif

    }





    /// Scene with simple engines
    ///
    void testAnimate()
    {
        // simulation
        simulation::Simulation* simulation;

        // Image Container
        typedef sofa::component::container::ImageContainer< defaulttype::Image<unsigned char> > ImageContainer;
        ImageContainer::SPtr imageContainer;

        // Image Engine
        typedef sofa::component::engine::TestImageEngine< defaulttype::Image<unsigned char> > TestImageEngine;
        TestImageEngine::SPtr imageEngine;

        // Create a scene
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        // Root node
        root = simulation->createNewGraph("root");

        // Input image
        // Image container
        imageContainer = sofa::modeling::addNew<ImageContainer>(root);

        // Set path to image for imageContainer
        std::string fileName = std::string(IMAGETEST_SCENES_DIR) + "/" + "beam.raw";
        imageContainer->m_filename.setValue(fileName);

        // ImageEngine
        imageEngine = sofa::modeling::addNew<TestImageEngine>(root);

        // Set data link: image of ImageContainer is input image of Engine.
        sofa::modeling::setDataLink(&imageContainer->image,&imageEngine->inputImage);

        // ImageEngine listening is true to update image at each time step
        imageEngine->f_listening.setValue(true);

        TestImageEngine::SPtr imageEngine2 = sofa::modeling::addNew<TestImageEngine>(root);
        sofa::modeling::setDataLink(&imageEngine->outputImage,&imageEngine2->inputImage);




        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());

        //  do several steps of animation
        for(int l=0;l<2;++l)
        {
            sofa::simulation::getSimulation()->animate(root.get(),0.5);
//            std::cerr << "== loop "<<l<<std::endl;
//            std::cerr << "engine2 size " << imageEngine2->inputImage.getValue().getCImg(0).size() << std::endl;
//            std::cerr << "image shared shared: " << imageContainer->image.getValue().getCImg(0).is_shared() << " " << &imageContainer->image.getValue().getCImg(0)<< std::endl;
//            std::cerr << "engine input shared: " << imageEngine->inputImage.getValue().getCImg(0).is_shared() << " " << &imageEngine->inputImage.getValue().getCImg(0)<< std::endl;
//            std::cerr << "engine output shared: " << imageEngine->outputImage.getValue().getCImg(0).is_shared() << " " << &imageEngine->outputImage.getValue().getCImg(0)<< std::endl;
//            std::cerr << "engine2 input shared: " << imageEngine2->inputImage.getValue().getCImg(0).is_shared() << " " << &imageEngine2->inputImage.getValue().getCImg(0)<< std::endl;
        }


        // Check if pointers of images that should be shared are equal
        ASSERT_EQ(&imageContainer->image.getValue(),&imageEngine->inputImage.getValue());
        ASSERT_EQ(&imageEngine->outputImage.getValue(),&imageEngine2->inputImage.getValue());


    }



    /// Scene with an ImageViewer
    void testImageViewer()
    {
        // simulation
        simulation::Simulation* simulation;

        // Image Container
        typedef sofa::component::container::ImageContainer< defaulttype::Image<unsigned char> > ImageContainer;
        ImageContainer::SPtr imageContainer;

        // Image Engine
        typedef sofa::component::engine::TestImageEngine< defaulttype::Image<unsigned char> > TestImageEngine;
        TestImageEngine::SPtr imageEngine;

        // Image Viewer
        typedef sofa::component::misc::ImageViewer< defaulttype::Image<unsigned char> > ImageViewer;
        ImageViewer::SPtr imageViewer;

        // Create a scene
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        // Root node
        root = simulation->createNewGraph("root");

        // Input image
        // Image container
        imageContainer = sofa::modeling::addNew<ImageContainer>(root);

        // Set path to image for imageContainer
        std::string fileName = std::string(IMAGETEST_SCENES_DIR) + "/" + "beam.raw";
        imageContainer->m_filename.setValue(fileName);

        // ImageEngine
        imageEngine = sofa::modeling::addNew<TestImageEngine>(root);

        // Set data link: image of ImageContainer is input image of Engine.
        sofa::modeling::setDataLink(&imageContainer->image,&imageEngine->inputImage);

        // ImageEngine listening is true to update image at each time step
        imageEngine->f_listening.setValue(true);


        //ImageViewer
        imageViewer = sofa::modeling::addNew<ImageViewer>(root);

        // Set data link: output image of engine is image of ImageViewer.
        sofa::modeling::setDataLink(&imageEngine->outputImage,&imageViewer->image);
        //sofa::modeling::setDataLink(&imageContainer->image,&imageViewer->image);



        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());

        //  do several steps of animation
        for(int l=0;l<2;++l)
        {
            sofa::simulation::getSimulation()->animate(root.get(),0.5);
//            std::cerr << "== loop "<<l<<std::endl;
//            std::cerr << "viewer size " << imageViewer->image.getValue().getCImg(0).size() << std::endl;
//            std::cerr << "image shared shared: " << imageContainer->image.getValue().getCImg(0).is_shared() << " " << &imageContainer->image.getValue().getCImg(0)<< std::endl;
//            std::cerr << "engine input shared: " << imageEngine->inputImage.getValue().getCImg(0).is_shared() << " " << &imageEngine->inputImage.getValue().getCImg(0)<< std::endl;
//            std::cerr << "engine output shared: " << imageEngine->outputImage.getValue().getCImg(0).is_shared() << " " << &imageEngine->outputImage.getValue().getCImg(0)<< std::endl;
//            std::cerr << "viewer input shared: " << imageViewer->image.getValue().getCImg(0).is_shared() << " " << &imageViewer->image.getValue().getCImg(0)<< std::endl;
        }


        // Check if pointers of images that should be shared are equal
        ASSERT_EQ(&imageContainer->image.getValue(),&imageEngine->inputImage.getValue());
        ASSERT_EQ(&imageEngine->outputImage.getValue(),&imageViewer->image.getValue());

    }



};

// Test
TEST_F(ImageEngine_test , testDataLink )
{
    ASSERT_NO_THROW(this->testDataLink());
}

TEST_F(ImageEngine_test , testEngineDataLink )
{
    ASSERT_NO_THROW(this->testAnimate());
}

TEST_F(ImageEngine_test , testImageViewer )
{
    ASSERT_NO_THROW(this->testImageViewer());
}


}// namespace sofa



////////////////////////////////////

#include "../DepthMapToMeshEngine.h"
#include "../ImageAccumulator.h"
#include "../ImageDataDisplay.h"
#include "../ImageFilter.h"
#include "../ImageOperation.h"
#include "../ImageSampler.h"
#include "../ImageToRigidMassEngine.h"
#include "../ImageTransformEngine.h"
#include "../ImageTransform.h"
#include "../ImageValuesFromPositions.h"
#include "../MarchingCubesEngine.h"
#include "../MergeImages.h"
#include "../MeshToImageEngine.h"
#include "TestImageEngine.h"
#include "../TransferFunction.h"
#include "../VoronoiToMeshEngine.h"


namespace sofa {

/// a utility for ImageDataEngine test
/// allocating all engine's input Data<Image>
template <typename DataEngineType>
struct ImageDataEngine_test : public DataEngine_test<DataEngineType>
{
    typedef core::objectmodel::DDGNode DDGNode;
    typedef DDGNode::DDGLinkContainer DDGLinkContainer;

    virtual void init()
    {
        DataEngine_test<DataEngineType>::init();

        const DDGLinkContainer& parent_inputs = this->m_engineInput->DDGNode::getInputs();
        for( unsigned i=0, iend=parent_inputs.size() ; i<iend ; ++i )
        {
            core::objectmodel::BaseData* data = static_cast<core::objectmodel::BaseData*>(parent_inputs[i]);

            const defaulttype::AbstractTypeInfo *typeinfo = data->getValueTypeInfo();

            if( typeinfo->name().find("Image") != std::string::npos || typeinfo->name().find("BranchingImage") != std::string::npos )
            {
                defaulttype::BaseImage* img = static_cast<defaulttype::BaseImage*>( data->beginEditVoidPtr() );
//                std::cerr<<data->getName()<<" is a Data<Image>\n";
                // allocate input
                img->setDimensions( defaulttype::BaseImage::imCoord(2,2,2,1,1) );
                img->fill(1.0);
                data->endEditVoidPtr();
            }
        }
    }

};




// testing every engines of image plugin here

typedef testing::Types<
 /*TestDataEngine< component::engine::DepthMapToMeshEngine<defaulttype::ImageUC> > // crash on MAC (opengl related?)
,*/TestDataEngine< component::engine::ImageAccumulator<defaulttype::ImageUC> >
,TestDataEngine< component::engine::ImageDataDisplay<defaulttype::ImageUC,defaulttype::ImageUC> >
,TestDataEngine< component::engine::ImageFilter<defaulttype::ImageUC,defaulttype::ImageUC> >
,TestDataEngine< component::engine::ImageOperation<defaulttype::ImageUC> >
,TestDataEngine< component::engine::ImageSampler<defaulttype::ImageUC> > // ???
,TestDataEngine< component::engine::ImageToRigidMassEngine<defaulttype::ImageUC> >
,TestDataEngine< component::engine::ImageTransformEngine >
//,TestDataEngine< component::engine::ImageTransform<defaulttype::ImageUC> > // other components are required
,TestDataEngine< component::engine::ImageValuesFromPositions<defaulttype::ImageUC> >
,TestDataEngine< component::engine::MarchingCubesEngine<defaulttype::ImageUC> >
,TestDataEngine< component::engine::MergeImages<defaulttype::ImageUC> >
,TestDataEngine< component::engine::MeshToImageEngine<defaulttype::ImageUC> >
,TestDataEngine< component::engine::TestImageEngine<defaulttype::ImageUC> >
,TestDataEngine< component::engine::TransferFunction<defaulttype::ImageUC,defaulttype::ImageUC> >
,TestDataEngine< component::engine::VoronoiToMeshEngine<defaulttype::ImageUC> >
> TestTypes; // the types to instanciate.


//// ========= Tests to run for each instanciated type
TYPED_TEST_CASE( ImageDataEngine_test, TestTypes );

//// test number of call to DataEngine::update
TYPED_TEST( ImageDataEngine_test , basic_test )
{
    this->run_basic_test();
}

}// namespace sofa
