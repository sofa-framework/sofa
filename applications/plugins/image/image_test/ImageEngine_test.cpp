/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/Data.h>
#include <plugins/SceneCreator/SceneCreator.h>
//Including Simulation
#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <plugins/SofaTest/Sofa_test.h>
#include <plugins/image/ImageContainer.h>
#include <plugins/image/ImageViewer.h>
#include <plugins/image/TestImageEngine.h>

namespace sofa {

/**  Test suite for engine data image.
 * Create a simple scene with an engine which computes an output image from an input image at each time step.
 * Visualize the output image of the engine with ImageViewer.
 * The input image of ImageViewer is then linked to the ouput image of the engine.
 * Copy on Write option is true.
 * When we launch the simulation at the end during destruction there is a segmentation fault,
 * because the buffer data of the ouput image of the Engine is shared with the buffer of the image of image viewer.
 * Note: the function draw of ImageViewer is acutally not called in this test (it works with the gui).
  */
struct ImageEngine_test : public Sofa_test<>
{
    // Image Container
    typedef sofa::component::container::ImageContainer< defaulttype::Image<unsigned char> > ImageContainer;
    ImageContainer::SPtr imageContainer;

    // Image Engine
    typedef sofa::component::engine::TestImageEngine< defaulttype::Image<unsigned char> > TestImageEngine;
    TestImageEngine::SPtr imageEngine;

    // Image Viewer
    typedef sofa::component::misc::ImageViewer< defaulttype::Image<unsigned char> > ImageViewer;
    ImageViewer::SPtr imageViewer;

    // Root of the scene graph
    simulation::Node::SPtr root;

    // simulation
    simulation::Simulation* simulation;

    /// Create the scene to test
    void SetUp()
    { 
        // Create a scene
        sofa::component::init();
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

    }

    // Test animate
    void testAnimate()
    {
        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());

        //  do several steps of animation
        for(int l=0;l<2;++l)
        {
            sofa::simulation::getSimulation()->animate(root.get(),0.5);
        }

    }

    // Unload scene
    void TearDown()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
    }

};

// Test
TEST_F(ImageEngine_test , testImageDataLink )
{
    ASSERT_NO_THROW(this->testAnimate());
}

}// namespace sofa


