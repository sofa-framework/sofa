/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/Data.h>
#include <SceneCreator/SceneCreator.h>

#include <SofaTest/Sofa_test.h>
#include <image/ImageContainer.h>

namespace sofa {

/**  Test suite for data image link.
Create two data image and a link between them.
The first data image is the data of an image container. Check that the second data image is the same image.
Then compare data pointers to see if data link duplicates the datas.
  */
struct DataImageLink_test : public Sofa_test<>
{
    typedef defaulttype::Image<unsigned char> Image;
    // Image Container
    typedef sofa::component::container::ImageContainer< defaulttype::Image<unsigned char> > ImageContainer;
    ImageContainer::SPtr imageContainer;
    ImageContainer::SPtr imageContainer2;

    // 2 data images
    core::objectmodel::Data< defaulttype::Image<unsigned char> > data1;
    core::objectmodel::Data< defaulttype::Image<unsigned char> > data2;

    /// Create a link between the two images
    void SetUp()
    { 
    }

    // Test link
    void testImageDataLink()
    {
        // Image container
        imageContainer = sofa::core::objectmodel::New<ImageContainer>();

        // Set path to image for imageContainer
        std::string fileName = std::string(IMAGETEST_SCENES_DIR) + "/" + "beam.raw";
        imageContainer->m_filename.setValue(fileName);

        // Init image container
        imageContainer->init();

        // Set data1 = image in imageContainer
        data1.setValue(imageContainer->image.getValue());

        // Set data link
        sofa::modeling::setDataLink(&data1,&data2);


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

        // Check that data values are the same
        ASSERT_EQ(data1.getValue(),data2.getValue());
    }

    void loadImage()
    {
        ImageContainer::SPtr ic = sofa::core::objectmodel::New<ImageContainer>();
        // Set path to image for imageContainer
        std::string fileName = std::string(IMAGETEST_SCENES_DIR) + "/" + "beam.raw";
        ic->m_filename.setValue(fileName);

        // Init image container
        ic->init();

        sofa::modeling::setDataLink(&ic->image,&data1);
        ASSERT_NE(data1.getValue(),data2.getValue());
    }

    void createAndDeleteLinkedData()
    {

        loadImage();
        // Check that data values are the same
        ASSERT_NE(data1.getValue(),data2.getValue());

    }

};

// Test
TEST_F(DataImageLink_test , createAndDeleteLinkedData )
{
    ASSERT_NO_THROW( this->createAndDeleteLinkedData() );
}
TEST_F(DataImageLink_test , testImageDataLink )
{
    this->testImageDataLink();
}

}// namespace sofa


