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
#include <sofa/helper/io/ImagePNG.h>
#include <sofa/helper/system/FileRepository.h>
#include <cstring>

#include <SofaTest/Sofa_test.h>

namespace sofa {

class ImagePNG_test : public Sofa_test<>
{
protected:

    void SetUp()
    {
        sofa::helper::system::DataRepository.addFirstPath(FRAMEWORK_TEST_RESOURCES_DIR);
    }
    void TearDown()
    {
        sofa::helper::system::DataRepository.removePath(FRAMEWORK_TEST_RESOURCES_DIR);
    }

    struct ImagePNGTestData
    {
        std::string filename;
        unsigned int width;
        unsigned int height;
        unsigned int bpp;
        const unsigned char* data;

        ImagePNGTestData(const std::string& filename, unsigned int width, unsigned int height
            , unsigned int bpp, const unsigned char* data)
            : filename(filename), width(width), height(height), bpp(bpp), data(data)
        {

        }

        void testBench()
        {
            sofa::helper::io::ImagePNG img;

            EXPECT_TRUE(img.load(filename));
            EXPECT_EQ(width, img.getWidth());
            EXPECT_EQ(height, img.getHeight());
            EXPECT_EQ(width*height, img.getPixelCount());
            EXPECT_EQ(bpp, img.getBytesPerPixel());

            const unsigned char* testdata = img.getPixels();

            EXPECT_TRUE(0 == std::memcmp(data, testdata, width*height*bpp));
        }
    };
};

TEST_F(ImagePNG_test, ImagePNG_NoFile)
{
    /// This generate a test failure if no error message is generated.
    EXPECT_MSG_EMIT(Error) ;

    sofa::helper::io::ImagePNG imgNoFile;
    EXPECT_FALSE(imgNoFile.load("image/randomnamewhichdoesnotexist.png"));
}

TEST_F(ImagePNG_test, ImagePNG_NoImg)
{
    /// This generate a test failure if no error message is generated.
    EXPECT_MSG_EMIT(Error) ;

    sofa::helper::io::ImagePNG imgNoImage;
    EXPECT_FALSE(imgNoImage.load("image/imagetest_noimage.png"));
}

TEST_F(ImagePNG_test, ImagePNG_BlackWhite)
{
    unsigned int width = 800;
    unsigned int height = 600;
    unsigned int bpp = 3;
    unsigned int totalsize = width*height*bpp;

    unsigned char* imgdata = new unsigned char[totalsize];
    //half image (800x300) is black the other one is white
    std::fill(imgdata, imgdata + (800*300*3), 0);
    std::fill(imgdata + (800 * 300 * 3), imgdata + (800 * 600 * 3), 255);

    ImagePNGTestData imgBW("image/imagetest_blackwhite.png", width, height, bpp, imgdata);
    imgBW.testBench();
}

}// namespace sofa
