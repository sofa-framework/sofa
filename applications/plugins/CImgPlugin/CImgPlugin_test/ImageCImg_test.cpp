/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <CImgPlugin/ImageCImg.h>

#include <sofa/helper/system/FileRepository.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

namespace sofa {

//used to compare lossy images
const float PIXEL_TOLERANCE = 1.28; //0.5% difference on the average of the image

class ImageCImg_test : public BaseTest
{
protected:
    ImageCImg_test() {

    }

    void SetUp() override
    {
        sofa::helper::system::DataRepository.addFirstPath(CIMGPLUGIN_RESOURCES_DIR);
    }
    void TearDown() override
    {
        sofa::helper::system::DataRepository.removePath(CIMGPLUGIN_RESOURCES_DIR);
    }

    bool checkExtension(const std::string& ext)
    {
        const auto extItBegin = sofa::helper::io::ImageCImgCreators::cimgSupportedExtensions.cbegin();
        const auto extItEnd = sofa::helper::io::ImageCImgCreators::cimgSupportedExtensions.cend();

        return std::find(extItBegin, extItEnd, ext) != extItEnd;
    }

    struct ImageCImgTestData
    {
        std::string filename;
        unsigned int width;
        unsigned int height;
        unsigned int bpp;
        const unsigned char* data;

        ImageCImgTestData(const std::string& filename, unsigned int width, unsigned int height
                , unsigned int bpp, const unsigned char* data)
                : filename(filename), width(width), height(height), bpp(bpp), data(data)
        {

        }

        bool comparePixels(bool lossy, const unsigned char* testdata)
        {
            bool res = true;
            if(lossy)
            {
                const unsigned int total = width*height*bpp;
                //we will compare the average of pixels
                //and it has to be within a certain ratio with the reference
                //there are much better algorithms
                //but that is not the really the point here.

                const float totalRef = std::accumulate(data, data + total, 0, std::plus<>());
                const float totalTest = std::accumulate(testdata, testdata + total, 0, std::plus<>());

                const auto difference = std::abs(totalRef - totalTest) / static_cast<float>(total);
                res = difference < PIXEL_TOLERANCE;
            }
            else
            {
                res = (0 == std::memcmp(data, testdata, width*height*bpp));
            }

            return res;
        }

        void testBench(bool lossy = false)
        {
            sofa::helper::io::ImageCImg img;

            const bool isLoaded = img.load(filename);
            ASSERT_TRUE(isLoaded);
            //necessary to test if the image was effectively loaded
            //otherwise segfault from Image (and useless to test the rest anyway)

            ASSERT_EQ(width, img.getWidth());
            ASSERT_NE(width + 123 , img.getWidth());
            ASSERT_EQ(height, img.getHeight());
            ASSERT_NE(height + 41, img.getHeight());
            ASSERT_EQ(width*height, img.getPixelCount());
            ASSERT_NE(width*height+11, img.getPixelCount());

            ASSERT_EQ(bpp, img.getBytesPerPixel());
            ASSERT_NE(bpp-2, img.getBytesPerPixel());

            const unsigned char* testdata = img.getPixels();
            ASSERT_TRUE(comparePixels(lossy, testdata));
            //add errors
            unsigned char* testdata2 = img.getPixels();
            std::for_each(testdata2, testdata2 + width * height * bpp,
                          [](unsigned char &n){ n += static_cast<unsigned char>(std::ceil(PIXEL_TOLERANCE)); });

            ASSERT_FALSE(comparePixels(lossy, testdata2));
        }
    };

};

TEST_F(ImageCImg_test, ImageCImg_NoFile)
{
    /// This generates a test failure if no error message is generated.
    EXPECT_MSG_EMIT(Error) ;

    sofa::helper::io::ImageCImg imgNoFile;
    EXPECT_FALSE(imgNoFile.load("randomnamewhichdoesnotexist.png"));

}

TEST_F(ImageCImg_test, ImageCImg_NoImg)
{
    EXPECT_MSG_EMIT(Error) ;
    EXPECT_MSG_NOEMIT(Warning) ;

    sofa::helper::io::ImageCImg imgNoImage;
    EXPECT_FALSE(imgNoImage.load("imagetest_noimage.png"));
}

TEST_F(ImageCImg_test, ImageCImg_ReadBlackWhite_png)
{
    EXPECT_MSG_NOEMIT(Error, Warning) ;

    const unsigned int width = 800;
    const unsigned int height = 600;
    const unsigned int bpp = 3;//images are RGB
    const unsigned int totalsize = width*height*bpp;
    const unsigned int halfTotalsize = totalsize / 2;

    std::vector<unsigned char> imgdata(totalsize, 0);
    //half image (800x300) is black the other one is white
    std::fill(imgdata.begin() + halfTotalsize , imgdata.end(), 255);

    if(checkExtension("png"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.png", width, height, bpp, imgdata.data());
        imgBW.testBench();
    }
}

TEST_F(ImageCImg_test, ImageCImg_ReadBlackWhite_jpg)
{
    EXPECT_MSG_NOEMIT(Error, Warning) ;

    const unsigned int width = 800;
    const unsigned int height = 600;
    const unsigned int bpp = 3;//images are RGB
    const unsigned int totalsize = width*height*bpp;
    const unsigned int halfTotalsize = totalsize / 2;

    std::vector<unsigned char> imgdata(totalsize, 0);
    //half image (800x300) is black the other one is white
    std::fill(imgdata.begin() + halfTotalsize , imgdata.end(), 255);

    if(checkExtension("jpg"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.jpg", width, height, bpp, imgdata.data());
        imgBW.testBench(true);
    }
}

TEST_F(ImageCImg_test, ImageCImg_ReadBlackWhite_tiff)
{
    EXPECT_MSG_NOEMIT(Error, Warning) ;

    const unsigned int width = 800;
    const unsigned int height = 600;
    const unsigned int bpp = 3;//images are RGB
    const unsigned int totalsize = width*height*bpp;
    const unsigned int halfTotalsize = totalsize / 2;

    std::vector<unsigned char> imgdata(totalsize, 0);
    //half image (800x300) is black the other one is white
    std::fill(imgdata.begin() + halfTotalsize , imgdata.end(), 255);

    if(checkExtension("tiff"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.tiff", width, height, bpp, imgdata.data());
        imgBW.testBench();
    }
}

TEST_F(ImageCImg_test, ImageCImg_ReadBlackWhite_bmp)
{
    EXPECT_MSG_NOEMIT(Error, Warning) ;

    const unsigned int width = 800;
    const unsigned int height = 600;
    const unsigned int bpp = 3;//images are RGB
    const unsigned int totalsize = width*height*bpp;
    const unsigned int halfTotalsize = totalsize / 2;

    std::vector<unsigned char> imgdata(totalsize, 0);
    //half image (800x300) is black the other one is white
    std::fill(imgdata.begin() + halfTotalsize , imgdata.end(), 255);

    if(checkExtension("bmp"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.bmp", width, height, bpp, imgdata.data());
        imgBW.testBench();
    }
}


TEST_F(ImageCImg_test, ImageCImg_WriteBlackWhite)
{
    EXPECT_MSG_NOEMIT(Error, Warning) ;

    const unsigned int width = 800;
    const unsigned int height = 600;
    const unsigned int bpp = 3;//image is RGB
    const unsigned int totalsize = width*height*bpp;
    const unsigned int halfTotalsize = totalsize / 2;

    std::vector<unsigned char> imgdata(totalsize, 0);
    //half image (800x300) is black the other one is white
    std::fill(imgdata.begin() + halfTotalsize , imgdata.end(), 255);

    sofa::helper::io::ImageCImg img;
    const bool isLoaded = img.load("imagetest_blackwhite.png");
    ASSERT_TRUE(isLoaded);

    const std::string output_bw_path = sofa::helper::system::DataRepository.getFirstPath() + "/output_bw.png";
    const bool isWritten = img.save(output_bw_path);
    ASSERT_TRUE(isWritten);

    ImageCImgTestData imgBW("output_bw.png", width, height, bpp, imgdata.data());
    imgBW.testBench();

    std::remove(output_bw_path.c_str());
}

}// namespace sofa
