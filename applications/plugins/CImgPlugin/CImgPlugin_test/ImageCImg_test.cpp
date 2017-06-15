/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <CImgPlugin/ImageCImg.h>

#include <sofa/helper/system/FileRepository.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <gtest/gtest.h>

#include <SofaTest/Sofa_test.h>

namespace sofa {

//used to compare lossy images
const float PIXEL_TOLERANCE = 1.28; //0.5% difference on the average of the image

class ImageCImg_test : public Sofa_test<>
{
protected:
    ImageCImg_test() {

    }

    void SetUp()
    {
        sofa::helper::system::DataRepository.addFirstPath(CIMGPLUGIN_RESOURCES_DIR);
    }
    void TearDown()
    {
        sofa::helper::system::DataRepository.removePath(CIMGPLUGIN_RESOURCES_DIR);
    }

    bool checkExtension(const std::string& ext)
    {
        std::vector<std::string>::const_iterator extItBegin = sofa::helper::io::ImageCImgCreators::cimgSupportedExtensions.cbegin();
        std::vector<std::string>::const_iterator extItEnd = sofa::helper::io::ImageCImgCreators::cimgSupportedExtensions.cend();

        return (std::find(extItBegin, extItEnd, ext) != extItEnd);
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
                unsigned int total = width*height*bpp;
                //we will compare the average of pixels
                //and it has to be within a certain ratio with the reference
                //there are much better algorithms
                //but that is not the really the point here.

                float totalRef = std::accumulate(data, data + total, 0, std::plus<unsigned int>());
                float totalTest = std::accumulate(testdata, testdata + total, 0, std::plus<unsigned int>());

                res = fabs(totalRef - totalTest)/total < PIXEL_TOLERANCE;

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

            bool isLoaded = img.load(filename);
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
            std::for_each(testdata2, testdata2+width*height*bpp, [](unsigned char &n){ n+=PIXEL_TOLERANCE + 1; });

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

TEST_F(ImageCImg_test, ImageCImg_ReadBlackWhite)
{
    EXPECT_MSG_NOEMIT(Error, Warning) ;

    unsigned int width = 800;
    unsigned int height = 600;
    unsigned int bpp = 3;//images are RGB
    unsigned int totalsize = width*height*bpp;
    unsigned int halfTotalsize = totalsize * 0.5;

    unsigned char* imgdata = new unsigned char[totalsize];
    //half image (800x300) is black the other one is white
    std::fill(imgdata, imgdata + halfTotalsize, 0);
    std::fill(imgdata + halfTotalsize , imgdata + totalsize, 255);

    if(checkExtension("png"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.png", width, height, bpp, imgdata);
        imgBW.testBench();
    }
    if(checkExtension("jpg"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.jpg", width, height, bpp, imgdata);
        imgBW.testBench(true);
    }
    if(checkExtension("tiff"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.tiff", width, height, bpp, imgdata);
        imgBW.testBench();
    }
    if(checkExtension("bmp"))
    {
        ImageCImgTestData imgBW("imagetest_blackwhite.bmp", width, height, bpp, imgdata);
        imgBW.testBench();
    }
}


TEST_F(ImageCImg_test, ImageCImg_WriteBlackWhite)
{
    EXPECT_MSG_NOEMIT(Error, Warning) ;

    unsigned int width = 800;
    unsigned int height = 600;
    unsigned int bpp = 3;//image is RGB
    unsigned int totalsize = width*height*bpp;
    unsigned int halfTotalsize = totalsize * 0.5;

    unsigned char* imgdata = new unsigned char[totalsize];
    //half image (800x300) is black the other one is white
    std::fill(imgdata, imgdata + halfTotalsize, 0);
    std::fill(imgdata + halfTotalsize , imgdata + totalsize, 255);

    sofa::helper::io::ImageCImg img;
    bool isLoaded = img.load("imagetest_blackwhite.png");
    ASSERT_TRUE(isLoaded);

    bool isWritten = img.save(sofa::helper::system::DataRepository.getFirstPath() + "/output_bw.png");
    ASSERT_TRUE(isWritten);

    delete[] imgdata;
    imgdata = new unsigned char[totalsize];
    std::fill(imgdata, imgdata + halfTotalsize, 0);
    std::fill(imgdata + halfTotalsize , imgdata + totalsize, 255);

    ImageCImgTestData imgBW("output_bw.png", width, height, bpp, imgdata);
    imgBW.testBench();
}

}// namespace sofa
