#ifndef DISTANCEZONEIMAGETOOLBOX_H
#define DISTANCEZONEIMAGETOOLBOX_H

#include "distancezoneimagetoolboxaction.h"
#include "zonegeneratorimagetoolbox.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <image/ImageTypes.h>

#include "../labelimagetoolbox.h"

#include <image/image_gui/config.h>

namespace sofa
{

namespace component
{

namespace engine
{

using helper::vector;
using defaulttype::Vec;
using defaulttype::Vector3;
using namespace sofa::defaulttype;


class SOFA_IMAGE_GUI_API DistanceZoneImageToolBoxNoTemplated: public LabelImageToolBox
{
public:
    SOFA_CLASS(DistanceZoneImageToolBoxNoTemplated,LabelImageToolBox);

    typedef Vec<2,unsigned int> PixCoord;
    typedef sofa::defaulttype::Vec3d Vec3d;
    typedef sofa::defaulttype::Vec2d Vec2d;
    typedef sofa::defaulttype::Vec2i Vec2i;
    typedef vector<Vec3d> VecCoord;
    typedef vector<Vec2d> VecVec2d;
    typedef vector<PixCoord> VecPixCoord;
    typedef std::list<unsigned int> VecIndex;
    typedef std::list<Vec2i> ListVec2i;


    
    DistanceZoneImageToolBoxNoTemplated():LabelImageToolBox()


       /* , d_ip(initData(&d_ip, "imageposition",""))
        , d_p(initData(&d_p, "3Dposition",""))
        , d_axis(initData(&d_axis, (unsigned int)4,"axis",""))
        , d_value(initData(&d_value,"value",""))
        , d_vecCoord(initData(&d_vecCoord,"out","Output list of space position of each pixel on contour"))
        , d_vecPixCoord(initData(&d_vecPixCoord,"out2","Output list of image position of each pixel on contour"))
        , threshold(initData(&threshold,"threshold",""))
        , radius(initData(&radius,"radius",""))*/
    {
    
    }
    
    virtual void init() override
    {/*
        d_ip.setGroup("PixelClicked");
        d_p.setGroup("PixelClicked");
        d_axis.setGroup("PixelClicked");
        d_value.setGroup("PixelClicked");

        addOutput(&d_vecCoord);
        addOutput(&d_vecPixCoord);*/
    }
    
    virtual sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=NULL) override
    {
        return new sofa::gui::qt::DistanceZoneImageToolBoxAction(this,parent);
    }
    
    
    virtual void generate()=0;
    //virtual void getImageSize(unsigned int& x,unsigned int& y,unsigned int &z)=0;
     
    
public:
    /*Data<pixCoord> d_ip;
    Data<Vec3d> d_p;
    Data<unsigned int> d_axis;
    Data<std::string> d_value;
    Data<VecCoord> d_vecCoord; ///< Output list of space position of each pixel on contour
    Data<VecPixCoord> d_vecPixCoord; ///< Output list of image position of each pixel on contour

    Data<double> threshold;
    Data<int> radius;
    */
};



template<class _ImageTypes>
class SOFA_IMAGE_GUI_API DistanceZoneImageToolBox: public DistanceZoneImageToolBoxNoTemplated
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DistanceZoneImageToolBox,_ImageTypes),DistanceZoneImageToolBoxNoTemplated);
    
    typedef DistanceZoneImageToolBoxNoTemplated Inherited;
    
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;

    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;

    typedef sofa::defaulttype::ImageLPTransform<SReal> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    
    DistanceZoneImageToolBox():DistanceZoneImageToolBoxNoTemplated(),
      d_image(initData(&d_image, "imageIn", "Input image")),
      d_imageOut(initData(&d_imageOut, "imageOut", "OutputImage"))
    {
    
    }
    
    virtual void init() override
    {
        Inherited::init();
        addInput(&d_image);
        addOutput(&d_imageOut);
        /*
        raImage im(this->d_image);    if( ! im->getCImgList().size() ) {serr<<"no input image"<<sendl; return;}

        color.setValue(im->getCImg().min());
        */
        generate();
    }


    virtual void generate() override
    {
        raImage im_in(this->d_image);
        waImage im_out(this->d_imageOut);

        im_out->clear();
        im_out->setDimensions(im_in->getDimensions());

        const unsigned int dimX = im_in->getCImg().width();
        const unsigned int dimY = im_in->getCImg().height();
//        const unsigned int dimZ = im_in->getCImg().depth();
        const unsigned int dimS = im_in->getCImg().spectrum();

//        const unsigned int nbPixels = dimX*dimY;

//        unsigned int max = 1 + (unsigned int)(sqrt((double)(dimX*dimX+dimY*dimY)));

        double maxdistance=0;
        // calculate max distance
        for(int i=0;i<(int)dimX;i++)
        for(int j=0;j<(int)dimY;j++)
        {
            unsigned int currentColor = im_in->getCImg()(i,j,0,0,0);

            for(int ii=0;ii<(int)dimX;ii++)
            for(int jj=0;jj<(int)dimY;jj++)
            {
                unsigned int color2 = im_in->getCImg()(ii,jj,0,0,0);
                int x = i-ii, y= j-jj;
                double distance = sqrt( (float)(x*x+y*y) );

                if(currentColor!=color2 && maxdistance<distance)
                {
                    maxdistance = distance;
                }
            }

        }


        // set color
        for(int i=0;i<(int)dimX;i++)
        for(int j=0;j<(int)dimY;j++)
        {
            unsigned int currentColor = im_in->getCImg()(i,j,0,0,0);

            //std::cout << currentColor << std::endl;

            double currentDistance = 1;



            for(int ii=0;ii<(int)dimX;ii++)
            for(int jj=0;jj<(int)dimY;jj++)
            {
                unsigned int color2 = im_in->getCImg()(ii,jj,0,0,0);
                int x = i-ii, y= j-jj;
                double distance = sqrt( (float)(x*x+y*y) )/maxdistance;

                if(currentColor!=color2 && distance < currentDistance)
                {
                    currentDistance = distance;
                }
            }


            //T pColor[dimS];// = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
            T* pColor = new T[dimS];
            T colorr = (currentDistance * (double) std::numeric_limits<T>::max());
            for(unsigned int c=0;c<dimS;c++)pColor[c]=colorr;

            im_out->getCImg().draw_point(i, j, 0, pColor);
            delete [] pColor; 
        }

        //for(unsigned int i=0;i<sizemax;i++)std::cout << "test " << i << " :" << test[i]<<std::endl;

        
        

/*
        for(unsigned int i=0;i<dimX;i++)
        for(unsigned int j=0;j<dimY;j++)
        {
            const T pColor[] = {(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
            std::cout <<pColor[0]<<std::endl;

            im_out->getCImg().draw_point(i, j, 0, pColor);
        }
*/

        //vecCoord2 vec;




       // std::cout << "generate"<<std::endl;
    }



protected:
    Data< ImageTypes >   d_image; ///< Input image
//    Data< TransformType> d_transform;
    Data< ImageTypes >   d_imageOut; ///< OutputImage

//    Data<T> color;
//    vector<pixCoord> processList;

};



}}}

#endif // DistanceZoneImageToolBox_H
