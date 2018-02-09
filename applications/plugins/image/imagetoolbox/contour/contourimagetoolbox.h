#ifndef CONTOURIMAGETOOLBOX_H
#define CONTOURIMAGETOOLBOX_H

#include <image/image_gui/config.h>

#include "contourimagetoolboxaction.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <image/ImageTypes.h>

#include "../labelimagetoolbox.h"

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


class SOFA_IMAGE_GUI_API ContourImageToolBoxNoTemplated: public LabelImageToolBox
{
public:
    SOFA_CLASS(ContourImageToolBoxNoTemplated,LabelImageToolBox);

    typedef Vec<3,unsigned int> pixCoord;
    typedef sofa::defaulttype::Vec3d Coord;
    typedef vector<Vec3d> VecCoord;
    typedef vector<pixCoord> VecPixCoord;
    
    ContourImageToolBoxNoTemplated():LabelImageToolBox()
        , d_ip(initData(&d_ip, "imageposition",""))
        , d_p(initData(&d_p, "3Dposition",""))
        , d_axis(initData(&d_axis, (unsigned int)4,"axis",""))
        , d_value(initData(&d_value,"value",""))
        , d_vecCoord(initData(&d_vecCoord,"out","Output list of space position of each pixel on contour"))
        , d_vecPixCoord(initData(&d_vecPixCoord,"out2","Output list of image position of each pixel on contour"))
        , threshold(initData(&threshold,"threshold",""))
        , radius(initData(&radius,"radius",""))
    {
    
    }
    
    virtual void init() override
    {
        d_ip.setGroup("PixelClicked");
        d_p.setGroup("PixelClicked");
        d_axis.setGroup("PixelClicked");
        d_value.setGroup("PixelClicked");

        addOutput(&d_vecCoord);
        addOutput(&d_vecPixCoord);
    }
    
    virtual sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=NULL) override
    {
        return new sofa::gui::qt::ContourImageToolBoxAction(this,parent);
    }
    
    
    virtual void segmentation()=0;
    virtual void getImageSize(unsigned int& x,unsigned int& y,unsigned int &z)=0;
     
    
public:
    Data<pixCoord> d_ip;
    Data<Vec3d> d_p;
    Data<unsigned int> d_axis;
    Data<std::string> d_value;
    Data<VecCoord> d_vecCoord; ///< Output list of space position of each pixel on contour
    Data<VecPixCoord> d_vecPixCoord; ///< Output list of image position of each pixel on contour

    Data<double> threshold;
    Data<int> radius;
    
};



template<class _ImageTypes>
class SOFA_IMAGE_GUI_API ContourImageToolBox: public ContourImageToolBoxNoTemplated
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ContourImageToolBox,_ImageTypes),ContourImageToolBoxNoTemplated);
    
    typedef ContourImageToolBoxNoTemplated Inherited;
    
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;

    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;

    typedef sofa::defaulttype::ImageLPTransform<SReal> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    
    ContourImageToolBox():ContourImageToolBoxNoTemplated()
        , d_image(initData(&d_image,"image","Input image"))
        , d_transform(initData(&d_transform,"transform","Transform"))
        , d_imageOut(initData(&d_imageOut,"imageOut","Image containing the contour"))
    {
    
    }
    
    virtual void init() override
    {
        Inherited::init();
        addInput(&d_image);
        addInput(&d_transform);
        addOutput(&d_imageOut);

        raImage im(this->d_image);    if( ! im->getCImgList().size() ) {serr<<"no input image"<<sendl; return;}

        color.setValue(im->getCImg().min());

        segmentation();
    }
    
    
    virtual void segmentation() override
    {

        raImage im_in(this->d_image);
        waImage im_out(this->d_imageOut);

        im_out->clear();
        im_out->setDimensions(im_in->getDimensions());
        im_out->getCImg().assign(im_in->getCImg());

        if(d_ip.getValue() == pixCoord(0,0,0)) return;

        helper::WriteAccessor<Data<VecCoord> > out = d_vecCoord;
        out.clear();
        out.push_back(d_p.getValue());

        helper::WriteAccessor<Data<VecPixCoord> > out2 = d_vecPixCoord;
        out2.clear();
        out2.push_back(d_ip.getValue());

        const T pColor[] = {color.getValue(), color.getValue(), color.getValue()};
        im_out->getCImg().draw_point(d_ip.getValue()[0], d_ip.getValue()[1], d_ip.getValue()[2], pColor);

        vector<bool> pixel_tag;
        const unsigned int dimX = im_in->getCImg().width();
        const unsigned int dimY = im_in->getCImg().height();
        const unsigned int dimZ = im_in->getCImg().depth();
        pixel_tag.resize(dimX*dimY*dimZ, false);


        processList.clear();
        processList.push_back(d_ip.getValue());


        /*Treat the selected pixel and its neighbourhood in 3D (works in 2D)*/
        while(processList.size() != 0)
        {
            pixCoord pixC = processList[0];

            const unsigned int i = pixC[0];
            const unsigned int j = pixC[1];
            const unsigned int k = pixC[2];

            raImage im_in(this->d_image);

            if (im_in->getCImg().containsXYZC(i+1,j,k))   isPixelOnContour( i+1,j,k, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j,k))   isPixelOnContour( i-1,j,k, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j+1,k))   isPixelOnContour( i,j+1,k, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j+1,k)) isPixelOnContour( i+1,j+1,k, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j+1,k)) isPixelOnContour( i-1,j+1,k, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j-1,k))   isPixelOnContour( i,j-1,k, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j-1,k)) isPixelOnContour( i+1,j-1,k, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j-1,k)) isPixelOnContour( i-1,j-1,k, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j,k+1))   isPixelOnContour( i,j,k+1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j,k+1)) isPixelOnContour( i+1,j,k+1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j,k+1)) isPixelOnContour( i-1,j,k+1, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j+1,k+1))   isPixelOnContour( i,j+1,k+1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j+1,k+1)) isPixelOnContour( i+1,j+1,k+1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j+1,k+1)) isPixelOnContour( i-1,j+1,k+1, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j-1,k+1))   isPixelOnContour( i,j-1,k+1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j-1,k+1)) isPixelOnContour( i+1,j-1,k+1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j-1,k+1)) isPixelOnContour( i-1,j-1,k+1, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j,k-1))   isPixelOnContour( i,j,k-1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j,k-1)) isPixelOnContour( i+1,j,k-1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j,k-1)) isPixelOnContour( i-1,j,k-1, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j+1,k-1))   isPixelOnContour( i,j+1,k-1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j+1,k-1)) isPixelOnContour( i+1,j+1,k-1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j+1,k-1)) isPixelOnContour( i-1,j+1,k-1, pixC, pixel_tag );

            if (im_in->getCImg().containsXYZC(i,j-1,k-1))   isPixelOnContour( i,j-1,k-1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i+1,j-1,k-1)) isPixelOnContour( i+1,j-1,k-1, pixC, pixel_tag );
            if (im_in->getCImg().containsXYZC(i-1,j-1,k-1)) isPixelOnContour( i-1,j-1,k-1, pixC, pixel_tag );

            processList.erase(processList.begin());
        }

    }

    virtual void getImageSize(unsigned int& x,unsigned int& y,unsigned int &z) override
    {
        raImage im(this->d_image);

        x = im->getCImg().width();
        y = im->getCImg().height();
        z = im->getCImg().depth();
    }


    void isPixelOnContour(const unsigned int i, const unsigned int j, const unsigned int k, pixCoord pixC, vector<bool> &pixel_tag)
    {

        helper::WriteAccessor<Data<VecCoord> > out = d_vecCoord;
        helper::WriteAccessor<Data<VecPixCoord> > out2 = d_vecPixCoord;

        raImage im_in(this->d_image);

        const unsigned int dimX = im_in->getCImg().width() -1;
        const unsigned int dimY = im_in->getCImg().height() -1;

        if ( isPixelInDirectionRange( i,j,k, pixC)
                && isPixelInGradNormRange( i,j,k, pixC)
                && isPixelInPositionRange( i,j,k )
                && !pixel_tag[k*dimY*dimX + j*dimX + i] )
        {
            pixCoord pixC = pixCoord(i,j,k);
            Coord c;
            getPixelCoord( c,pixC );
            out.push_back(c);
            out2.push_back(pixC);

            pixel_tag[k*dimY*dimX + j*dimX + i] = true;

            waImage im_out(this->d_imageOut);

            const T pColor[] = {color.getValue(), color.getValue(), color.getValue()};
            if ( im_out->getCImg().containsXYZC(i, j, k)) im_out->getCImg().draw_point(i, j, k, pColor);

            processList.push_back(pixC);
         }
    }


    bool isPixelInDirectionRange( const unsigned int i, const unsigned int j, const unsigned int k, const pixCoord pixC )
    {

        const double eps = 0.5; //threshold for the orthoganality

        const unsigned int x = pixC[0];
        const unsigned int y = pixC[1];
        const unsigned int z = pixC[2];

        raImage im_in(this->d_image);
        Vec<3,double> grad;

        if ( !im_in->getCImg().containsXYZC(x-1,y,z) || !im_in->getCImg().containsXYZC(x+1,y,z)) return false;
        if ( !im_in->getCImg().containsXYZC(x,y-1,z) || !im_in->getCImg().containsXYZC(x,y+1,z)) return false;
        if ( !im_in->getCImg().containsXYZC(x,y,z-1) || !im_in->getCImg().containsXYZC(x,y,z+1)) return false;

        grad[0] = ( im_in->getCImg()(x+1,y,z) - im_in->getCImg()(x-1,y,z) )/2;
        grad[1] = ( im_in->getCImg()(x,y+1,z) - im_in->getCImg()(x,y-1,z) )/2;
        grad[2] = ( im_in->getCImg()(x,y,z+1) - im_in->getCImg()(x,y,z-1) )/2;

        double x2, y2, z2;
        if (grad.norm() > 1e-10)
        {
            x2 = grad[0] / grad.norm();
            y2 = grad[1] / grad.norm();
            z2 = grad[2] / grad.norm();
        }
        else return false;

        Coord c1;
        Coord c2;
        pixCoord p2 = pixCoord(i,j,k);

        getPixelCoord(c1,pixC);
        getPixelCoord(c2,p2);

        double x1, y1, z1;
        if ((c2-c1).norm() > 1e-10)
        {
            x1 = (c2[0] - c1[0]) / (c2-c1).norm();
            y1 = (c2[1] - c1[1]) / (c2-c1).norm();
            z1 = (c2[2] - c1[2]) / (c2-c1).norm();
        }
        else return false;

        double scalProd = x1*x2 + y1*y2 + z1*z2;

        if ( scalProd < eps && scalProd > - eps )
            return true;

        return false;
    }


    bool isPixelInGradNormRange( const unsigned int i, const unsigned int j, const unsigned int k, const pixCoord pixC)
    {
        const unsigned int x = pixC[0];
        const unsigned int y = pixC[1];
        const unsigned int z = pixC[2];

        raImage im_in(this->d_image);
        Vec<3,double> grad1;
        Vec<3,double> grad2;

        grad1[0] = ( im_in->getCImg()(i+1,j,k) - im_in->getCImg()(i-1,j,k) )/2;
        grad1[1] = ( im_in->getCImg()(i,j+1,k) - im_in->getCImg()(i,j-1,k) )/2;
        grad1[2] = ( im_in->getCImg()(i,j,k+1) - im_in->getCImg()(i,j,k-1) )/2;

        grad2[0] = ( im_in->getCImg()(x+1,y,z) - im_in->getCImg()(x-1,y,z) )/2;
        grad2[1] = ( im_in->getCImg()(x,y+1,z) - im_in->getCImg()(x,y-1,z) )/2;
        grad2[2] = ( im_in->getCImg()(x,y,z+1) - im_in->getCImg()(x,y,z-1) )/2;

        const double eps = threshold.getValue();

        if( grad2.norm() < 10 ) // random value... get a better one
            return false;

        if( grad1.norm() > grad2.norm() - eps && grad1.norm() < grad1.norm() + eps)
            return true;

        return false;
    }


    bool isPixelInPositionRange( const unsigned int i, const unsigned int j, const unsigned int k )
    {
        const unsigned int r = radius.getValue();

        const pixCoord p = d_ip.getValue();

        if ( (i - p[0])*(i - p[0]) + (j - p[1])*(j - p[1]) + (k - p[2])*(k - p[2]) < r*r )
                 return true;

        return false;
    }


    void getPixelCoord( Coord &c, const pixCoord p)
    {

        Vec<2,Coord> BB = getBB();
        const double xmax = BB[1][0];
        const double xmin = BB[0][0];

        const double ymax = BB[1][1];
        const double ymin = BB[0][1];

        const double zmax = BB[1][2];
        const double zmin = BB[0][2];

        raImage im_in(this->d_image);
        const unsigned int dimX = im_in->getCImg().width();
        const unsigned int dimY = im_in->getCImg().height();
        const unsigned int dimZ = im_in->getCImg().depth();

        const double dx = (xmax - xmin) /  double(dimX) ;
        c[0] = (( xmin + p[0]*dx ) + ( xmin + (p[0]-1)*dx )) / 2;

        const double dy = (ymax - ymin) /  double(dimY) ;
        c[1] = (( ymin + p[1]*dy ) + ( ymin + (p[1]-1)*dy )) / 2;

        const double dz = (zmax - zmin) /  double(dimZ) ;
        c[2] = (( zmin + p[2]*dz ) + ( zmin + (p[2]-1)*dz )) / 2;

    }


    Vec<2,Coord> getBB() // get image corners
    {

        Vec<2,Coord> BB;
        raImage rimage(this->d_image);
        raTransform rtransform(this->d_transform);

        const imCoord dim=rimage->getDimensions();
        Vec<8,Coord> p;
        p[0]=Vector3(0,0,0);
        p[1]=Vector3(dim[0]-1,0,0);
        p[2]=Vector3(0,dim[1]-1,0);
        p[3]=Vector3(dim[0]-1,dim[1]-1,0);
        p[4]=Vector3(0,0,dim[2]-1);
        p[5]=Vector3(dim[0]-1,0,dim[2]-1);
        p[6]=Vector3(0,dim[1]-1,dim[2]-1);
        p[7]=Vector3(dim[0]-1,dim[1]-1,dim[2]-1);

        Coord tp=rtransform->fromImage(p[0]);
        BB[0]=tp;
        BB[1]=tp;
        for(unsigned int j=1; j<8; j++)
        {
            tp=rtransform->fromImage(p[j]);
            for(unsigned int k=0; k<tp.size(); k++)
            {
                if(BB[0][k]>tp[k]) BB[0][k]=tp[k];
                if(BB[1][k]<tp[k]) BB[1][k]=tp[k];
            }
        }
        return BB;

    }

protected:
    Data< ImageTypes >   d_image; ///< Input image
    Data< TransformType> d_transform; ///< Transform
    Data< ImageTypes >   d_imageOut; ///< Image containing the contour

    Data<T> color;
    vector<pixCoord> processList;

};


}}}

#endif // ContourImageToolBox_H
