#ifndef ZONEGENERATORIMAGETOOLBOX_H
#define ZONEGENERATORIMAGETOOLBOX_H

#include "zonegeneratorimagetoolboxaction.h"

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


class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBoxNoTemplated: public LabelImageToolBox
{
public:
    SOFA_CLASS(ZoneGeneratorImageToolBoxNoTemplated,LabelImageToolBox);

    typedef Vec<2,unsigned int> PixCoord;
    typedef sofa::defaulttype::Vec3d Vec3d;
    typedef sofa::defaulttype::Vec2d Vec2d;
    typedef sofa::defaulttype::Vec2i Vec2i;
    typedef vector<Vec3d> VecCoord;
    typedef vector<Vec2d> VecVec2d;
    typedef vector<PixCoord> VecPixCoord;
    typedef std::list<unsigned int> VecIndex;
    typedef std::list<Vec2i> ListVec2i;


    
    ZoneGeneratorImageToolBoxNoTemplated():LabelImageToolBox()


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
        return new sofa::gui::qt::ZoneGeneratorImageToolBoxAction(this,parent);
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
class SOFA_IMAGE_GUI_API ZoneGeneratorImageToolBox: public ZoneGeneratorImageToolBoxNoTemplated
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ZoneGeneratorImageToolBox,_ImageTypes),ZoneGeneratorImageToolBoxNoTemplated);
    
    typedef ZoneGeneratorImageToolBoxNoTemplated Inherited;
    
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;

    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;

    typedef sofa::defaulttype::ImageLPTransform<SReal> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    struct PointData
    {
        T color;
        Vec2i position;
        int radius;
        unsigned int counter;

        ListVec2i last;
    };

    T color(int index,int /*max*/)
    {
        return index;
    }

    
    ZoneGeneratorImageToolBox():ZoneGeneratorImageToolBoxNoTemplated(),
      d_image(initData(&d_image, "imageIn", "Input image")),
      d_imageOut(initData(&d_imageOut, "imageOut", "OutputImage")),
      d_size(initData(&d_size,Vec2d(1,1), "size","")),
      d_seed(initData(&d_seed, (unsigned int)0 , "seed","")),
      d_radius(initData(&d_radius, (float)0.1 , "radius","")),
      d_k(initData(&d_k, (unsigned int)100 , "k",""))
    {
    
    }
    
    virtual void init() override
    {
        Inherited::init();
        addInput(&d_image);
        addInput(&d_size);
        addInput(&d_seed);
        addInput(&d_radius);
        addInput(&d_k);
        addOutput(&d_imageOut);
        /*
        raImage im(this->d_image);    if( ! im->getCImgList().size() ) {serr<<"no input image"<<sendl; return;}

        color.setValue(im->getCImg().min());
        */
        generate();
    }

    inline float randf()
    {
        return (float)rand()/(float)RAND_MAX;
    }

    virtual void generate() override
    {

        //std::cout << "generate"<<std::endl;
        raImage im_in(this->d_image);
        waImage im_out(this->d_imageOut);

        im_out->clear();
        im_out->setDimensions(im_in->getDimensions());




        const unsigned int dimX = im_in->getCImg().width();
        const unsigned int dimY = im_in->getCImg().height();
//        const unsigned int dimZ = im_in->getCImg().depth();
        const unsigned int dimS = im_in->getCImg().spectrum();

//        const unsigned int nbPixels = dimX*dimY;




        //generate Samples
        //Fast Poisson Disk Sampling in Arbitrary Dimensions
        //http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
        //http://www.developpez.net/forums/d694076/autres-langages/algorithmes/distribution-homogene-points-surface-rectangulaire/

        // Step 1
        const float radius = d_radius.getValue();
        const float k = d_k.getValue();
        const Vec2d size = d_size.getValue();

        const Vec2i resoBG(size.x()/(radius/sqrt(2.0)),size.y()/(radius/sqrt(2.0)));
        const Vec2d sizeCellBG(size.x()/(float)resoBG.x(),size.y()/(float)resoBG.y());

        VecVec2d BG;
        BG.resize(resoBG.x()*resoBG.y());

        /*std::cout << "Step 1" << std::endl;
        std::cout << "radius " << radius <<std::endl;
        std::cout << "k " << k <<std::endl;
        std::cout << "size " << size <<std::endl;
        std::cout << "reso BG Grid " << resoBG <<std::endl;
        std::cout << "size cell BG Grid " << sizeCellBG <<std::endl << std::endl;
*/
        // Step 2
        srand(d_seed.getValue());

        Vec2d samples(size.x()*randf(),size.y()*randf());
        Vec2i position(samples.x()/sizeCellBG.x(),samples.y()/sizeCellBG.y());
        unsigned int index(position.x()+position.y()*resoBG.x());
        VecIndex vecId;
/*
        std::cout << "Step 2" <<std::endl;
        std::cout << "samples " << samples << std::endl;
        std::cout << "position " << position << std::endl;
        std::cout << "index " << index << std::endl << std::endl;
*/
        BG[index]=samples;
        vecId.push_back(index);

        // Step3

 //       std::cout << "Step 3" << std::endl;
//        unsigned int count=1;
        while(vecId.size())
        {
            Vec2d baseSample = BG[vecId.front()];
            vecId.pop_front();
//            std::cout << "    baseSample"<<baseSample<< " ( "<<vecId.size()<<" ) "<<std::endl;

            for(unsigned int j=0;j<k;)
            {
                Vec2d samples(4*sizeCellBG.x()*randf(),4*sizeCellBG.y()*randf());
                samples -= Vec2d(2*sizeCellBG.x(),2*sizeCellBG.y());
                samples += baseSample;
                Vec2i position(samples.x()/sizeCellBG.x(),samples.y()/sizeCellBG.y());

                double x = baseSample.x()-samples.x();
                double y = baseSample.y()-samples.y();
                double distance = x*x+y*y;
                double radius2 = radius*radius;

                if((2*radius2)>distance && radius2<distance)
                if(samples.x()>0 && samples.x()<size.x() && samples.y()>0 && samples.y()<size.y())
                if(position.x()>=0 && position.x()<resoBG.x() && position.y()>=0 && position.y()<resoBG.y())
                {
                    unsigned int index(position.x()+position.y()*resoBG.x());
                    j++;
                    //test if the cell of grid have no position
                    if(BG[index].x()==0 && BG[index].y()==0)
                    {
                        bool iscorrect=true;
                        //test radius < distance < 2*radius
                        for(int i=0;i<resoBG.x()*resoBG.y() && iscorrect;i++)
                        {
                            Vec2d &sam = BG[i];

                            if(sam.x()!=0 && sam.y()!=0)
                            {
                                double x = sam.x()-samples.x();
                                double y = sam.y()-samples.y();
                                double distance = x*x+y*y;

                                if(radius2>distance)
                                {
                                    iscorrect=false;
                                }
                            }
                        }

                        if(iscorrect)
                        {
                            BG[index]=samples;
                            vecId.push_back(index);

//                            std::cout << "         samples "<< samples << std::endl;
//                            std::cout << "         index"<< index << std::endl;

//                            count ++;
                        }
                    }
                }
            }
        }
//        std::cout << " count " << count << std::endl;


        // init backgroundColor;
        for(unsigned int i=0;i<dimX;i++)
        for(unsigned int j=0;j<dimY;j++)
        {
            T* pColor = new T[dimS];// = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
            for(int c=0;c<(int)dimS;c++)pColor[c]=color(0,1);
            //std::cout <<pColor[0]<<std::endl;

            im_out->getCImg().draw_point(i, j, 0, pColor);
            delete [] pColor;
        }

        // list position of area
        vector<PointData> listPosition;
        for(unsigned int i=0;i<BG.size();i++)
        {
            if(BG[i] != Vec2d(0,0) && BG[i].x()>=0 && BG[i].y()>=0 && BG[i].x()<dimX && BG[i].y()<dimY)
            {
                //std::cout << "BG " << BG[i] << std::endl;

                Vec2d position(BG[i].x()/size.x(),BG[i].y()/size.y());
                Vec2i pospixel(position.x()*dimX,position.y()*dimY);
                PointData p;
                p.position = pospixel;
                p.last.push_back(pospixel);
                listPosition.push_back(p);
                //std::cout << "pospixel "<<pospixel<<std::endl;

                //const T pColor[] = {1};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
                //std::cout <<pColor[0]<<std::endl;

                //im_out->getCImg().draw_point(pospixel.x(), pospixel.y(), 0, pColor);
            }
        }
        
        // select color
        unsigned int sizemax = listPosition.size();
        for(unsigned int i=0;i<sizemax;i++)
        {
            listPosition[i].color = color(i+1,sizemax+1);

            T* pColor = new T[dimS];// = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
            for(unsigned int c=0;c<dimS;c++)pColor[c]=listPosition[i].color;
            //std::cout <<pColor[0]<<std::endl;

            //std::cout << listPosition[i].color <<std::endl;

            im_out->getCImg().draw_point(listPosition[i].position.x() , listPosition[i].position.y(), 0, pColor);
            delete [] pColor;
        }

//        unsigned int probcount = sizemax+sizemax/2;

        int kk=sizemax;
        vector< unsigned int> test;
        test.resize(sizemax);
        for(unsigned int i=0;i<sizemax;i++)test[i]=0;
        //while(listPosition.size())


        while(kk>0)
        {
            /*
            unsigned int index = rand()%probcount;
            if(index<sizemax)index/=2;
            else index-=sizemax/2;

            if(index>sizemax)exit(0);
            test[index]++;*/

            unsigned int index = rand()%sizemax;

            PointData &p = listPosition[index];

            unsigned int plsize = p.last.size();

            for(unsigned int i=0;i<plsize;i++)
            {
                Vec2i v = p.last.front();
                p.last.pop_front();

   //             std::cout <<v.x()<<std::endl;

                uint random = rand()%16;

                if(v.x()>0)
                {
                    if(im_out->getCImg()(v.x()-1,v.y(),0,0)==0)
                    {
                        T* pColor = new T[dimS];// = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
                        for(unsigned int c=0;c<dimS;c++)pColor[c]=p.color;

                        im_out->getCImg().draw_point(v.x()-1 , v.y(), 0, pColor);

                        p.counter++;
                        if(random!=0)p.last.push_back(Vec2i(v.x()-1,v.y()));
                        else p.last.push_front(Vec2i(v.x()-1,v.y()));
                        delete [] pColor;
                    }
                }

                if(v.y()>0 )
                {
                    if(im_out->getCImg()(v.x(),v.y()-1,0,0)==0)
                    {
                        T* pColor = new T[dimS]; // = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
                        for(unsigned int c=0;c<dimS;c++)pColor[c]=p.color;

                        im_out->getCImg().draw_point(v.x() , v.y()-1, 0, pColor);

                        p.counter++;
                        if(random!=1)p.last.push_back(Vec2i(v.x(),v.y()-1));
                        else p.last.push_front(Vec2i(v.x(),v.y()-1));
                        delete [] pColor;
                    }
                }

                if(v.x()<(int)(dimX-1))
                {
                    if(im_out->getCImg()(v.x()+1,v.y(),0,0)==0)
                    {
                        T* pColor = new T[dimS];// = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
                        for(unsigned int c=0;c<dimS;c++)pColor[c]=p.color;

                        im_out->getCImg().draw_point(v.x()+1 , v.y(), 0, pColor);

                        p.counter++;
                        if(random!=2)p.last.push_back(Vec2i(v.x()+1,v.y()));
                        else p.last.push_front(Vec2i(v.x()+1,v.y()));
                        delete [] pColor;
                    }
                }

                if(v.y()<(int)(dimY-1))
                {
                    if(im_out->getCImg()(v.x(),v.y()+1,0,0)==0)
                    {
                        T* pColor = new T[dimS];// = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
                        for(unsigned int c=0;c<dimS;c++)pColor[c]=p.color;

                        im_out->getCImg().draw_point(v.x() , v.y()+1, 0, pColor);

                        p.counter++;
                        if(random!=3)p.last.push_back(Vec2i(v.x(),v.y()+1));
                        else p.last.push_front(Vec2i(v.x(),v.y()+1));
                        delete [] pColor;
                    }
                }

                if(p.last.size()==0)
                {
                    kk--;
                }

            }

            if(index!=sizemax-1)
            {
                std::swap_ranges(listPosition.begin()+index, listPosition.begin()+index+1, listPosition.begin()+index+1);
            }
        }

/*
        unsigned int max = 1 + (unsigned int)(sqrt((double)(dimX*dimX+dimY*dimY)));

        for(int i=0;i<dimX;i++)
        for(int j=0;j<dimY;j++)
        {
            unsigned int currentColor = im_out->getCImg()(i,j,0,0,0);

            std::cout << currentColor << std::endl;

            unsigned int currentDistance = max;

            for(int ii=0;ii<dimX;ii++)
            for(int jj=0;jj<dimY;jj++)
            {
                unsigned int color2 = im_out->getCImg()(ii,jj,0,0,0);
                int x = i-ii, y= j-jj;
                unsigned int distance = (unsigned int) sqrt( (float)(x*x+y*y) );

                if(currentColor!=color2 && distance < currentDistance)
                {
                    currentDistance = distance;
                }
            }

            T pColor[dimS];// = {0};//(unsigned short)(((float)i/(float)dimX)*(float)USHRT_MAX)};
            T colorr = color(currentDistance,max);
            for(unsigned int c=0;c<dimS;c++)pColor[c]=colorr;

            im_out->getCImg().draw_point(i , j, 1, pColor);

        }*/

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




     //   std::cout << "generate"<<std::endl;
    }

/*
    virtual void generate()
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


        //Treat the selected pixel and its neighbourhood in 3D (works in 2D)
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

    virtual void getImageSize(unsigned int& x,unsigned int& y,unsigned int &z)
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

    }*/

protected:
    Data< ImageTypes >   d_image; ///< Input image
//    Data< TransformType> d_transform;
    Data< ImageTypes >   d_imageOut; ///< OutputImage

    Data< Vec2d > d_size;
    Data< unsigned int > d_seed;
    Data< float > d_radius;
    Data< unsigned int > d_k;

//    Data<T> color;
//    vector<pixCoord> processList;

};



}}}

#endif // ZoneGeneratorImageToolBox_H
