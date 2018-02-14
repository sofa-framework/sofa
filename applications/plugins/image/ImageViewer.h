/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_IMAGE_IMAGEVIEWER_H
#define SOFA_IMAGE_IMAGEVIEWER_H


#include <image/config.h>
#include "ImageTypes.h"
#include "VectorVis.h"

#include <sofa/helper/io/Image.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Event.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <SofaGeneralVisual/RecordedCamera.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace misc
{

/**
   * \brief This component is responsible for displaying images in SOFA
   *
   *  ImageViewer scene options:
   *
   *  <b>image</b> - a link to the imge in the ImageContainer component
   *
   *  <b>transform</b> - a link to the transformation in the ImageContainer component
   *
   *  <b>histo</b> -
   *
   *  <b>plane</b> -
   *
   *  <b>vectorvis</b> - Describes the settings for vizualizing vectors and tensors. Input string should be in the form:
   *	"subsampleXY subsampleZ scale rgb shape tensorOrder", where:
   *	subsampleXY is an integer <i>n</i> where in the X and Y planes, a shape is drawn every <i>n</i> voxels.
   *    subsampleZ is an integer <i>n</i> where in the Z plane, a shape is drawn every <i>n</i> voxels.
   *	scale is an integer <i>n</i> such that each shape is drawn <i>n</i> times its normal size.
   *    rgb is a bool. When true, an image with 3 channels is displayed as an rgb image, and when false, it is
   *		displayed as a greyscale image where the value is the L2 norm of all the channels.
   *	shape is a bool. When true, an image with 3 channels has vectors displayed, and an image with 6 channels has tensors displayed.
   *    tensorOrder is a string describing the order in which the 6 tensors values are provided in the image. The three supported types are:
   *		LowerTriRowMajor
   *		UpperTriRowMajor
   *		DiagonalFirst
   *
   *	The default vectorvis configuration is "5 5 10 true false LowerTriRowMajor"
   */
template<class _ImageTypes>
class SOFA_IMAGE_API ImageViewer : public sofa::core::objectmodel::BaseObject
{
public:
    typedef sofa::core::objectmodel::BaseObject Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageViewer, _ImageTypes), Inherited);
    
    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image; ///< input image
    
    Data<bool> showSlicedModels; ///< display visual models on cutPlanes

    // @name Histogram
    /**@{*/
    typedef defaulttype::Histogram<T> HistogramType;
    typedef helper::WriteOnlyAccessor<Data< HistogramType > > waHisto;
    typedef helper::ReadAccessor<Data< HistogramType > > raHisto;
    Data< HistogramType > histo;
    /**@}*/
    
    //@name Transform data
    /**@{*/
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;
    /**@}*/
    
    // @name Plane selection
    /**@{*/
    typedef defaulttype::ImagePlane<T> ImagePlaneType;
    typedef helper::ReadAccessor<Data< ImagePlaneType > > raPlane;
    typedef helper::WriteOnlyAccessor<Data< ImagePlaneType > > waPlane;
    Data< ImagePlaneType > plane;
    /**@}*/

    // @name Vector of 3D points for navigation
    /**@{*/
    typedef helper::ReadAccessor <core::objectmodel::Data<helper::vector<Coord> > >raPoints;
    typedef helper::WriteOnlyAccessor<core::objectmodel::Data<helper::vector<Coord> > >waPoints;
    Data< helper::vector<Coord> > points;
    /**@}*/
    
    // @name Vector visualization
    /**@{*/
    typedef helper::ReadAccessor<Data< defaulttype::VectorVis > > raVis;
    typedef helper::WriteOnlyAccessor<Data< defaulttype::VectorVis > > waVis;
    Data<defaulttype::VectorVis> vectorVisualization;
    /**@}*/
    
    Data <int> scroll; ///< 0 if no scrolling, 1 for up, 2 for down, 3 left, and 4 for right
    Data <bool> display; ///< Boolean to activate/desactivate the display of the image

    typedef component::visualmodel::VisualModelImpl VisuModelType;
    
    std::string getTemplateName() const  override {	return templateName(this);	}
    static std::string templateName(const ImageViewer<ImageTypes>* = NULL)	{ return ImageTypes::Name(); }
    
    ImageViewer() : Inherited()
      , image(initData(&image,ImageTypes(),"image","input image"))
      , showSlicedModels(initData(&showSlicedModels, false, "slicedModels", "display visual models on cutPlanes"))
      , histo(initData(&histo, HistogramType(256,256,false),"histo",""))
      , transform(initData(&transform, TransformType(), "transform" , ""))
      , plane ( initData ( &plane, ImagePlaneType(), "plane" , "" ) )
      , points ( initData ( &points, helper::vector<Coord> (), "points" , "" ) )
      , vectorVisualization ( initData (&vectorVisualization, defaulttype::VectorVis(), "vectorvis", ""))
      , scroll( initData (&scroll, int(0), "scrollDirection", "0 if no scrolling, 1 for up, 2 for down, 3 left, and 4 for right"))
      , display( initData(&display, true, "display", "true if image is displayed, false otherwise"))

    {
        this->addAlias(&image, "outputImage");
        this->addAlias(&transform, "outputTransform");
        
        image.setGroup("Image");
        image.setReadOnly(true);
        
        histo.setGroup("Histogram");
        histo.setWidget("imagehistogram");
        
        transform.setGroup("Transform");
        transform.setReadOnly(true);
        
        plane.setGroup("Image");
        plane.setWidget("imageplane");
        
        vectorVisualization.setWidget("vectorvis");

#ifndef SOFA_NO_OPENGL
        for(unsigned int i=0;i<3;i++)	cutplane_tex[i]=NULL;
#endif //SOFA_NO_OPENGL
    }
    
    
    virtual ~ImageViewer()
    {
#ifndef SOFA_NO_OPENGL
        for(unsigned int i=0;i<3;i++)	if(cutplane_tex[i]) delete cutplane_tex[i];
#endif //SOFA_NO_OPENGL
    }
    
    virtual void init() override
    {
        
        // getvisuals
        std::vector<VisuModelType*> visuals;
        sofa::core::objectmodel::BaseContext* context = this->getContext();
        context->get<VisuModelType>(&visuals,core::objectmodel::BaseContext::SearchRoot);
        
        // set histogram data
        waHisto whisto(this->histo);        whisto->setInput(image.getValue());
        
        // record user values of plane position
        bool usedefaultplanecoord=true;
        if(this->plane.isSet())  usedefaultplanecoord=false;
        
        // set plane data
        waPlane wplane(this->plane);        wplane->setInput(image.getValue(),transform.getValue(),visuals);
        
        // set default plane pos
        if(usedefaultplanecoord) wplane->setPlane(typename ImagePlaneType::pCoord(wplane->getDimensions()[0]/2,wplane->getDimensions()[1]/2,wplane->getDimensions()[2]/2));
        
        // enable vecorvis ?
        if(wplane->getDimensions()[3]<2) vectorVisualization.setDisplayed(false);
        else         vectorVisualization.setGroup("Vectors");

        raVis rvis(this->vectorVisualization);

        whisto->setMergeChannels(!rvis->getRgb());
        wplane->setMergeChannels(!rvis->getRgb());
        wplane->setClamp(whisto->getClamp());
    }
    
    
    virtual void reinit() override
    {
        waHisto whisto(this->histo);
        waPlane wplane(this->plane);
        raVis rvis(this->vectorVisualization);
        
        whisto->setMergeChannels(!rvis->getRgb());
        wplane->setMergeChannels(!rvis->getRgb());
        wplane->setClamp(whisto->getClamp());

    }
    
    virtual void handleEvent( sofa::core::objectmodel::Event* event) override
    {
        typename ImagePlaneType::pCoord pc(0,0,0);

        if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
        {
            sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);

            waPlane wplane(this->plane);
            unsigned int xmax = this->image.getValue().getDimensions()[0];
            unsigned int ymax = this->image.getValue().getDimensions()[1];
            unsigned int zmax = this->image.getValue().getDimensions()[2];
            switch(ev->getKey())
            {
            case '4':
                pc = wplane->getPlane();
                if (pc[0] > 0) pc[0]--;
                wplane->setPlane(pc);
                break;
            case '6':
                pc = wplane->getPlane();
                if (pc[0] < xmax) pc[0]++;
                wplane->setPlane(pc);
                break;
            case '-':
                pc = wplane->getPlane();
                if (pc[1] > 0) pc[1]--;
                wplane->setPlane(pc);
                break;
            case '+':
                pc = wplane->getPlane();
                if (pc[1] < ymax) pc[1]++;
                wplane->setPlane(pc);
                break;
            case '8':
                pc = wplane->getPlane();
                if (pc[2] > 0) pc[2]--;
                wplane->setPlane(pc);
                break;
            case '2':
                pc = wplane->getPlane();
                if (pc[2] < zmax) pc[2]++;
                wplane->setPlane(pc);
                break;
            case '5':
                reinit();
                break;
            }
        }

        if (sofa::simulation::AnimateBeginEvent::checkEventType(event) && this->scroll.getValue() > 0)
        {
            waPlane wplane(this->plane);
            unsigned int xmax = this->image.getValue().getDimensions()[0];
            unsigned int ymax = this->image.getValue().getDimensions()[1];
            unsigned int zmax = this->image.getValue().getDimensions()[2];
            switch(this->scroll.getValue())
            {
            case 1:
                pc = wplane->getPlane();
                if (pc[0] > 1) pc[0]--;
                wplane->setPlane(pc);
                break;
            case 2:
                pc = wplane->getPlane();
                if (pc[0] < xmax-1) pc[0]++;
                wplane->setPlane(pc);
                break;
            case 3:
                pc = wplane->getPlane();
                if (pc[1] > 1) pc[1]--;
                wplane->setPlane(pc);
                break;
            case 4:
                pc = wplane->getPlane();
                if (pc[1] < ymax-1) pc[1]++;
                wplane->setPlane(pc);
                break;
            case 5:
                pc = wplane->getPlane();
                if (pc[2] > 1) pc[2]--;
                wplane->setPlane(pc);
                break;
            case 6:
                pc = wplane->getPlane();
                if (pc[2] < zmax-1) pc[2]++;
                wplane->setPlane(pc);
                break;
            }
        }
    }
    
    virtual void draw(const core::visual::VisualParams* vparams) override
    {
#ifndef SOFA_NO_OPENGL
        if (!vparams->displayFlags().getShowVisualModels() || display.getValue()==false) return;

        bool initialized=true;
        for(unsigned int i=0;i<3;i++) if(!cutplane_tex[i]) initialized=false;
        if(!initialized)
        {
            for(unsigned int i=0;i<3;i++)
            {
                cutplane_tex[i]= new helper::gl::Texture(new helper::io::Image,false);
                cutplane_tex[i]->getImage()->init(cutplane_res,cutplane_res,32);
            }
            updateTextures();

            for(unsigned int i=0;i<3;i++) cutplane_tex[i]->init();
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        }

        waPoints wpoints(this->points);
        waPlane wplane(this->plane);
        // Set input for wplane
        std::vector<VisuModelType*> visuals;
        sofa::core::objectmodel::BaseContext* context = this->getContext();
        context->get<VisuModelType>(&visuals,core::objectmodel::BaseContext::SearchRoot);
        wplane->setInput(image.getValue(),transform.getValue(),visuals);
        wplane->setTime( this->getContext()->getTime() );

        bool imagedirty=image.isDirty();
        if(imagedirty)
        {
            this->image.update();
            waHisto whisto(this->histo);
            // Set input for whisto
            whisto->setInput(image.getValue());
            whisto->update();
            wplane->setClamp(whisto->getClamp());
        }
        
        if(wplane->isImagePlaneDirty() || transform.isDirty() || imagedirty)
        {
            this->transform.update();
            this->image.update();
            wplane->setImagePlaneDirty(false);
            
            updateTextures();
        }
        
        if (wplane->isnewPointClicked())
        {
            raPlane rplane(this->plane);
            Coord point = rplane->getNewPoint();
            wpoints.push_back(point);
            wplane->setNewPointClicked(false);

            sofa::simulation::Node* root = dynamic_cast<simulation::Node*>(this->getContext());
            if(root)
            {
                sofa::component::visualmodel::RecordedCamera* currentCamera = root->getNodeObject<sofa::component::visualmodel::RecordedCamera>();
                if(currentCamera)
                {
                    currentCamera->m_translationPositions.setValue(this->points.getValue());
                }
            }
        }

        if(vectorVisualization.getValue().getShape())
        {
            if(wplane->getDimensions()[3] == 3)
                drawArrows(vparams);
            if(wplane->getDimensions()[3] == 6)
                drawEllipsoid();
        }
        glPushAttrib( GL_LIGHTING_BIT | GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
        drawCutplanes();
        glPopAttrib();
#endif //SOFA_NO_OPENGL
    }


    void getCorners(defaulttype::Vec<8,defaulttype::Vector3> &c) // get image corners
    {
        raImage rimage(this->image);
        const imCoord dim= rimage->getDimensions();

        defaulttype::Vec<8,defaulttype::Vector3> p;
        p[0]=defaulttype::Vector3(-0.5,-0.5,-0.5);
        p[1]=defaulttype::Vector3(dim[0]-0.5,-0.5,-0.5);
        p[2]=defaulttype::Vector3(-0.5,dim[1]-0.5,-0.5);
        p[3]=defaulttype::Vector3(dim[0]-0.5,dim[1]-0.5,-0.5);
        p[4]=defaulttype::Vector3(-0.5,-0.5,dim[2]-0.5);
        p[5]=defaulttype::Vector3(dim[0]-0.5,-0.5,dim[2]-0.5);
        p[6]=defaulttype::Vector3(-0.5,dim[1]-0.5,dim[2]-0.5);
        p[7]=defaulttype::Vector3(dim[0]-0.5,dim[1]-0.5,dim[2]-0.5);

        raTransform rtransform(this->transform);
        for(unsigned int i=0;i<p.size();i++) c[i]=rtransform->fromImage(p[i]);
    }

    virtual void computeBBox(const core::ExecParams*  params, bool /*onlyVisible=false*/ ) override
    {
        //        if( onlyVisible) return;
        defaulttype::Vec<8,defaulttype::Vector3> c;
        getCorners(c);

        Real bbmin[3]  = {c[0][0],c[0][1],c[0][2]} , bbmax[3]  = {c[0][0],c[0][1],c[0][2]};
        for(unsigned int i=1;i<c.size();i++)
            for(unsigned int j=0;j<3;j++)
            {
                if(bbmin[j]>c[i][j]) bbmin[j]=c[i][j];
                if(bbmax[j]<c[i][j]) bbmax[j]=c[i][j];
            }
        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(bbmin,bbmax));
    }
    
    
protected:
    
    static const unsigned cutplane_res=1024;

#ifndef SOFA_NO_OPENGL
    helper::gl::Texture* cutplane_tex[3];
#endif //SOFA_NO_OPENGL

    //Draw vectors as arrows
    void drawArrows(const core::visual::VisualParams* vparams)
    {
        raImage rimage(this->image);
        raPlane rplane(this->plane);
        raTransform rtransform(this->transform);
        raVis rVis(this->vectorVisualization);

        double size = rVis->getShapeScale();
        imCoord dims=rplane->getDimensions();
        defaulttype::Vec<3,int> sampling(rVis->getSubsampleXY(),rVis->getSubsampleXY(),rVis->getSubsampleZ());
        defaulttype::Vec4f colour(1.0,0.5,0.5,1.0);

        unsigned int x,y,z;
        for (z=0;z<3;z++)
        {
            if(z==0) { x=2;y=1;}
            else if(z==1) { x=0;y=2;}
            else { x=0;y=1;}
            Coord ip; ip[z]=rplane->getPlane()[z];
            if(ip[z]<dims[z] && dims[x]>1 && dims[y]>1)
                for(ip[x] = 0; ip[x] < dims[x]; ip[x] += sampling[x])
                    for(ip[y] = 0; ip[y] < dims[y]; ip[y] += sampling[y])
                    {
                        Coord base = rtransform->fromImage(ip);
                        cimg_library::CImg<T> vect = rimage->getCImg(rplane->getTime()).get_vector_at(ip[0],ip[1],ip[2]);
                        Coord relativeVec((double)vect[0], (double)vect[1], (double)vect[2]);
                        vparams->drawTool()->drawArrow(base,base+relativeVec*size,size*relativeVec.norm()/10,colour);
                    }
        }
    }

    //Draw tensors as ellipsoids
    void drawEllipsoid()
    {
#ifndef SOFA_NO_OPENGL
        raImage rimage(this->image);
        raPlane rplane(this->plane);
        raTransform rtransform(this->transform);
        raVis rVis(this->vectorVisualization);

        double size = rVis->getShapeScale();
        imCoord dims=rplane->getDimensions();
        defaulttype::Vec<3,int> sampling(rVis->getSubsampleXY(),rVis->getSubsampleXY(),rVis->getSubsampleZ());

        int counter=0;

        unsigned int x,y,z;
        for (z=0;z<3;z++)
        {
            if(z==0) { x=2;y=1;}
            else if(z==1) { x=0;y=2;}
            else { x=0;y=1;}
            Coord ip; ip[z]=rplane->getPlane()[z];
            if(ip[z]<dims[z] && dims[x]>1 && dims[y]>1)
                for(ip[x] = 0; ip[x] < dims[x]; ip[x] += sampling[x])
                    for(ip[y] = 0; ip[y] < dims[y]; ip[y] += sampling[y])
                    {

                        counter++ ;

                        Coord base = rtransform->fromImage(ip);

                        cimg_library::CImg<T> vector = rimage->getCImg(rplane->getTime()).get_vector_at(ip[0], ip[1], ip[2]);

                        //CImg::get_tensor_at() assumes a different tensor input than we expect.
                        // That is why we are generating the tensor manually from the vector instead.
                        cimg_library::CImg<T> tensor;

                        if(rVis->getTensorOrder().compare("LowerTriRowMajor") == 0)
                        {
                            tensor = computeTensorFromLowerTriRowMajorVector(vector);
                        }
                        else if(rVis->getTensorOrder().compare("UpperTriRowMajor") == 0)
                        {
                            tensor = computeTensorFromUpperTriRowMajorVector(vector);
                        }
                        else if(rVis->getTensorOrder().compare("DiagonalFirst") == 0)
                        {
                            tensor = computeTensorFromDiagonalFirstVector(vector);
                        }
                        else
                        {
                            //                                                    FF commented this out due to ambiguous overload for operator<< on gcc4.4
                            //                                                        serr << "ImageViewer: Tensor input order \"" << rVis->getTensorOrder() << "\" is not valid." << sout;
                            return;
                        }

                        cimg_library::CImgList<T> eig = tensor.get_symmetric_eigen();
                        const cimg_library::CImg<T> &val = eig[0];
                        const cimg_library::CImg<T> &vec = eig[1];

                        glPushMatrix();

                        GLUquadricObj* ellipsoid = gluNewQuadric();
                        glTranslated(base[0], base[1], base[2]);
                        GLdouble transformMatrix[16];

                        int index=0;
                        for(int i=0; i<vec.width(); i++)
                        {
                            for(int j=0; j<vec.height(); j++)
                            {
                                transformMatrix[index] = (double)vec(i,j);
                                index++;
                            }
                            transformMatrix[index] = 0;
                            index++;
                        }
                        transformMatrix[12] = transformMatrix[13] = transformMatrix[14] = 0;
                        transformMatrix[15] = 1;

                        //Same method for getting colours as MedInria
                        double colourR = fabs((double)vec(0,0));
                        double colourG = fabs((double)vec(0,1));
                        double colourB = fabs((double)vec(0,2));

                        colourR = (colourR > 1.0) ? 1.0 : colourR;
                        colourG = (colourG > 1.0) ? 1.0 : colourG;
                        colourB = (colourB > 1.0) ? 1.0 : colourB;

                        glColor3d(colourR, colourG, colourB);
                        glEnable(GL_COLOR_MATERIAL);

                        glMultMatrixd(transformMatrix);
                        glScaled((double)val(0)*size/10, (double)val(1)*size/10, (double)val(2)*size/10);
                        gluSphere(ellipsoid, 1.0, 10, 10);
                        gluDeleteQuadric(ellipsoid);

                        glDisable(GL_COLOR_MATERIAL);

                        glPopMatrix();

                    }
        }
#endif //SOFA_NO_OPENGL
    }

    cimg_library::CImg<T> computeTensorFromLowerTriRowMajorVector(cimg_library::CImg<T> vector)
    {
        cimg_library::CImg<T> tensor(3,3);
        tensor(0,0) = vector(0);
        tensor(1,0) = tensor(0,1) = vector(1);
        tensor(1,1) = vector(2);
        tensor(2,0) = tensor(0,2) = vector(3);
        tensor(2,1) = tensor(1,2) = vector(4);
        tensor(2,2) = vector(5);

        return tensor;
    }

    cimg_library::CImg<T> computeTensorFromUpperTriRowMajorVector(cimg_library::CImg<T> vector)
    {
        cimg_library::CImg<T> tensor(3,3);
        tensor(0,0) = vector(0);
        tensor(0,1) = tensor(1,0) = vector(1);
        tensor(0,2) = tensor(2,0) = vector(2);
        tensor(1,1) = vector(3);
        tensor(1,2) = tensor(2,1) = vector(4);
        tensor(2,2) = vector(5);

        return tensor;
    }

    cimg_library::CImg<T> computeTensorFromDiagonalFirstVector(cimg_library::CImg<T> vector)
    {
        cimg_library::CImg<T> tensor(3,3);
        tensor(0,0) = vector(0);
        tensor(1,1) = vector(1);
        tensor(2,2) = vector(2);
        tensor(1,0) = tensor(0,1) = vector(3);
        tensor(2,0) = tensor(0,2) = vector(4);
        tensor(2,1) = tensor(1,2) = vector(5);
        return tensor;
    }

    //Draw the boxes around the slices
    void drawCutplanes()
    {
#ifndef SOFA_NO_OPENGL
        raPlane rplane(this->plane);
        if (!rplane->getDimensions()[0]) return;

        this->transform.update();

        float color[]={1.,1.,1.,1.}, specular[]={0.,0.,0.,0.};
        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
        glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
        glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);
        glColor4fv(color);

        glDisable(GL_CULL_FACE);

        for (unsigned int i=0;i<3;i++)
            if(rplane->getPlane()[i]<rplane->getDimensions()[i])
                if(i==0 || rplane->getDimensions()[0]>1)
                    if(i==1 || rplane->getDimensions()[1]>1)
                        if(i==2 || rplane->getDimensions()[2]>1)
                        {
                            defaulttype::Vec<4,defaulttype::Vector3> pts = rplane->get_sliceCoord(rplane->getPlane()[i],i);
                            defaulttype::Vector3 n=cross(pts[1]-pts[0],pts[2]-pts[0]); n.normalize();

                            glEnable( GL_TEXTURE_2D );
                            glDisable( GL_LIGHTING);
                            cutplane_tex[i]->bind();
                            glBegin(GL_QUADS);
                            glNormal3d(n[0],n[1],n[2]);
                            glTexCoord2d(0,0); glVertex3d(pts[0][0],pts[0][1],pts[0][2]);
                            glTexCoord2d(1,0); glVertex3d(pts[1][0],pts[1][1],pts[1][2]);
                            glTexCoord2d(1,1); glVertex3d(pts[2][0],pts[2][1],pts[2][2]);
                            glTexCoord2d(0,1); glVertex3d(pts[3][0],pts[3][1],pts[3][2]);
                            glEnd ();
                            cutplane_tex[i]->unbind();
                            glDisable( GL_TEXTURE_2D );
                            glEnable( GL_LIGHTING);

                            //Outline
                            glLineWidth(2.0);
                            glBegin(GL_LINE_LOOP);
                            glNormal3d(n[0],n[1],n[2]);
                            glVertex3d(pts[0][0],pts[0][1],pts[0][2]);
                            glVertex3d(pts[1][0],pts[1][1],pts[1][2]);
                            glVertex3d(pts[2][0],pts[2][1],pts[2][2]);
                            glVertex3d(pts[3][0],pts[3][1],pts[3][2]);
                            glEnd ();

                        }
#endif //SOFA_NO_OPENGL
    }


    //Update and draw the slices
    void updateTextures()
    {
#ifndef SOFA_NO_OPENGL
        raPlane rplane(this->plane);
        if (!rplane->getDimensions()[0]) return;

        for (unsigned int i=0;i<3;i++)
        {
            cimg_library::CImg<unsigned char> cplane = convertToUC( rplane->get_slice(rplane->getPlane()[i],i).resize(cutplane_res,cutplane_res,1,-100,1).cut(rplane->getClamp()[0],rplane->getClamp()[1]) );

            if (showSlicedModels.getValue())
            {
              cimg_library::CImg<unsigned char> cmodelplane =
                  convertToUC(rplane->get_slicedModels(rplane->getPlane()[i], i)
                                  .resize(cutplane_res, cutplane_res, 1, -100, 1)
                                  .cut(rplane->getClamp()[0], rplane->getClamp()[1]));

              cimg_forXYC(cmodelplane, x, y, c)
              {
                if (cmodelplane(x, y, 0, c) == 0)
                  cmodelplane(x, y, 0, c) = (unsigned char)(cplane(x, y, 0, 0));
              }
              if (cmodelplane)
              {
                cimg_forXY(cmodelplane, x, y)
                {
                  unsigned char* b = cutplane_tex[i]->getImage()->getPixels() +
                                     4 * (y * cutplane_res + x);
                  for (unsigned int c = 0;
                       c < 3 && c < (unsigned int)cmodelplane.spectrum(); c++)
                    b[c] = cmodelplane(x, y, 0, c);
                  b[3] = (unsigned char)(-1);
                }
                cutplane_tex[i]->update();
              }
            }
            else if(cplane)
            {
                cimg_forXY(cplane,x,y)
                {
                    unsigned char *b=cutplane_tex[i]->getImage()->getPixels()+4*(y*cutplane_res+x);
                    for(unsigned int c=0; c<3 && c<(unsigned int)cplane.spectrum() ;c++) b[c]=cplane(x,y,0,c);
                    for(unsigned int c=cplane.spectrum(); c<3;c++) b[c]=b[0];
                    b[3]=(unsigned char)(-1);
                }
                cutplane_tex[i]->update();
            }
        }
#endif //SOFA_NO_OPENGL
    }

};


} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGEVIEWER_H
