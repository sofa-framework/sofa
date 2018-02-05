/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef IMAGE_KINECT_H
#define IMAGE_KINECT_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

#ifdef WIN32
#include <process.h>
#else
#include <pthread.h>
#endif

#include <libfreenect.h>
//#include <libfreenect_sync.h>
//#include <libfreenect-registration.h>


namespace sofa
{

namespace component
{

namespace container
{

using namespace cimg_library;
using defaulttype::Vec;
using defaulttype::Vector3;

void* globalKinectClassPointer;

#ifdef WIN32
#else
pthread_mutex_t backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
//pthread_cond_t frame_cond = PTHREAD_COND_INITIALIZER;
#endif


class Kinect : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS( Kinect , Inherited);

    // image data
    typedef defaulttype::ImageUC ImageTypes;
    typedef ImageTypes::T T;
    typedef ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    // transform data
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    // depth data
    typedef defaulttype::ImageUS DepthTypes;
    typedef DepthTypes::T dT;
    typedef DepthTypes::imCoord dCoord;
    typedef helper::WriteAccessor<Data< DepthTypes > > waDepth;
    typedef helper::ReadAccessor<Data< DepthTypes > > raDepth;
    Data< DepthTypes > depthImage;
    Data< TransformType > depthTransform;

    Data<unsigned int> deviceID;
    Data<helper::OptionsGroup> resolution;
    Data<helper::OptionsGroup> videoMode;
    Data<helper::OptionsGroup> depthMode;
    Data<helper::OptionsGroup> ledMode;
    Data<int> tiltAngle;
    Data<defaulttype::Vector3> accelerometer;
    Data<bool> drawBB;
    Data<bool> drawGravity;
    Data<float> showArrowSize;


    virtual std::string getTemplateName() const	{ return templateName(this); }
    static std::string templateName(const Kinect* = NULL) {	return std::string(); }

    Kinect() : Inherited()
        , image(initData(&image,ImageTypes(),"image","image"))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , depthImage(initData(&depthImage,DepthTypes(),"depthImage","depth map"))
        , depthTransform(initData(&depthTransform, TransformType(), "depthTransform" , ""))
        , deviceID ( initData ( &deviceID,(unsigned int)0,"deviceID","device ID" ) )
        , resolution ( initData ( &resolution,"resolution","resolution" ) )
        , videoMode ( initData ( &videoMode,"videoMode","video mode" ) )
        , depthMode ( initData ( &depthMode,"depthMode","depth mode" ) )
        , ledMode ( initData ( &ledMode,"ledMode","led mode" ) )
        , tiltAngle(initData(&tiltAngle,0,"tiltAngle","tilt angle in [-30,30]"))
        , accelerometer(initData(&accelerometer,Vector3(0,0,0),"accelerometer","Accelerometer data"))
        , drawBB(initData(&drawBB,true,"drawBB","draw bounding box"))
        , drawGravity(initData(&drawGravity,true,"drawGravity","draw acceleration"))
        , showArrowSize(initData(&showArrowSize,0.1f,"showArrowSize","size of the axis"))
        , die(0)
        , got_rgb(0)
        , got_depth(0)
    {
        globalKinectClassPointer = (void*) this; // used for kinect callbacks

        this->addAlias(&image, "inputImage");
        this->addAlias(&transform, "inputTransform");
        transform.setGroup("Transform");
        depthTransform.setGroup("Transform");
        f_listening.setValue(true);  // to update camera during animate

        helper::OptionsGroup opt1(3 ,"320x240" ,"640x480" ,"1280x1024" );
        opt1.setSelectedItem(1);
        resolution.setValue(opt1);

        helper::OptionsGroup opt2(4 ,"RGB" ,"IR_8bits" ,"YUV_RGB" ,"YUV_RAW");
        opt2.setSelectedItem(0);
        videoMode.setValue(opt2);

        helper::OptionsGroup opt3(2 ,"Raw" ,"Registered" );
        opt3.setSelectedItem(1);
        depthMode.setValue(opt3);

        helper::OptionsGroup opt4(6 ,"Off" ,"Green" ,"Red" ,"Yellow" ,"Blink Green" ,"Blink Yellow");
        opt4.setSelectedItem(1);
        ledMode.setValue(opt4);

        accelerometer.setReadOnly(true);
    }


    virtual void clear()
    {
        waImage wimage(this->image);
        wimage->clear();
        waDepth wdepth(this->depthImage);
        wdepth->clear();
    }

    virtual ~Kinect()
    {
        clear();
        die = 1;
#ifdef WIN32
        WaitForSingleObject(freenect_thread, INFINITE);
#else
        pthread_join(freenect_thread, NULL);
#endif
        free(depth_mid);
        free(depth_back);
        free(rgb_mid);
        free(rgb_back);
    }

    // mutex functions
    void mutex_lock()
    {
#ifdef WIN32
        WaitForSingleObject(backbuf_mutex, INFINITE);
#else
        //                    pthread_cond_wait(&frame_cond, &backbuf_mutex);
        pthread_mutex_lock(&backbuf_mutex);
#endif
    }

    void mutex_unlock()
    {
#ifdef WIN32
        ReleaseMutex(backbuf_mutex);
#else
        //        pthread_cond_signal(&frame_cond);
        pthread_mutex_unlock(&backbuf_mutex);
#endif
    }

    // callbacks with static wrappers
    static void _depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)    { reinterpret_cast<sofa::component::container::Kinect*>(globalKinectClassPointer)->depth_cb(dev, v_depth, timestamp);  }
    void depth_cb(freenect_device *dev, void *v_depth, uint32_t /*timestamp*/)
    {
        mutex_lock();
        // swap buffers
        depth_back = depth_mid;
        freenect_set_depth_buffer(dev, depth_back);
        depth_mid = (unsigned char*)v_depth;
        got_depth++;
        mutex_unlock();
    }

    static void _rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp)    { reinterpret_cast<sofa::component::container::Kinect*>(globalKinectClassPointer)->rgb_cb(dev, rgb, timestamp);  }
    void rgb_cb(freenect_device *dev, void *rgb, uint32_t /*timestamp*/)
    {
        mutex_lock();
        // swap buffers
        rgb_back = rgb_mid;
        freenect_set_video_buffer(dev, rgb_back);
        rgb_mid = (unsigned char*)rgb;
        got_rgb++;
        mutex_unlock();
    }

    virtual void init()
    {
        // convert sofa data into kinect params
        reinit();
        backup_res=res;
        backup_vf=vf;
        backup_df=df;
        backup_led=led;
        backup_tiltangle=tiltAngle.getValue();

        // init context and device
        if (freenect_init(&f_ctx, NULL) < 0) { serr<<"freenect_init() failed"<<sendl; return ; }
        //freenect_set_log_level(f_ctx, FREENECT_LOG_DEBUG);
        int nr_devices = freenect_num_devices (f_ctx); sout<<"Number of devices found: "<<nr_devices<<sendl;
        if (nr_devices < 1) {serr<<"No Kinect found"<<sendl; return; }
        int user_device_number = (int)this->deviceID.getValue();
        if (freenect_open_device(f_ctx, &f_dev, user_device_number) < 0) { serr<<"Could not open device "<<user_device_number<<sendl; return; }

        // allocate buffers with maximum posible resolution (used in kinect thread)
        depth_back = (unsigned char*)malloc(640*480*3);
        depth_mid = (unsigned char*)malloc(640*480*3);
        rgb_back = (unsigned char*)malloc(1280*1024*3);
        rgb_mid = (unsigned char*)malloc(1280*1024*3);

        // set kinect params
        freenect_set_tilt_degs(f_dev,tiltAngle.getValue());
        freenect_set_led(f_dev,led);
        freenect_set_depth_callback(f_dev, _depth_cb);
        freenect_set_video_callback(f_dev, _rgb_cb);
        freenect_set_video_mode(f_dev, freenect_find_video_mode(res, vf));
        freenect_set_depth_mode(f_dev, freenect_find_depth_mode(res, df));
        freenect_set_video_buffer(f_dev, rgb_back);
        freenect_set_depth_buffer(f_dev, depth_back);

        // allocate image data (sofa thread)
        waImage wimage(this->image);
        waTransform wt(this->transform);
        waDepth wdepth(this->depthImage);
        waTransform wdt(this->depthTransform);

        if(wimage->isEmpty()) wimage->getCImgList().push_back(CImg<T>());
        CImg<T>& rgbimg=wimage->getCImg(0);
        if(vf==FREENECT_VIDEO_IR_8BIT && res==FREENECT_RESOLUTION_MEDIUM) rgbimg.resize(640,488,1,1);
        else if(vf==FREENECT_VIDEO_IR_8BIT && res==FREENECT_RESOLUTION_HIGH) rgbimg.resize(1280,1024,1,1);
        else if(res==FREENECT_RESOLUTION_LOW) rgbimg.resize(320,240,1,3);
        else if(res==FREENECT_RESOLUTION_HIGH) rgbimg.resize(1280,1024,1,3);
        else rgbimg.resize(640,480,1,3);

        wt->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
        wt->update(); // update of internal data

        if(wdepth->isEmpty()) wdepth->getCImgList().push_back(CImg<dT>());
        CImg<dT>& depthimg=wdepth->getCImg(0);
        depthimg.resize(640,480,1,1);

        wdt->setCamPos((Real)(wdepth->getDimensions()[0]-1)/2.0,(Real)(wdepth->getDimensions()[1]-1)/2.0); // for perspective transforms
        wdt->update(); // update of internal data

        // run kinect thread
#ifdef WIN32
        freenect_thread =_beginthread( sofa::component::container::Kinect::_freenect_threadfunc, 0, this);
#else
        pthread_create( &freenect_thread, NULL, sofa::component::container::Kinect::_freenect_threadfunc, this);
#endif

        loadCamera();
    }

    virtual void reinit()
    {
        // convert sofa data into kinect params
        res=(freenect_resolution)this->resolution.getValue().getSelectedId();

        if(this->depthMode.getValue().getSelectedId()==0) df=FREENECT_DEPTH_11BIT;
        else df=FREENECT_DEPTH_REGISTERED;

        if(this->videoMode.getValue().getSelectedId()==0) vf=FREENECT_VIDEO_RGB;
        else if(this->videoMode.getValue().getSelectedId()==1) vf=FREENECT_VIDEO_IR_8BIT;
        else if(this->videoMode.getValue().getSelectedId()==2) vf=FREENECT_VIDEO_YUV_RGB;
        else vf=FREENECT_VIDEO_YUV_RAW;

        led=(freenect_led_options)this->ledMode.getValue().getSelectedId();
        if(led==5) led=LED_BLINK_RED_YELLOW;

        if(tiltAngle.getValue()<-30) tiltAngle.setValue(-30);
        else if(tiltAngle.getValue()>30) tiltAngle.setValue(30);
    }


#ifdef WIN32
    uintptr_t freenect_thread;
    static void freenect_threadfunc(void *arg) { reinterpret_cast<sofa::component::container::Kinect*>(arg)->freenect_threadfunc(); }
#else
    pthread_t freenect_thread;
    static void* _freenect_threadfunc (void *arg) { reinterpret_cast<sofa::component::container::Kinect*>(arg)->freenect_threadfunc(); return NULL; }
#endif


    void freenect_threadfunc()
    {
        freenect_start_depth(f_dev);
        freenect_start_video(f_dev);

        while (!die && freenect_process_events(f_ctx) >= 0)
        {
            if (vf!=backup_vf || res!=backup_res)
            {
                backup_res=res;  backup_vf=vf;

                waImage wimage(this->image);
                waTransform wt(this->transform);

                if(!wimage->isEmpty())
                {
                    CImg<T>& rgbimg=wimage->getCImg(0);
                    if(vf==FREENECT_VIDEO_IR_8BIT && res==FREENECT_RESOLUTION_MEDIUM) rgbimg.resize(640,488,1,1);
                    else if(vf==FREENECT_VIDEO_IR_8BIT && res==FREENECT_RESOLUTION_HIGH) rgbimg.resize(1280,1024,1,1);
                    else if(res==FREENECT_RESOLUTION_LOW) rgbimg.resize(320,240,1,3);
                    else if(res==FREENECT_RESOLUTION_HIGH) rgbimg.resize(1280,1024,1,3);
                    else rgbimg.resize(640,480,1,3);
                    wt->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
                    wt->update(); // update of internal data
                }

                freenect_stop_video(f_dev);
                freenect_set_video_mode(f_dev, freenect_find_video_mode(res, vf));
                freenect_start_video(f_dev);
            }
            if (df!=backup_df)
            {
                backup_df=df;
                freenect_stop_depth(f_dev);
                freenect_set_depth_mode(f_dev, freenect_find_depth_mode(res, df));
                freenect_start_depth(f_dev);
            }
            if (led!=backup_led)
            {
                backup_led=led;
                freenect_set_led(f_dev,led);
            }
            int angle=tiltAngle.getValue();
            if (angle!=backup_tiltangle)
            {
                backup_tiltangle=angle;
                freenect_set_tilt_degs(f_dev,angle);
            }
        }

        freenect_stop_depth(f_dev);
        freenect_stop_video(f_dev);
        freenect_close_device(f_dev);
        freenect_shutdown(f_ctx);
    }


protected:

    // to kill kinect thread
    int die;

    // buffers for kinect thread
    unsigned char *rgb_mid, *rgb_back,*depth_mid, *depth_back;

    // kinect params
    freenect_context *f_ctx;
    freenect_device *f_dev;
    freenect_resolution res,backup_res;
    freenect_depth_format df,backup_df;
    freenect_video_format vf,backup_vf;
    freenect_led_options led,backup_led;
    int backup_tiltangle;

    // flags to know if images are ready
    int got_rgb ;
    int got_depth ;

    // copy video buffers to image Data (done at init and at each simulation step)
    void loadCamera()
    {
        if (vf==backup_vf && df==backup_df && res==backup_res) // wait for resolution update in kinect thread
        {
            if (got_depth)
            {
                got_depth = 0;

                waDepth wdepth(this->depthImage);
                waTransform wdt(this->depthTransform);

                if(!wdepth->isEmpty())
                {
                    CImg<dT>& depth=wdepth->getCImg(0);
                    mutex_lock();
                    memcpy(depth.data(),  (unsigned short*)depth_mid , depth.width()*depth.height()*sizeof(unsigned short));
                    mutex_unlock();
                }
            }
            if (got_rgb)
            {
                got_rgb = 0;
                waImage wimage(this->image);
                waTransform wt(this->transform);

                if(!wimage->isEmpty())
                {
                    CImg<T>& rgbimg=wimage->getCImg(0);
                    mutex_lock();
                    if(rgbimg.spectrum()==3)  // deinterlace
                    {
                        unsigned char* rgb = (unsigned char*)rgb_mid;
                        unsigned char *ptr_r = rgbimg.data(0,0,0,0), *ptr_g = rgbimg.data(0,0,0,1), *ptr_b = rgbimg.data(0,0,0,2);
                        for ( int siz = 0 ; siz<rgbimg.width()*rgbimg.height(); siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++); }
                    }
                    else memcpy(rgbimg.data(),  rgb_mid, rgbimg.width()*rgbimg.height()*sizeof(T));
                    mutex_unlock();
                }
            }

            // update accelerometer data
            freenect_update_tilt_state(f_dev);
            freenect_raw_tilt_state* state = freenect_get_tilt_state(f_dev);
            double dx,dy,dz;
            freenect_get_mks_accel(state, &dx, &dy, &dz);
            this->accelerometer.setValue(Vector3(dx,dy,dz));
        }
    }



    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if (dynamic_cast<simulation::AnimateEndEvent*>(event)) loadCamera();
    }


    void getCorners(Vec<8,Vector3> &c) // get image corners
    {
        raDepth rimage(this->depthImage);
        const imCoord dim= rimage->getDimensions();

        Vec<8,Vector3> p;
        p[0]=Vector3(-0.5,-0.5,-0.5);
        p[1]=Vector3(dim[0]-0.5,-0.5,-0.5);
        p[2]=Vector3(-0.5,dim[1]-0.5,-0.5);
        p[3]=Vector3(dim[0]-0.5,dim[1]-0.5,-0.5);
        p[4]=Vector3(-0.5,-0.5,dim[2]-0.5);
        p[5]=Vector3(dim[0]-0.5,-0.5,dim[2]-0.5);
        p[6]=Vector3(-0.5,dim[1]-0.5,dim[2]-0.5);
        p[7]=Vector3(dim[0]-0.5,dim[1]-0.5,dim[2]-0.5);

        raTransform rtransform(this->depthTransform);
        for(unsigned int i=0; i<p.size(); i++) c[i]=rtransform->fromImage(p[i]);
    }

    virtual void computeBBox(const core::ExecParams*  params )
    {
        if (!drawBB.getValue()) return;
        Vec<8,Vector3> c;
        getCorners(c);

        Real bbmin[3]  = {c[0][0],c[0][1],c[0][2]} , bbmax[3]  = {c[0][0],c[0][1],c[0][2]};
        for(unsigned int i=1; i<c.size(); i++)
            for(unsigned int j=0; j<3; j++)
            {
                if(bbmin[j]>c[i][j]) bbmin[j]=c[i][j];
                if(bbmax[j]<c[i][j]) bbmax[j]=c[i][j];
            }
        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(bbmin,bbmax));
    }

    void draw(const core::visual::VisualParams* vparams)
    {
        // draw bounding box

        if (!vparams->displayFlags().getShowVisualModels()) return;
        if (!drawBB.getValue() && !drawGravity.getValue()) return;

        glPushAttrib( GL_LIGHTING_BIT | GL_ENABLE_BIT | GL_LINE_BIT );
        glPushMatrix();

        if (drawBB.getValue())
        {
            const float color[]= {1.,0.5,0.5,0.}, specular[]= {0.,0.,0.,0.};
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
            glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
            glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);
            glColor4fv(color);
            glLineWidth(2.0);

            Vec<8,Vector3> c;
            getCorners(c);
            glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);  glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[2][0],c[2][1],c[2][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();
        }

        if(drawGravity.getValue())
        {
            const Vec<4,float> col(0,1,0,1);
            raTransform rtransform(this->depthTransform);
            Vector3 camCenter = rtransform->fromImage(Vector3(-0.5,-0.5,-0.5));
            Vector3 acc = rtransform->qrotation.rotate(this->accelerometer.getValue());
            vparams->drawTool()->drawArrow(camCenter, camCenter+acc*showArrowSize.getValue(), showArrowSize.getValue()*0.1, col);
        }

        glPopMatrix ();
        glPopAttrib();
    }






};






}

}

}


#endif /*IMAGE_Kinect_H*/
