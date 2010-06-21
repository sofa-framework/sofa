/*
 * VRPNImager.h
 *
 *  Created on: 14 May 2010
 *      Author: peterlik
 */

#ifndef SOFAVRPNCLIENT_VRPNIMAGER_H_
#define SOFAVRPNCLIENT_VRPNIMAGER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/Event.h>

//#include <sofa/helper/RandomGenerator.h>
#include <sofa/core/DataEngine.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/Quater.h>


#include <VRPNDevice.h>

//#include <vrpn/vrpn_Connection.h>
//#include <vrpn/vrpn_FileConnection.h>
#include <vrpn/vrpn_Imager.h>


namespace sofavrpn
{

namespace client
{

struct VRPNImagerData
{

    //sofa::helper::vector<vrpn_TRACKERCB> data;
    bool got_dimensions;          //< Heard image dimensions from server?
    bool ready_for_region;        //< Everything set up to handle a region?
    bool already_posted;          //< Posted redisplay since the last display?
    int Xdim, Ydim;               //< Dimensions in X and Y
    vrpn_Imager_Remote *remote_imager;
    float rigidPointData[7];

    unsigned char *image;        //< Pointer to the storage for the image



    VRPNImagerData() :
        got_dimensions(false),
        ready_for_region(false),
        already_posted(false),
        image(NULL)
    {}
};

void  VRPN_CALLBACK handle_discarded_frames(void *, const vrpn_IMAGERDISCARDEDFRAMESCB info);
void  VRPN_CALLBACK handle_description_message(void *, const struct timeval);
void  VRPN_CALLBACK handle_region_change(void *userdata, const vrpn_IMAGERREGIONCB info);
void  VRPN_CALLBACK handle_end_of_frame(void *,const struct _vrpn_IMAGERENDFRAMECB);

template<class DataTypes>
class VRPNImager :  public virtual VRPNDevice
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VRPNImager, DataTypes), VRPNDevice);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef sofa::helper::Quater<Real> Quat;


    typedef typename DataTypes::VecCoord VecCoord;

    VRPNImager();
    virtual ~VRPNImager();

//	void init();
//	void reinit();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const VRPNImager<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Data<Point> rigidPoint;
    //Data<sofa::helper::vector<Vec3 > > f_positions;
    //Data<sofa::helper::vector<Quat> > f_orientations;
    Data<Point> f_rigidPoint;
    Data<Real>  f_scale;

private:
    vrpn_Imager_Remote      *g_imager;      //< Imager client object
    VRPNImagerData  imagerData;
    unsigned int imageTextureID;


    bool connectToServer();
    void update();
    void draw();

    void handleEvent(sofa::core::objectmodel::Event* event);
};

#if defined(WIN32) && !defined(SOFAVRPNCLIENT_VRPNIMAGER_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNImager<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNImager<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_VRPNIMAGER_H_ */
