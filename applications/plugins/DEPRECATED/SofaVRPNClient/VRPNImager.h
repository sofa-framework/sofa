/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
    sofa::core::objectmodel::Data<Point> f_rigidPoint; ///< RigidPoint
    sofa::core::objectmodel::Data<Real>  f_scale; ///< Scale

private:
    vrpn_Imager_Remote      *g_imager;      //< Imager client object
    VRPNImagerData  imagerData;
    unsigned int imageTextureID;


    bool connectToServer();
    void update();
    void draw();

    void handleEvent(sofa::core::objectmodel::Event* event);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFAVRPNCLIENT_VRPNIMAGER_CPP_)
#ifndef SOFA_FLOAT
extern template class SOFA_SOFAVRPNCLIENT_API VRPNImager<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_SOFAVRPNCLIENT_API VRPNImager<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_VRPNIMAGER_H_ */
