/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
/*
 * ToolTracker.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_TOOLTRACKER_H_
#define SOFAVRPNCLIENT_TOOLTRACKER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/Quat.h>
#include <VRPNDevice.h>

#include <vrpn/vrpn_Analog.h>

namespace sofavrpn
{

namespace client
{

/*
 * Find a tool given various parameters...
 *
 */

template<class DataTypes>
class ToolTracker : public virtual sofa::core::objectmodel::BaseObject, public virtual sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ToolTracker, DataTypes), sofa::core::objectmodel::BaseObject);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef typename sofa::defaulttype::RigidTypes::Coord RPoint;
    typedef typename sofa::defaulttype::RigidTypes::Coord RCoord;

    //input
    sofa::core::objectmodel::Data<VecCoord > f_points;
    //distances between each point for the given tool
    sofa::core::objectmodel::Data<sofa::helper::vector<double> > f_distances;


    //output
    sofa::core::objectmodel::Data<Coord> f_center;
    sofa::core::objectmodel::Data<sofa::defaulttype::Quat> f_orientation;
    //the same...
    sofa::core::objectmodel::Data<RCoord> f_rigidCenter;

    //parameters
    sofa::core::objectmodel::Data<bool> f_drawTool;

    ToolTracker();
    virtual ~ToolTracker();

//	void init();
//	void reinit();
    void update();
    void draw();

private:

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFAVRPNCLIENT_TOOLTRACKER_CPP_)
#ifndef SOFA_FLOAT
extern template class SOFA_SOFAVRPNCLIENT_API ToolTracker<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_SOFAVRPNCLIENT_API ToolTracker<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_TOOLTRACKER_H_ */
