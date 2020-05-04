/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GEOMAGIC_GEOMAGICVISUALMODEL_H
#define SOFA_GEOMAGIC_GEOMAGICVISUALMODEL_H

//Geomagic include
#include <Geomagic/config.h>
#include <SofaOpenglVisual/OglModel.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <Geomagic/GeomagicDriver.h>

namespace sofa 
{

namespace component 
{

namespace controller
{

//using namespace sofa::defaulttype;
//using core::objectmodel::Data;

#define NBJOINT 6
using namespace sofa::defaulttype;


class SOFA_GEOMAGIC_API GeomagicVisualModel
{

public:
    typedef RigidTypes::Coord Coord;
    typedef RigidTypes::VecCoord VecCoord;

    struct VisualComponent
    {
        simulation::Node::SPtr node;
        sofa::component::visualmodel::OglModel::SPtr visu;
        sofa::component::mapping::RigidMapping< Rigid3Types , Vec3Types  >::SPtr mapping;
    };


    GeomagicVisualModel();
	virtual ~GeomagicVisualModel();

    void initDisplay(sofa::simulation::Node::SPtr node, const std::string& _deviceName, double _scale);
    void updateVisulation(const GeomagicDriver::Coord& posDevice, HDdouble angle1[3], HDdouble angle2[3]);
    void drawDevice(bool button1Status = false, bool button2Status = false);


    bool isVisuActivated() { return m_visuActive; }
    bool isVisuInitiate() { return m_initVisuDone; }

    /// variable pour affichage graphique
    enum
    {
        VN_stylus = 0,
        VN_joint2 = 1,
        VN_joint1 = 2,
        VN_arm2 = 3,
        VN_arm1 = 4,
        VN_joint0 = 5,
        VN_base = 6,
        NVISUALNODE = 7
    };
    VisualComponent visualNode[NVISUALNODE];
    static const char* visualNodeNames[NVISUALNODE];
    static const char* visualNodeFiles[NVISUALNODE];
    simulation::Node::SPtr m_omniVisualNode;
    component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>::SPtr rigidDOF;

    VecCoord m_posDeviceVisu; ///< position of the hpatic devices for rendering. first pos is equal to d_posDevice

private:
    bool m_visuActive; ///< Internal boolean to detect activation switch of the draw
    bool m_initVisuDone; ///< Internal boolean activated only if visu initialization done without return
    double m_scale;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_GEOMAGIC_GEOMAGICVISUALMODEL_H
