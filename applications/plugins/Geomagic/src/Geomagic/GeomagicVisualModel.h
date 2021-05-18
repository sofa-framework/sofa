/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef SOFA_GEOMAGIC_VISUALMODEL_H
#define SOFA_GEOMAGIC_VISUALMODEL_H

//Geomagic include
#include <Geomagic/config.h>
#include <Geomagic/GeomagicDriver.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaLoader/MeshObjLoader.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

//Visualization
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/Node.h>

namespace sofa 
{

namespace component 
{

namespace controller
{

using namespace sofa::defaulttype;

/**
* Class used by GeomagicDriver to display the Geomagic device position and motion using visual models in the 3D scene.
*/
class SOFA_GEOMAGIC_API GeomagicVisualModel
{
public:
    typedef RigidTypes::Coord Coord;
    typedef RigidTypes::VecCoord VecCoord;

    struct VisualComponent
    {
        simulation::Node::SPtr node;
        sofa::component::loader::MeshObjLoader::SPtr loader;
        sofa::component::visualmodel::OglModel::SPtr visu;        
        sofa::component::mapping::RigidMapping< Rigid3Types , Vec3Types  >::SPtr mapping;
    };


    GeomagicVisualModel();
	virtual ~GeomagicVisualModel();

    /// Main Method to init the visual component tree of OGLModels. Called by Geomagic InitDevice() if drawVisual is on.
    void initDisplay(sofa::simulation::Node::SPtr node, const std::string& _deviceName, double _scale);

    /// Method to update the visualNode using the current device position and the angles of the different node of the device. Updated by Geomagic UpdatePosition()
    void updateDisplay(const GeomagicDriver::Coord& posDevice, HDdouble angle1[3], HDdouble angle2[3]);

    /// Method called by Geomagic Draw method to display the geomagic OglModel
    void drawDevice(bool button1Status = false, bool button2Status = false);

    /// Get status if visualisation is activated
    bool isDisplayActivated() const { return m_displayActived; }
    /// Activate or not the visualisation
    void activateDisplay(bool value);

    /// Get status if visualisation is init
    bool isDisplayInitiate() const { return m_initDisplayDone; }

protected:
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
    bool m_displayActived; ///< Internal boolean to detect activation switch of the draw
    bool m_initDisplayDone; ///< Internal boolean activated only if visu initialization done without return
    double m_scale;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_GEOMAGIC_VISUALMODEL_H
