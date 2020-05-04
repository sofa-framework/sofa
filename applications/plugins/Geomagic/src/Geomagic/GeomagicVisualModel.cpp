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

#include <Geomagic/GeomagicVisualModel.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>

namespace sofa
{

namespace component
{

namespace controller
{


const char* GeomagicVisualModel::visualNodeNames[NVISUALNODE] =
{
    "stylus",
    "joint 2",
    "joint 1",
    "arm 2",
    "arm 1",
    "joint 0",
    "base"
};
const char* GeomagicVisualModel::visualNodeFiles[NVISUALNODE] =
{
    "mesh/stylusO.obj",
    "mesh/articulation5O.obj",
    "mesh/articulation4O.obj",
    "mesh/articulation3O.obj",
    "mesh/articulation2O.obj",
    "mesh/articulation1O.obj",
    "mesh/BASEO.obj"
};


GeomagicVisualModel::GeomagicVisualModel()
{

}


GeomagicVisualModel::~GeomagicVisualModel()
{

}

void GeomagicVisualModel::initDisplay(sofa::simulation::Node::SPtr node, const std::string& _deviceName, double _scale)
{
    //Initialization of the visual components
    //resize vectors
    m_posDeviceVisu.resize(NVISUALNODE + 1);

    m_visuActive = false;
    m_scale = _scale;

    for (int i = 0; i<NVISUALNODE; i++)
    {
        visualNode[i].visu = NULL;
        visualNode[i].mapping = NULL;
    }

    //create a specific node containing rigid position for visual models
    //sofa::simulation::Node::SPtr rootContext = static_cast<simulation::Node*>(this->getContext()->getRootContext());
    //m_omniVisualNode = rootContext->createChild("omniVisu " + d_deviceName.getValue());
    m_omniVisualNode = node->createChild("omniVisu " + _deviceName);
    m_omniVisualNode->updateContext();

    rigidDOF = sofa::core::objectmodel::New<component::container::MechanicalObject<sofa::defaulttype::Rigid3Types> >();
    m_omniVisualNode->addObject(rigidDOF);
    rigidDOF->name.setValue("rigidDOF");

    VecCoord& posDOF = *(rigidDOF->x.beginEdit());
    posDOF.resize(NVISUALNODE + 1);
    rigidDOF->x.endEdit();

    rigidDOF->init();
    m_omniVisualNode->updateContext();


    //creation of subnodes for each part of the device visualization
    for (int i = 0; i<NVISUALNODE; i++)
    {
        visualNode[i].node = m_omniVisualNode->createChild(visualNodeNames[i]);

        if (visualNode[i].visu == NULL && visualNode[i].mapping == NULL)
        {

            // create the visual model and add it to the graph //
            visualNode[i].visu = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
            visualNode[i].node->addObject(visualNode[i].visu);
            visualNode[i].visu->name.setValue("VisualParticles");
            visualNode[i].visu->fileMesh.setValue(visualNodeFiles[i]);

            visualNode[i].visu->init();
            visualNode[i].visu->initVisual();
            visualNode[i].visu->updateVisual();

            // create the visual mapping and at it to the graph //
            visualNode[i].mapping = sofa::core::objectmodel::New< sofa::component::mapping::RigidMapping< Rigid3Types, Vec3Types > >();
            visualNode[i].node->addObject(visualNode[i].mapping);
            visualNode[i].mapping->setModels(rigidDOF.get(), visualNode[i].visu.get());
            visualNode[i].mapping->name.setValue("RigidMapping");
            visualNode[i].mapping->f_mapConstraints.setValue(false);
            visualNode[i].mapping->f_mapForces.setValue(false);
            visualNode[i].mapping->f_mapMasses.setValue(false);
            visualNode[i].mapping->index.setValue(i + 1);
            visualNode[i].mapping->init();
        }
        if (i<NVISUALNODE)
            m_omniVisualNode->removeChild(visualNode[i].node);
    }

    m_omniVisualNode->updateContext();

    for (int i = 0; i<NVISUALNODE; i++)
    {
        visualNode[i].node->updateContext();
    }

    m_initVisuDone = true;

    for (int j = 0; j<NVISUALNODE; j++)
    {
        sofa::helper::vector< sofa::defaulttype::Vec3 > &scaleMapping = *(visualNode[j].mapping->points.beginEdit());
        for (size_t i = 0; i<scaleMapping.size(); i++)
            scaleMapping[i] *= (float)(_scale);
        visualNode[j].mapping->points.endEdit();
    }
}


void GeomagicVisualModel::updateVisulation(const GeomagicDriver::Coord& posDevice, HDdouble angle1[3], HDdouble angle2[3])
{
    sofa::defaulttype::SolidTypes<double>::Transform tampon;
    m_posDeviceVisu[0] = posDevice;
    tampon.set(m_posDeviceVisu[0].getCenter(), m_posDeviceVisu[0].getOrientation());

    //get position stylus
    m_posDeviceVisu[1 + VN_stylus] = Coord(tampon.getOrigin(), tampon.getOrientation());

    //get pos joint 2
    sofa::helper::Quater<double> quarter2(Vec3d(0.0, 0.0, 1.0), angle2[2]);
    sofa::defaulttype::SolidTypes<double>::Transform transform_segr2(Vec3d(0.0, 0.0, 0.0), quarter2);
    tampon *= transform_segr2;
    m_posDeviceVisu[1 + VN_joint2] = Coord(tampon.getOrigin(), tampon.getOrientation());

    //get pos joint 1
    sofa::helper::Quater<double> quarter3(Vec3d(1.0, 0.0, 0.0), angle2[1]);
    sofa::defaulttype::SolidTypes<double>::Transform transform_segr3(Vec3d(0.0, 0.0, 0.0), quarter3);
    tampon *= transform_segr3;
    m_posDeviceVisu[1 + VN_joint1] = Coord(tampon.getOrigin(), tampon.getOrientation());

    //get pos arm 2
    sofa::helper::Quater<double> quarter4(Vec3d(0.0, 1.0, 0.0), -angle2[0]);
    sofa::defaulttype::SolidTypes<double>::Transform transform_segr4(Vec3d(0.0, 0.0, 0.0), quarter4);
    tampon *= transform_segr4;
    m_posDeviceVisu[1 + VN_arm2] = Coord(tampon.getOrigin(), tampon.getOrientation());
    //get pos arm 1
    sofa::helper::Quater<double> quarter5(Vec3d(1.0, 0.0, 0.0), -(M_PI / 2) + angle1[2] - angle1[1]);
    sofa::defaulttype::SolidTypes<double>::Transform transform_segr5(Vec3d(0.0, 13.33*m_scale, 0.0), quarter5);
    tampon *= transform_segr5;
    m_posDeviceVisu[1 + VN_arm1] = Coord(tampon.getOrigin(), tampon.getOrientation());

    //get pos joint 0
    sofa::helper::Quater<double> quarter6(Vec3d(1.0, 0.0, 0.0), angle1[1]);
    sofa::defaulttype::SolidTypes<double>::Transform transform_segr6(Vec3d(0.0, 13.33*m_scale, 0.0), quarter6);
    tampon *= transform_segr6;
    m_posDeviceVisu[1 + VN_joint0] = Coord(tampon.getOrigin(), tampon.getOrientation());

    //get pos base
    sofa::helper::Quater<double> quarter7(Vec3d(0.0, 0.0, 1.0), angle1[0]);
    sofa::defaulttype::SolidTypes<double>::Transform transform_segr7(Vec3d(0.0, 0.0, 0.0), quarter7);
    tampon *= transform_segr7;
    m_posDeviceVisu[1 + VN_base] = Coord(tampon.getOrigin(), tampon.getOrientation());

    // update the omni visual node positions through the mappings
    if (m_omniVisualNode)
    {
        sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(m_omniVisualNode.get());
        sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(m_omniVisualNode.get());
    }
}



void GeomagicVisualModel::drawDevice(bool button1Status, bool button2Status)
{
    if (!m_initVisuDone)
        return;

    //Reactivate visual node
    if (!m_visuActive)
    {
        m_visuActive = true;

        for (int i = 0; i<NVISUALNODE; i++)
        {
            m_omniVisualNode->addChild(visualNode[i].node);
            visualNode[i].node->updateContext();
        }
        m_omniVisualNode->updateContext();
    }

    VecCoord& posDOF = *(rigidDOF->x.beginEdit());
    posDOF.resize(m_posDeviceVisu.size());
    for (int i = 0; i<NVISUALNODE + 1; i++)
    {
        posDOF[i].getCenter() = m_posDeviceVisu[i].getCenter();
        posDOF[i].getOrientation() = m_posDeviceVisu[i].getOrientation();
    }

    //if buttons pressed, change stylus color
    std::string color = "grey";
    if (button1Status)
    {
        if (button2Status)
        {
            color = "yellow";
        }
        else
        {
            color = "blue";
        }
    }
    else if (button2Status)
    {
        color = "red";
    }
    visualNode[0].visu->setColor(color);

    rigidDOF->x.endEdit();    
}


} // namespace controller

} // namespace component

} // namespace sofa
