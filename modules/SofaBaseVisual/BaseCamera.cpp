/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaBaseVisual/BaseCamera.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/helper/gl/Axis.h>

#ifndef TIXML_USE_STL
#define TIXML_USE_STL
#endif
#include <tinyxml.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

BaseCamera::BaseCamera()
    :p_position(initData(&p_position, "position", "Camera's position"))
    ,p_orientation(initData(&p_orientation, "orientation", "Camera's orientation"))
    ,p_lookAt(initData(&p_lookAt, "lookAt", "Camera's look at"))
    ,p_distance(initData(&p_distance, "distance", "Distance between camera and look at"))
    ,p_fieldOfView(initData(&p_fieldOfView, (double) (45.0) , "fieldOfView", "Camera's FOV"))
    ,p_zNear(initData(&p_zNear, (double) 0.0 , "zNear", "Camera's zNear (value <= 0.0 == computed from bounding box)"))
    ,p_zFar(initData(&p_zFar, (double) 0.0 , "zFar", "Camera's zFar (value <= 0.0 == computed from bounding box)"))
    ,p_minBBox(initData(&p_minBBox, Vec3(0.0,0.0,0.0) , "minBBox", "minBBox"))
    ,p_maxBBox(initData(&p_maxBBox, Vec3(1.0,1.0,1.0) , "maxBBox", "maaxBBox"))
    ,p_widthViewport(initData(&p_widthViewport, (unsigned int) 800 , "widthViewport", "widthViewport"))
    ,p_heightViewport(initData(&p_heightViewport,(unsigned int) 600 , "heightViewport", "heightViewport"))
    ,p_type(initData(&p_type, (int) core::visual::VisualParams::PERSPECTIVE_TYPE, "projectionType", "Camera Type (0 = Perspective, 1 = Orthographic)"))
    ,p_activated(initData(&p_activated, true , "activated", "Camera activated ?"))
	,p_fixedLookAtPoint(initData(&p_fixedLookAtPoint, false, "fixedLookAt", "keep the lookAt point always fixed"))
{

}

BaseCamera::~BaseCamera()
{

}

void BaseCamera::activate()
{
    p_activated.setValue(true);
}

void BaseCamera::desactivate()
{
    p_activated.setValue(false);
}

bool BaseCamera::isActivated()
{
    return p_activated.getValue();
}

void BaseCamera::init()
{

}

void BaseCamera::bwdInit()
{

    if(p_position.isSet())
    {
        if(!p_orientation.isSet())
        {
            p_distance.setValue((p_lookAt.getValue() - p_position.getValue()).norm());

            Quat q  = getOrientationFromLookAt(p_position.getValue(), p_lookAt.getValue());
            p_orientation.setValue(q);
        }
        else if(!p_lookAt.isSet())
        {
            //distance assumed to be set
            if(!p_distance.isSet())
                sout << "Missing distance parameter ; taking default value (0.0, 0.0, 0.0)" << sendl;

            Vec3 lookat = getLookAtFromOrientation(p_position.getValue(), p_distance.getValue(), p_orientation.getValue());
            p_lookAt.setValue(lookat);
        }
        else
        {
            serr << "Too many missing parameters ; taking default ..." << sendl;
        }
    }
    else
    {
        if(p_lookAt.isSet() && p_orientation.isSet())
        {
            //distance assumed to be set
            if(!p_distance.isSet())
                sout << "Missing distance parameter ; taking default value (0.0, 0.0, 0.0)" << sendl;

            Vec3 pos = getPositionFromOrientation(p_lookAt.getValue(), p_distance.getValue(), p_orientation.getValue());
            p_position.setValue(pos);
        }
        else
        {
            serr << "Too many missing parameters ; taking default ..." << sendl;
        }
    }
    currentDistance = p_distance.getValue();
    currentZNear = p_zNear.getValue();
    currentZFar = p_zFar.getValue();
    p_minBBox.setValue(getContext()->f_bbox.getValue().minBBox());
    p_maxBBox.setValue(getContext()->f_bbox.getValue().maxBBox());


}

void BaseCamera::translate(const Vec3& t)
{
    Vec3 &pos = *p_position.beginEdit();
    pos += t;
    p_position.endEdit();

}

void BaseCamera::translateLookAt(const Vec3& t)
{
    Vec3 &lookat = *p_lookAt.beginEdit();
    lookat += t;
    currentLookAt = lookat;
    p_lookAt.endEdit();

}

void BaseCamera::rotate(const Quat& r)
{
    Quat &rot = *p_orientation.beginEdit();
    rot = rot * r;
    rot.normalize();
    p_orientation.endEdit();
}

void BaseCamera::moveCamera(const Vec3 &p, const Quat &q)
{
    translate(p);
	if ( !p_fixedLookAtPoint.getValue() )
	{
		translateLookAt(p);
	}
    rotate(q);
}

BaseCamera::Vec3 BaseCamera::cameraToWorldCoordinates(const Vec3& p)
{
    return p_orientation.getValue().rotate(p) + p_position.getValue();
}

BaseCamera::Vec3 BaseCamera::worldToCameraCoordinates(const Vec3& p)
{
    return p_orientation.getValue().inverseRotate(p - p_position.getValue());
}

BaseCamera::Vec3 BaseCamera::cameraToWorldTransform(const Vec3& v)
{
    Quat q = p_orientation.getValue();
    return q.rotate(v) ;
}

BaseCamera::Vec3 BaseCamera::worldToCameraTransform(const Vec3& v)
{
    return p_orientation.getValue().inverseRotate(v);
}

BaseCamera::Vec3 BaseCamera::screenToWorldCoordinates(int x, int y)
{
#ifndef SOFA_NO_OPENGL
    GLint viewport[4];
    GLdouble modelview[16];
    GLdouble projection[16];
    GLfloat winX, winY, winZ;
    GLdouble posX, posY, posZ;

    glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
    glGetDoublev( GL_PROJECTION_MATRIX, projection );
    glGetIntegerv( GL_VIEWPORT, viewport );

    winX = (float)x;
    winY = (float)viewport[3] - (float)y;
    glReadPixels( x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ );
    //winZ = 1.0;

    gluUnProject( winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

    return Vec3(posX, posY, posZ);
#else
	return Vec3(0,0,0);
#endif /* SOFA_NO_OPENGL */
}

void BaseCamera::getOpenGLMatrix(GLdouble mat[16])
{
    defaulttype::SolidTypes<SReal>::Transform world_H_cam(p_position.getValue(), this->getOrientation());
    world_H_cam.inversed().writeOpenGlMatrix(mat);
}


void BaseCamera::reinit()
{
    //Data "LookAt" has changed
    //-> Orientation needs to be updated
    if(currentLookAt !=  p_lookAt.getValue())
    {
        Quat newOrientation = getOrientationFromLookAt(p_position.getValue(), p_lookAt.getValue());
        p_orientation.setValue(newOrientation);

        currentLookAt = p_lookAt.getValue();
    }
}

BaseCamera::Quat BaseCamera::getOrientationFromLookAt(const BaseCamera::Vec3 &pos, const BaseCamera::Vec3& lookat)
{
    Vec3 zAxis = -(lookat - pos);
    zAxis.normalize();

    Vec3 yAxis = cameraToWorldTransform(Vec3(0,1,0));

    Vec3 xAxis = yAxis.cross(zAxis) ;
    xAxis.normalize();

    //std::cout << xAxis.norm2() << std::endl;
    if (xAxis.norm2() < 0.00001)
        xAxis = cameraToWorldTransform(Vec3(1.0, 0.0, 0.0));
    xAxis.normalize();

    yAxis = zAxis.cross(xAxis);

    Quat q;
    q = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
    q.normalize();
    return q;
}


BaseCamera::Vec3 BaseCamera::getLookAtFromOrientation(const BaseCamera::Vec3 &pos, const double &distance, const BaseCamera::Quat & orientation)
{
    Vec3 zWorld = orientation.rotate(Vec3(0,0,-1*distance));
    return zWorld+pos;
}

BaseCamera::Vec3 BaseCamera::getPositionFromOrientation(const BaseCamera::Vec3 &lookAt, const double &distance, const BaseCamera::Quat& orientation)
{
    Vec3 zWorld = orientation.rotate(Vec3(0,0,-1*distance));
    return zWorld-lookAt;
}

void BaseCamera::rotateCameraAroundPoint(Quat& rotation, const Vec3& point)
{
    Vec3 tempAxis;
    SReal tempAngle;
    Quat orientation = this->getOrientation();
    Vec3& position = *p_position.beginEdit();
    double distance = (point - p_position.getValue()).norm();

    rotation.quatToAxis(tempAxis, tempAngle);
    //std::cout << tempAxis << " " << tempAngle << std::endl;
    Quat tempQuat (orientation.inverse().rotate(-tempAxis ), tempAngle);
    orientation = orientation*tempQuat;

    Vec3 trans = point + orientation.rotate(Vec3(0,0,-distance)) - position;
    position = position + trans;

    p_orientation.setValue(orientation);
    p_position.endEdit();
}

void BaseCamera::rotateWorldAroundPoint(Quat &rotation, const Vec3 &point, Quat orientationCam)
{
    Vec3 tempAxis;
    SReal tempAngle;
    //Quat orientationCam = this->getOrientation();
    Vec3& positionCam = *p_position.beginEdit();

    rotation.quatToAxis(tempAxis, tempAngle);
    Quat tempQuat (orientationCam.rotate(-tempAxis), tempAngle);

    defaulttype::SolidTypes<SReal>::Transform world_H_cam(positionCam, orientationCam);
    defaulttype::SolidTypes<SReal>::Transform world_H_pivot(point, Quat());
    defaulttype::SolidTypes<SReal>::Transform pivotBefore_R_pivotAfter(Vec3(0.0,0.0,0.0), tempQuat);
    defaulttype::SolidTypes<SReal>::Transform camera_H_WorldAfter = world_H_cam.inversed() * world_H_pivot * pivotBefore_R_pivotAfter * world_H_pivot.inversed();
    //defaulttype::SolidTypes<double>::Transform camera_H_WorldAfter = worldBefore_H_cam.inversed()*worldBefore_R_worldAfter;

    positionCam = camera_H_WorldAfter.inversed().getOrigin();
    orientationCam = camera_H_WorldAfter.inversed().getOrientation();

	if ( !p_fixedLookAtPoint.getValue() )
	{
		p_lookAt.setValue(getLookAtFromOrientation(positionCam, p_distance.getValue(), orientationCam));
		currentLookAt = p_lookAt.getValue();
	}

    p_orientation.setValue(orientationCam);
    p_position.endEdit();
}

void BaseCamera::computeZ()
{
    //if (!p_zNear.isSet() || !p_zFar.isSet())
    {
        double zNear = 1e10;
        double zFar = -1e10;
        double zNearTemp = zNear;
        double zFarTemp = zFar;

        const Vec3& currentPosition = getPosition();
        Quat currentOrientation = this->getOrientation();

        const Vec3 & minBBox = p_minBBox.getValue();
        const Vec3 & maxBBox = p_maxBBox.getValue();

        currentOrientation.normalize();
        helper::gl::Transformation transform;

        currentOrientation.buildRotationMatrix(transform.rotation);
        for (unsigned int i=0 ; i< 3 ; ++i)
            transform.translation[i] = -currentPosition[i];

        for (int corner=0; corner<8; ++corner)
        {
            Vec3 p((corner&1)?minBBox[0]:maxBBox[0],
                    (corner&2)?minBBox[1]:maxBBox[1],
                    (corner&4)?minBBox[2]:maxBBox[2]);
            //TODO: invert transform...
            p = transform * p;
            double z = -p[2];
            if (z < zNearTemp) zNearTemp = z;
            if (z > zFarTemp)  zFarTemp = z;
        }

        //get the same zFar and zNear calculations as QGLViewer
        sceneCenter = (minBBox + maxBBox)*0.5;
        sceneRadius = 0.5*(maxBBox - minBBox).norm();

        double distanceCamToCenter = (currentPosition - sceneCenter).norm();
        double zClippingCoeff = 5;
        double zNearCoeff = 0.001;

        zFar = distanceCamToCenter + zClippingCoeff*sceneRadius ;
        zNear = distanceCamToCenter- zClippingCoeff*sceneRadius;

        double zMin = zNearCoeff * zClippingCoeff * sceneRadius;
        if (zNear < zMin)
            zNear = zMin;

        if(p_zNear.getValue() >= p_zFar.getValue())
        {
            currentZNear = zNear;
            currentZFar = zFar;
        }
        else
        {
            if (p_zNear.getValue() <= 0.0)
                currentZNear = zNear;
            else
                currentZNear = p_zNear.getValue();

            if (p_zFar.getValue() <= 0.0)
                currentZFar = zFar;
            else
                currentZFar = p_zFar.getValue();
        }

    }
}

void BaseCamera::fitSphere(const Vec3 &center, SReal radius)
{
    SReal fov_radian = getFieldOfView() * (M_PI/180);
    SReal hor_fov_radian = getHorizontalFieldOfView() * (M_PI/180);
    const SReal yview = radius / sin(fov_radian/2.0);
    const SReal xview = radius / sin(hor_fov_radian/2.0);
    SReal distance = std::max(xview,yview);
    const Quat& orientation = p_orientation.getValue();
    Vec3 viewDirection = orientation.rotate(Vec3(0.0, 0.0, -1.0));

    Vec3 newPos = center - viewDirection*distance;
    p_position.setValue(newPos);
}

void BaseCamera::fitBoundingBox(const Vec3 &min, const Vec3 &max)
{
    SReal diameter = std::max(fabs(max[1]-min[1]), fabs(max[0]-min[0]));
    diameter = std::max((SReal)fabs(max[2]-min[2]), diameter);
    Vec3 center = (min + max)*0.5;

    fitSphere(center,0.5*diameter);

}

void BaseCamera::setView(const Vec3& position, const Quat &orientation)
{
    p_position.setValue(position);
    p_orientation.setValue(orientation);
    computeZ();
}

void BaseCamera::setDefaultView(const Vec3 & gravity)
{
    const Vec3 & minBBox = p_minBBox.getValue();
    const Vec3 & maxBBox = p_maxBBox.getValue();
    sceneCenter = (minBBox + maxBBox)*0.5;

    //LookAt
    p_lookAt.setValue(sceneCenter);
    currentLookAt = p_lookAt.getValue();

    //Orientation
    Vec3 xAxis (1.0, 0.0, 0.0);
    Vec3 yAxis = -gravity;
    // If no gravity defined set the yAxis as 0 1 0;
    if(gravity==Vec3(0.0, 0.0, 0.0))
    {
        yAxis = Vec3(0.0,1.0,0.0);
    }
    yAxis.normalize();

    if( 1.0 - fabs(dot(xAxis, yAxis)) < 0.001)
        xAxis = Vec3(0.0,1.0,0.0);

    Vec3 zAxis = xAxis.cross(yAxis);
    zAxis.normalize();
    xAxis = yAxis.cross(zAxis);
    xAxis.normalize();
    Quat q = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
    q.normalize();
    p_orientation.setValue(q);

    //Distance
    double coeff = 3.0;
    double dist = (minBBox - sceneCenter).norm() * coeff;
    p_distance.setValue(dist);
    currentDistance = dist;

    //Position
    Vec3 pos = currentLookAt + zAxis*dist;
    p_position.setValue(pos);

    computeZ();
}

void BaseCameraXMLExportSingleParameter(TiXmlElement* root, core::objectmodel::BaseData& data, const std::string& comment)
{
    TiXmlElement* node = new TiXmlElement( data.getName().c_str() );
    node->SetAttribute("value", data.getValueString().c_str() );
    if(!comment.empty())
    {
        TiXmlComment* com = new TiXmlComment( comment.c_str() );
        root->LinkEndChild(com);
    }
    root->LinkEndChild(node);
}

bool BaseCamera::exportParametersInFile(const std::string& viewFilename)
{
    TiXmlDocument doc;
    TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );
    doc.LinkEndChild( decl );

    TiXmlElement* root = new TiXmlElement( "Camera" );
    root->SetAttribute("version", "1.0" );
    doc.LinkEndChild( root );

    BaseCameraXMLExportSingleParameter(root, p_position, "Vector of 3 reals (x, y, z)");
    BaseCameraXMLExportSingleParameter(root, p_orientation, "Quaternion (x, y, z, w)");
    BaseCameraXMLExportSingleParameter(root, p_lookAt, "Vector of 3 reals (x, y, z)");
    BaseCameraXMLExportSingleParameter(root, p_fieldOfView, "Real");
    BaseCameraXMLExportSingleParameter(root, p_distance, "Real");
    BaseCameraXMLExportSingleParameter(root, p_zNear, "Real");
    BaseCameraXMLExportSingleParameter(root, p_zFar, "Real");
    BaseCameraXMLExportSingleParameter(root, p_type, "Int (0 -> Perspective, 1 -> Orthographic)");

    return doc.SaveFile( viewFilename.c_str() );
}

bool BaseCameraXMLImportSingleParameter(TiXmlElement* root, core::objectmodel::BaseData& data, BaseCamera* c)
{
    if(root)
    {
        TiXmlNode* node = root->FirstChild( data.getName().c_str() );
        if(node)
        {
            TiXmlElement* element = node->ToElement();
            if(element)
            {
                const char* attrValue;
                attrValue = element->Attribute("value");
                if(attrValue)
                {
                    std::string m_string; m_string.assign(attrValue);
                    bool retvalue = data.read(m_string);
                    if(!retvalue)
                        c->serr << "Unreadable value for " << data.getName() << " field." << c->sendl;
                    return retvalue;
                }
                else
                {
                    c->serr << "Attribute value has not been found for " << data.getName() << " field." << c->sendl;
                    return false;
                }
            }
            else
            {
                c->serr << "Unknown error occured for " << data.getName() << " field." << c->sendl;
                return false;
            }
        }
        else
        {
            c->serr << "Field " << data.getName() << " has not been found." << c->sendl;
            return false;
        }
    }
    else return false;
}

bool BaseCamera::importParametersFromFile(const std::string& viewFilename)
{
    sout << "Reading " << viewFilename << " for view parameters." << sendl;
    TiXmlDocument doc(viewFilename.c_str());
    if (!doc.LoadFile())
    {
        return false;
    }

    TiXmlHandle hDoc(&doc);
    TiXmlElement* root;

    root = hDoc.FirstChildElement().ToElement();

    if (!root)
        return false;

    //std::string camVersion;
    //root->QueryStringAttribute ("version", &camVersion);

    BaseCameraXMLImportSingleParameter(root, p_position, this);
    BaseCameraXMLImportSingleParameter(root, p_orientation, this);
    BaseCameraXMLImportSingleParameter(root, p_lookAt, this);
    BaseCameraXMLImportSingleParameter(root, p_fieldOfView, this);
    BaseCameraXMLImportSingleParameter(root, p_distance, this);
    BaseCameraXMLImportSingleParameter(root, p_zNear, this);
    BaseCameraXMLImportSingleParameter(root, p_zFar, this);
    BaseCameraXMLImportSingleParameter(root, p_type, this);

    return true;
}

void BaseCamera::handleEvent(sofa::core::objectmodel::Event* /* event */)
{

}


} // namespace visualmodel

} //namespace component

} //namespace sofa

