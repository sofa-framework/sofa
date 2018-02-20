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
#define _USE_MATH_DEFINES // for C++
#include <cmath>

#include <SofaBaseVisual/BaseCamera.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/simulation/AnimateBeginEvent.h>

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
    ,p_zNear(initData(&p_zNear, (double) 0.0 , "zNear", "Camera's zNear"))
    ,p_zFar(initData(&p_zFar, (double) 0.0 , "zFar", "Camera's zFar"))
    ,p_computeZClip(initData(&p_computeZClip, (bool) true, "computeZClip", "Compute Z clip planes (Near and Far) according to the bounding box"))
    ,p_minBBox(initData(&p_minBBox, Vec3(0.0,0.0,0.0) , "minBBox", "minBBox"))
    ,p_maxBBox(initData(&p_maxBBox, Vec3(1.0,1.0,1.0) , "maxBBox", "maxBBox"))
    ,p_widthViewport(initData(&p_widthViewport, (unsigned int) 800 , "widthViewport", "widthViewport"))
    ,p_heightViewport(initData(&p_heightViewport,(unsigned int) 600 , "heightViewport", "heightViewport"))
    ,p_type(initData(&p_type,"projectionType", "Camera Type (0 = Perspective, 1 = Orthographic)"))
    ,p_activated(initData(&p_activated, true , "activated", "Camera activated ?"))
    ,p_fixedLookAtPoint(initData(&p_fixedLookAtPoint, false, "fixedLookAt", "keep the lookAt point always fixed"))
    ,p_modelViewMatrix(initData(&p_modelViewMatrix,  "modelViewMatrix", "ModelView Matrix"))
    ,p_projectionMatrix(initData(&p_projectionMatrix,  "projectionMatrix", "Projection Matrix"))
    ,b_setDefaultParameters(false)
{
    this->f_listening.setValue(true);
    this->p_projectionMatrix.setReadOnly(true);
    this->p_modelViewMatrix.setReadOnly(true);
    this->p_widthViewport.setReadOnly(true);
    this->p_heightViewport.setReadOnly(true);
    this->p_minBBox.setReadOnly(true);
    this->p_maxBBox.setReadOnly(true);

    sofa::helper::OptionsGroup type(2, "Perspective", "Orthographic");
    type.setSelectedItem(sofa::core::visual::VisualParams::PERSPECTIVE_TYPE);
    p_type.setValue(type);

    helper::vector<float>& wModelViewMatrix = *p_modelViewMatrix.beginEdit();
    helper::vector<float>& wProjectionMatrix = *p_projectionMatrix.beginEdit();

    wModelViewMatrix.resize(16);
    wProjectionMatrix.resize(16);

    p_modelViewMatrix.endEdit();
    p_projectionMatrix.endEdit();

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
            sout << "Too many missing parameters ; taking default ..." << sendl;
            b_setDefaultParameters = true;
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
            sout << "Too many missing parameters ; taking default ..." << sendl;
            b_setDefaultParameters = true;
        }
    }
    currentDistance = p_distance.getValue();
    currentZNear = p_zNear.getValue();
    currentZFar = p_zFar.getValue();
    p_minBBox.setValue(getContext()->f_bbox.getValue().minBBox());
    p_maxBBox.setValue(getContext()->f_bbox.getValue().maxBBox());

    updateOutputData();

}

void BaseCamera::translate(const Vec3& t)
{
    Vec3 &pos = *p_position.beginEdit();
    pos += t;
    p_position.endEdit();

    updateOutputData();
}

void BaseCamera::translateLookAt(const Vec3& t)
{
    Vec3 &lookat = *p_lookAt.beginEdit();
    lookat += t;
    currentLookAt = lookat;
    p_lookAt.endEdit();

    updateOutputData();

}

void BaseCamera::rotate(const Quat& r)
{
    Quat &rot = *p_orientation.beginEdit();
    rot = rot * r;
    rot.normalize();
    p_orientation.endEdit();

    updateOutputData();
}

void BaseCamera::moveCamera(const Vec3 &p, const Quat &q)
{
    translate(p);
    if ( !p_fixedLookAtPoint.getValue() )
    {
        translateLookAt(p);
    }
    rotate(q);

    updateOutputData();
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

// TODO: move to helper
// https://www.opengl.org/wiki/GluProject_and_gluUnProject_code
template<class Real>
bool glhUnProjectf(Real winx, Real winy, Real winz, Real *modelview, Real *projection, const core::visual::VisualParams::Viewport& viewport, Real *objectCoordinate)
{
    //Transformation matrices
    sofa::defaulttype::Mat<4,4, Real> matModelview(modelview);
    sofa::defaulttype::Mat<4, 4, Real> matProjection(projection);

    sofa::defaulttype::Mat<4, 4, Real> m, A;
    sofa::defaulttype::Vec<4, Real> in, out;

    A = matProjection * matModelview ;
    sofa::defaulttype::invertMatrix(m, A);

    //Transformation of normalized coordinates between -1 and 1
    in[0] = (winx - (Real)viewport[0]) / (Real)viewport[2] * 2.0 - 1.0;
    in[1] = (winy - (Real)viewport[1]) / (Real)viewport[3] * 2.0 - 1.0;
    in[2] = 2.0*winz - 1.0;
    in[3] = 1.0;
    //Objects coordinates
    out = m * in;

    if (out[3] == 0.0)
        return false;
    out[3] = 1.0 / out[3];
    objectCoordinate[0] = out[0] * out[3];
    objectCoordinate[1] = out[1] * out[3];
    objectCoordinate[2] = out[2] * out[3];
    return true;
}

BaseCamera::Quat BaseCamera::getOrientation()
{
    if(currentLookAt !=  p_lookAt.getValue())
    {
        Quat newOrientation = getOrientationFromLookAt(p_position.getValue(), p_lookAt.getValue());
        p_orientation.setValue(newOrientation);

        currentLookAt = p_lookAt.getValue();
    }

    return p_orientation.getValue();
}


unsigned int BaseCamera::getCameraType() const
{
    return p_type.getValue().getSelectedId();
}


void BaseCamera::setCameraType(unsigned int type)
{
    sofa::helper::OptionsGroup* optionsGroup = p_type.beginEdit();

    if (type == core::visual::VisualParams::ORTHOGRAPHIC_TYPE)
        optionsGroup->setSelectedItem(core::visual::VisualParams::ORTHOGRAPHIC_TYPE);
    else
        optionsGroup->setSelectedItem(core::visual::VisualParams::PERSPECTIVE_TYPE);

    p_type.endEdit();
}


double BaseCamera::getHorizontalFieldOfView()
{
    const sofa::core::visual::VisualParams* vp = sofa::core::visual::VisualParams::defaultInstance();
    const core::visual::VisualParams::Viewport viewport = vp->viewport();

    float screenwidth = (float)viewport[2];
    float screenheight = (float)viewport[3];
    float aspectRatio = screenwidth / screenheight;
    float fov_radian = (float)getFieldOfView()* (float)(M_PI/180);
    float hor_fov_radian = 2.0f * atan ( tan(fov_radian/2.0f) * aspectRatio );
    return hor_fov_radian*(180/M_PI);
}

BaseCamera::Vec3 BaseCamera::screenToWorldCoordinates(int x, int y)
{
    const sofa::core::visual::VisualParams* vp = sofa::core::visual::VisualParams::defaultInstance();

    const core::visual::VisualParams::Viewport viewport = vp->viewport();

    double winX = (double)x;
    double winY = (double)viewport[3] - (double)y;

    double pos[3];
    double modelview[16];
    double projection[16];

    this->getModelViewMatrix(modelview);
    this->getProjectionMatrix(projection);

    float fwinZ = 0.0;
    vp->drawTool()->readPixels(x, int(winY), 1, 1, NULL, &fwinZ);

    double winZ = (double)fwinZ;
    glhUnProjectf<double>(winX, winY, winZ, modelview, projection, viewport, pos);

    return Vec3(pos[0], pos[1], pos[2]);
}

void BaseCamera::getModelViewMatrix(double mat[16])
{
    defaulttype::SolidTypes<SReal>::Transform world_H_cam(p_position.getValue(), this->getOrientation());
    Mat3 rot = world_H_cam.inversed().getRotationMatrix();

    //rotation
    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
            mat[i * 4 + j] = rot[i][j];

    //translation
    Vec3 t = world_H_cam.inversed().getOrigin();
    mat[3] = t[0];
    mat[7] = t[1];
    mat[11] = t[2];
    //w
    mat[12] = 0;
    mat[13] = 0;
    mat[14] = 0;
    mat[15] = 1;

}

void BaseCamera::getOpenGLModelViewMatrix(double mat[16])
{
    defaulttype::SolidTypes<SReal>::Transform world_H_cam(p_position.getValue(), this->getOrientation());
    world_H_cam.inversed().writeOpenGlMatrix(mat);
}

void BaseCamera::getProjectionMatrix(double mat[16])
{
    float width = (float)p_widthViewport.getValue();
    float height = (float)p_heightViewport.getValue();
    //TODO: check if orthographic or projective

    computeZ();

    std::fill(mat, mat + 16, 0);

    if (p_type.getValue().getSelectedId() == core::visual::VisualParams::PERSPECTIVE_TYPE)
    {
        double pm00, pm11;
        double scale = 1.0 / tan(getFieldOfView() * M_PI / 180 * 0.5);
        double aspect = width / height;

        pm00 = scale / aspect;
        pm11 = scale;

        mat[0] = pm00; // FocalX
        mat[5] = pm11; // FocalY
        mat[10] = -(currentZFar + currentZNear) / (currentZFar - currentZNear);
        mat[11] = -2.0 * currentZFar * currentZNear / (currentZFar - currentZNear);;
        mat[14] = -1.0;
    }
    else
    {
        float xFactor = 1.0, yFactor = 1.0;
        if ((height != 0) && (width != 0))
        {
            if (height > width)
            {
                yFactor = (double)height / (double)width;
            }
            else
            {
                xFactor = (double)width / (double)height;
            }
        }

        double orthoCoef = tan((float)(M_PI / 180.0) * getFieldOfView() / 2.0);
        double zDist = orthoCoef * fabs(worldToCameraCoordinates(getLookAt())[2]);
        double halfWidth = zDist * xFactor;
        double halfHeight = zDist * yFactor;

        float left = -halfWidth;
        float right = halfWidth;
        float top = halfHeight;
        float bottom = -halfHeight;
        float zfar = currentZFar;
        float znear = currentZNear;

        mat[0] = 2 / (right-left);
        mat[1] = 0.0;
        mat[2] = 0.0;
        mat[3] = -1 * (right + left) / (right - left);

        mat[4] = 0.0;
        mat[5] = 2 / (top-bottom);
        mat[6] = 0.0;
        mat[7] = -1 * (top + bottom) / (top - bottom);

        mat[8] = 0;
        mat[9] = 0;
        mat[10] = -2 / (zfar - znear);
        mat[11] = -1 * (zfar + znear) / (zfar - znear);

        mat[12] = 0.0;
        mat[13] = 0.0;
        mat[14] = 0.0;
        mat[15] = 1.0;
    }
}

void BaseCamera::getOpenGLProjectionMatrix(double oglProjectionMatrix[16])
{
    double projectionMatrix[16];
    this->getProjectionMatrix(projectionMatrix);

    for(unsigned int i=0 ; i<4 ; i++)
    {
        for(unsigned int j=0 ; j<4 ; j++)
            oglProjectionMatrix[i+j*4] = projectionMatrix[i*4+j];
    }
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
    Quat tempQuat (orientation.inverse().rotate(-tempAxis ), tempAngle);
    orientation = orientation*tempQuat;

    Vec3 trans = point + orientation.rotate(Vec3(0,0,-distance)) - position;
    position = position + trans;

    p_orientation.setValue(orientation);
    p_position.endEdit();

    updateOutputData();
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

    updateOutputData();
}

void BaseCamera::computeZ()
{
    if (p_computeZClip.getValue())
    {
        double zNear = 1e10;
        double zFar = -1e10;

        const Vec3 & minBBox = p_minBBox.getValue();
        const Vec3 & maxBBox = p_maxBBox.getValue();

        //get the same zFar and zNear calculations as QGLViewer
        sceneCenter = (minBBox + maxBBox)*0.5;
        sceneRadius = 0.5*(maxBBox - minBBox).norm();

        //modelview transform
        defaulttype::SolidTypes<SReal>::Transform world_H_cam(p_position.getValue(), this->getOrientation());

        //double distanceCamToCenter = fabs((world_H_cam.inversed().projectPoint(sceneCenter))[2]);
        double distanceCamToCenter = (p_position.getValue() - sceneCenter).norm();

        double zClippingCoeff = 5;
        double zNearCoeff = 0.01;


        zNear = distanceCamToCenter - sceneRadius;
        zFar = (zNear + 2 * sceneRadius) * 1.1;
        zNear = zNear * zNearCoeff;

        double zMin = zNearCoeff * zClippingCoeff * sceneRadius;

        if (zNear < zMin)
            zNear = zMin;

        currentZNear = zNear;
        currentZFar = zFar;
    }
    else
    {
        if (p_zNear.getValue() >= p_zFar.getValue())
        {
            serr << "ZNear > ZFar !" << sendl;
        }
        else if (p_zNear.getValue() <= 0.0)
        {
            serr << "ZNear is negative!" << sendl;
        }
        else if (p_zFar.getValue() <= 0.0)
        {
            serr << "ZFar is negative!" << sendl;
        }
        else
        {
            currentZNear = p_zNear.getValue();
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

    if (b_setDefaultParameters)
    {
        //LookAt
        p_lookAt.setValue(sceneCenter);
        currentLookAt = p_lookAt.getValue();

        //Orientation
        Vec3 xAxis(1.0, 0.0, 0.0);
        Vec3 yAxis = -gravity;
        // If no gravity defined set the yAxis as 0 1 0;
        if (gravity == Vec3(0.0, 0.0, 0.0))
        {
            yAxis = Vec3(0.0, 1.0, 0.0);
        }
        yAxis.normalize();

        if (1.0 - fabs(dot(xAxis, yAxis)) < 0.001)
            xAxis = Vec3(0.0, 1.0, 0.0);

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
    }

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
    bool result = true;

    sout << "Reading " << viewFilename << " for view parameters." << sendl;
    TiXmlDocument doc(viewFilename.c_str());
    if (!doc.LoadFile())
    {
        result = false;
    }

    TiXmlHandle hDoc(&doc);
    TiXmlElement* root;

    root = hDoc.FirstChildElement().ToElement();

    if (!root)
        result = false;

    //std::string camVersion;
    //root->QueryStringAttribute ("version", &camVersion);
    if(result)
    {
        BaseCameraXMLImportSingleParameter(root, p_position, this);
        BaseCameraXMLImportSingleParameter(root, p_orientation, this);
        BaseCameraXMLImportSingleParameter(root, p_lookAt, this);
        BaseCameraXMLImportSingleParameter(root, p_fieldOfView, this);
        BaseCameraXMLImportSingleParameter(root, p_distance, this);
        BaseCameraXMLImportSingleParameter(root, p_zNear, this);
        BaseCameraXMLImportSingleParameter(root, p_zFar, this);
        BaseCameraXMLImportSingleParameter(root, p_type, this);
    }
    else
    {
        sout << "Error while reading " << viewFilename << "." << sendl;
    }
    return result;
}

void BaseCamera::updateOutputData()
{
    //Matrices
    //sofa::helper::WriteAccessor< Data<Mat4> > wModelViewMatrix = p_modelViewMatrix;
    //sofa::helper::WriteAccessor< Data<Mat4> > wProjectionMatrix = p_projectionMatrix;
    helper::vector<float>& wModelViewMatrix = *p_modelViewMatrix.beginEdit();
    helper::vector<float>& wProjectionMatrix = *p_projectionMatrix.beginEdit();

    double modelViewMatrix[16];
    double projectionMatrix[16];

    this->getModelViewMatrix(modelViewMatrix);
    this->getProjectionMatrix(projectionMatrix);

    for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
        {
            wModelViewMatrix[i*4+j] = modelViewMatrix[i * 4 + j];
            wProjectionMatrix[i*4+j] = projectionMatrix[i * 4 + j];
        }

    p_modelViewMatrix.endEdit();
    p_projectionMatrix.endEdit();

    //TODO: other info to update
    p_minBBox.setValue(getContext()->f_bbox.getValue().minBBox());
    p_maxBBox.setValue(getContext()->f_bbox.getValue().maxBBox());

    p_zNear.setValue(currentZNear);
    p_zFar.setValue(currentZFar);
}

void BaseCamera::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
        updateOutputData();
}


} // namespace visualmodel

} //namespace component

} //namespace sofa

