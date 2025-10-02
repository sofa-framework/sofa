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
#include <sofa/component/visual/BaseCamera.h>


#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/type/Mat.h>
using Mat3 = sofa::type::Mat3x3;
using Mat4 = sofa::type::Mat4x4;

#include <sofa/type/Transform.h>
#include <sofa/simulation/AnimateBeginEvent.h>

#include <sofa/helper/rmath.h>
using sofa::helper::isEqual;

#include <cmath>
#include <tinyxml2.h>

using sofa::type::RGBAColor ;

namespace sofa::component::visual
{

BaseCamera::BaseCamera()
    : d_position(initData(&d_position, "position", "Camera's position"))
    , d_orientation(initData(&d_orientation, "orientation", "Camera's orientation"))
    , d_lookAt(initData(&d_lookAt, "lookAt", "Camera's look at"))
    , d_distance(initData(&d_distance, "distance", "Distance between camera and look at"))
    , d_fieldOfView(initData(&d_fieldOfView, (double) (45.0) , "fieldOfView", "Camera's FOV"))
    , d_zNear(initData(&d_zNear, (double) 0.01 , "zNear", "Camera's zNear"))
    , d_zFar(initData(&d_zFar, (double) 100.0 , "zFar", "Camera's zFar"))
    , d_computeZClip(initData(&d_computeZClip, (bool)true, "computeZClip", "Compute Z clip planes (Near and Far) according to the bounding box"))
    , d_minBBox(initData(&d_minBBox, type::Vec3(0.0, 0.0, 0.0) , "minBBox", "minBBox"))
    , d_maxBBox(initData(&d_maxBBox, type::Vec3(1.0, 1.0, 1.0) , "maxBBox", "maxBBox"))
    , d_widthViewport(initData(&d_widthViewport, (unsigned int) 800 , "widthViewport", "widthViewport"))
    , d_heightViewport(initData(&d_heightViewport, (unsigned int) 600 , "heightViewport", "heightViewport"))
    , d_type(initData(&d_type, "projectionType", "Camera Type (0 = Perspective, 1 = Orthographic)"))
    , d_activated(initData(&d_activated, true , "activated", "Camera activated ?"))
    , d_fixedLookAtPoint(initData(&d_fixedLookAtPoint, false, "fixedLookAt", "keep the lookAt point always fixed"))
    , d_modelViewMatrix(initData(&d_modelViewMatrix, "modelViewMatrix", "ModelView Matrix"))
    , d_projectionMatrix(initData(&d_projectionMatrix, "projectionMatrix", "Projection Matrix"))
    , b_setDefaultParameters(false)
{
    this->f_listening.setValue(true);
    this->d_projectionMatrix.setReadOnly(true);
    this->d_modelViewMatrix.setReadOnly(true);
    this->d_widthViewport.setReadOnly(true);
    this->d_heightViewport.setReadOnly(true);
    this->d_minBBox.setReadOnly(true);
    this->d_maxBBox.setReadOnly(true);

    sofa::helper::OptionsGroup type{"Perspective", "Orthographic"};
    type.setSelectedItem(sofa::core::visual::VisualParams::PERSPECTIVE_TYPE);
    d_type.setValue(type);

    type::vector<SReal>& wModelViewMatrix = *d_modelViewMatrix.beginEdit();
    type::vector<SReal>& wProjectionMatrix = *d_projectionMatrix.beginEdit();

    wModelViewMatrix.resize(16);
    wProjectionMatrix.resize(16);

    d_modelViewMatrix.endEdit();
    d_projectionMatrix.endEdit();
}

BaseCamera::~BaseCamera()
{
}

void BaseCamera::activate()
{
    d_activated.setValue(true);
}

void BaseCamera::desactivate()
{
    d_activated.setValue(false);
}

bool BaseCamera::isActivated()
{
    return d_activated.getValue();
}

void BaseCamera::init()
{
    if(d_position.isSet())
    {
        if(!d_orientation.isSet())
        {
            d_distance.setValue((d_lookAt.getValue() - d_position.getValue()).norm());

            const Quat q  = getOrientationFromLookAt(d_position.getValue(), d_lookAt.getValue());
            d_orientation.setValue(q);
        }
        else if(!d_lookAt.isSet())
        {
            //distance assumed to be set
            if(!d_distance.isSet())
                msg_warning() << "Missing distance parameter ; taking default value (0.0, 0.0, 0.0)" ;

            const type::Vec3 lookat = getLookAtFromOrientation(d_position.getValue(), d_distance.getValue(), d_orientation.getValue());
            d_lookAt.setValue(lookat);
        }
        else
        {
            msg_warning() << "Too many missing parameters ; taking default ..." ;
            b_setDefaultParameters = true;
        }
    }
    else
    {
        if(d_lookAt.isSet() && d_orientation.isSet())
        {
            //distance assumed to be set
            if(!d_distance.isSet())
                msg_warning() << "Missing distance parameter ; taking default value (0.0, 0.0, 0.0)" ;

            const type::Vec3 pos = getPositionFromOrientation(d_lookAt.getValue(), d_distance.getValue(), d_orientation.getValue());
            d_position.setValue(pos);
        }
        else
        {
            msg_warning() << "Too many missing parameters ; taking default ..." ;
            b_setDefaultParameters = true;
        }
    }
    currentDistance = d_distance.getValue();
    currentZNear = d_zNear.getValue();
    currentZFar = d_zFar.getValue();
}

void BaseCamera::reinit()
{
    //Data "LookAt" has changed
    //-> Orientation needs to be updated
    if(currentLookAt != d_lookAt.getValue())
    {
        const Quat newOrientation = getOrientationFromLookAt(d_position.getValue(), d_lookAt.getValue());
        d_orientation.setValue(newOrientation);

        currentLookAt = d_lookAt.getValue();
    }

    updateOutputData();
}

void BaseCamera::bwdInit()
{
    d_minBBox.setValue(getContext()->f_bbox.getValue().minBBox());
    d_maxBBox.setValue(getContext()->f_bbox.getValue().maxBBox());

    updateOutputData();
}

void BaseCamera::translate(const type::Vec3& t)
{
    type::Vec3 &pos = *d_position.beginEdit();
    pos += t;
    d_position.endEdit();

    updateOutputData();
}

void BaseCamera::translateLookAt(const type::Vec3& t)
{
    type::Vec3 &lookat = *d_lookAt.beginEdit();
    lookat += t;
    currentLookAt = lookat;
    d_lookAt.endEdit();

    updateOutputData();

}

void BaseCamera::rotate(const Quat& r)
{
    Quat &rot = *d_orientation.beginEdit();
    rot = rot * r;
    rot.normalize();
    d_orientation.endEdit();

    updateOutputData();
}

void BaseCamera::moveCamera(const type::Vec3 &p, const Quat &q)
{
    translate(p);
    if ( !d_fixedLookAtPoint.getValue() )
    {
        translateLookAt(p);
    }
    rotate(q);

    updateOutputData();
}

type::Vec3 BaseCamera::cameraToWorldCoordinates(const type::Vec3& p)
{
    return d_orientation.getValue().rotate(p) + d_position.getValue();
}

type::Vec3 BaseCamera::worldToCameraCoordinates(const type::Vec3& p)
{
    return d_orientation.getValue().inverseRotate(p - d_position.getValue());
}

type::Vec3 BaseCamera::cameraToWorldTransform(const type::Vec3& v)
{
    const Quat q = d_orientation.getValue();
    return q.rotate(v) ;
}

type::Vec3 BaseCamera::worldToCameraTransform(const type::Vec3& v)
{
    return d_orientation.getValue().inverseRotate(v);
}

// TODO: move to helper
// https://www.opengl.org/wiki/GluProject_and_gluUnProject_code
template<class Real>
bool glhUnProjectf(Real winx, Real winy, Real winz, Real *modelview, Real *projection, const core::visual::VisualParams::Viewport& viewport, Real *objectCoordinate)
{
    //Transformation matrices
    sofa::type::Mat<4,4, Real> matModelview(modelview);
    sofa::type::Mat<4, 4, Real> matProjection(projection);

    sofa::type::Mat<4, 4, Real> m, A;
    sofa::type::Vec<4, Real> in, out;

    A = matProjection * matModelview ;
    const bool canInvert = sofa::type::invertMatrix(m, A);
    assert(canInvert);
    SOFA_UNUSED(canInvert);

    //Transformation of normalized coordinates between -1 and 1
    in[0] = (winx - (Real)viewport[0]) / (Real)viewport[2] * 2.0 - 1.0;
    in[1] = (winy - (Real)viewport[1]) / (Real)viewport[3] * 2.0 - 1.0;
    in[2] = 2.0*winz - 1.0;
    in[3] = 1.0;
    //Objects coordinates
    out = m * in;

    if (isEqual(out[3], 0.0))
        return false;
    out[3] = 1.0 / out[3];
    objectCoordinate[0] = out[0] * out[3];
    objectCoordinate[1] = out[1] * out[3];
    objectCoordinate[2] = out[2] * out[3];
    return true;
}

BaseCamera::Quat BaseCamera::getOrientation()
{
    if(currentLookAt != d_lookAt.getValue())
    {
        const Quat newOrientation = getOrientationFromLookAt(d_position.getValue(), d_lookAt.getValue());
        d_orientation.setValue(newOrientation);

        currentLookAt = d_lookAt.getValue();
    }

    return d_orientation.getValue();
}


unsigned int BaseCamera::getCameraType() const
{
    return d_type.getValue().getSelectedId();
}


void BaseCamera::setCameraType(unsigned int type)
{
    sofa::helper::OptionsGroup* optionsGroup = d_type.beginEdit();

    if (type == core::visual::VisualParams::ORTHOGRAPHIC_TYPE)
        optionsGroup->setSelectedItem(core::visual::VisualParams::ORTHOGRAPHIC_TYPE);
    else
        optionsGroup->setSelectedItem(core::visual::VisualParams::PERSPECTIVE_TYPE);

    d_type.endEdit();
}


double BaseCamera::getHorizontalFieldOfView()
{
    const sofa::core::visual::VisualParams* vp = sofa::core::visual::VisualParams::defaultInstance();
    const core::visual::VisualParams::Viewport viewport = vp->viewport();

    const float screenwidth = (float)viewport[2];
    const float screenheight = (float)viewport[3];
    const float aspectRatio = screenwidth / screenheight;
    const float fov_radian = (float)getFieldOfView()* (float)(M_PI/180);
    const float hor_fov_radian = 2.0f * atan ( tan(fov_radian/2.0f) * aspectRatio );
    return hor_fov_radian*(180/M_PI);
}

type::Vec3 BaseCamera::screenToWorldCoordinates(int x, int y)
{
    const sofa::core::visual::VisualParams* vp = sofa::core::visual::VisualParams::defaultInstance();

    const core::visual::VisualParams::Viewport viewport = vp->viewport();
    if (viewport.empty() || !vp->drawTool())
        return type::Vec3(0,0,0);

    const double winX = (double)x;
    const double winY = (double)viewport[3] - (double)y;

    double pos[3]{};
    double modelview[16];
    double projection[16];

    this->getModelViewMatrix(modelview);
    this->getProjectionMatrix(projection);


    float fwinZ = 0.0;
    vp->drawTool()->readPixels(x, int(winY), 1, 1, nullptr, &fwinZ);

    const double winZ = (double)fwinZ;
    glhUnProjectf<double>(winX, winY, winZ, modelview, projection, viewport, pos);
    return type::Vec3(pos[0], pos[1], pos[2]);
}

type::Vec2 BaseCamera::worldToScreenCoordinates(const type::Vec3& pos)
{
    const sofa::core::visual::VisualParams* vp = sofa::core::visual::VisualParams::defaultInstance();

    const core::visual::VisualParams::Viewport viewport = vp->viewport();
    sofa::type::Vec4 clipSpacePos = {pos.x(), pos.y(), pos.z(), 1.0};
    sofa::type::Mat4x4d modelview;
    sofa::type::Mat4x4d projection;

    this->getModelViewMatrix(modelview.ptr());
    this->getProjectionMatrix(projection.ptr());

    clipSpacePos = projection * (modelview * clipSpacePos);
    if (isEqual(clipSpacePos.w(), 0.0_sreal))
        return type::Vec2(std::nan(""), std::nan(""));

    sofa::type::Vec3 ndcSpacePos = sofa::type::Vec3(clipSpacePos.x(),clipSpacePos.y(), clipSpacePos.z()) * clipSpacePos.w();
    const type::Vec2 screenCoord = type::Vec2((ndcSpacePos.x() + 1.0) / 2.0 * viewport[2], (ndcSpacePos.y() + 1.0) / 2.0 * viewport[3]);
    return screenCoord + type::Vec2(viewport[0], viewport[1]);
}

void BaseCamera::getModelViewMatrix(double mat[16])
{
    const sofa::type::Transform<SReal> world_H_cam(d_position.getValue(), this->getOrientation());
    Mat3 rot = world_H_cam.inversed().getRotationMatrix();

    //rotation
    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
            mat[i * 4 + j] = rot(i,j);

    //translation
    type::Vec3 t = world_H_cam.inversed().getOrigin();
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
    const sofa::type::Transform<SReal> world_H_cam(d_position.getValue(), this->getOrientation());
    world_H_cam.inversed().writeOpenGlMatrix(mat);
}

void BaseCamera::getProjectionMatrix(double mat[16])
{
    const double width = double(d_widthViewport.getValue());
    const double height = double(d_heightViewport.getValue());
    //TODO: check if orthographic or projective

    computeZ();

    std::fill(mat, mat + 16, 0);

    if (d_type.getValue().getSelectedId() == core::visual::VisualParams::PERSPECTIVE_TYPE)
    {
        double pm00, pm11;
        const double scale = 1.0 / tan(getFieldOfView() * M_PI / 180 * 0.5);
        const double aspect = width / height;

        pm00 = scale / aspect;
        pm11 = scale;

        mat[0] = pm00; // FocalX
        mat[5] = pm11; // FocalY
        mat[10] = -(currentZFar + currentZNear) / (currentZFar - currentZNear);
        mat[11] = -2.0 * currentZFar * currentZNear / (currentZFar - currentZNear);
        mat[14] = -1.0;
    }
    else
    {
        double xFactor = 1.0, yFactor = 1.0;
        if ((height != 0) && (width != 0))
        {
            if (height > width)
            {
                yFactor = height / width;
            }
            else
            {
                xFactor = width / height;
            }
        }

        const double orthoCoef = tan((M_PI / 180.0) * getFieldOfView() / 2.0);
        const double zDist = orthoCoef * fabs(worldToCameraCoordinates(getLookAt())[2]);
        const double halfWidth = zDist * xFactor;
        const double halfHeight = zDist * yFactor;

        const double left = -halfWidth;
        const double right = halfWidth;
        const double top = halfHeight;
        const double bottom = -halfHeight;
        const double zfar = currentZFar;
        const double znear = currentZNear;

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

BaseCamera::Quat BaseCamera::getOrientationFromLookAt(const type::Vec3 &pos, const type::Vec3& lookat)
{
    type::Vec3 zAxis = -(lookat - pos);
    zAxis.normalize();

    type::Vec3 yAxis = cameraToWorldTransform(type::Vec3(0,1,0));

    type::Vec3 xAxis = yAxis.cross(zAxis) ;
    xAxis.normalize();

    if (xAxis.norm2() < 0.00001)
        xAxis = cameraToWorldTransform(type::Vec3(1.0, 0.0, 0.0));
    xAxis.normalize();

    yAxis = zAxis.cross(xAxis);

    Quat q;
    q = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
    q.normalize();
    return q;
}


type::Vec3 BaseCamera::getLookAtFromOrientation(const type::Vec3 &pos, const double &distance, const BaseCamera::Quat & orientation)
{
    const type::Vec3 zWorld = orientation.rotate(type::Vec3(0,0,-1*distance));
    return zWorld+pos;
}

type::Vec3 BaseCamera::getPositionFromOrientation(const type::Vec3 &lookAt, const double &distance, const BaseCamera::Quat& orientation)
{
    const type::Vec3 zWorld = orientation.rotate(type::Vec3(0,0,-1*distance));
    return zWorld-lookAt;
}

void BaseCamera::rotateCameraAroundPoint(Quat& rotation, const type::Vec3& point)
{
    type::Vec3 tempAxis;
    SReal tempAngle;
    Quat orientation = this->getOrientation();
    type::Vec3& position = *d_position.beginEdit();
    const double distance = (point - d_position.getValue()).norm();

    rotation.quatToAxis(tempAxis, tempAngle);
    const Quat tempQuat (orientation.inverse().rotate(-tempAxis ), tempAngle);
    orientation = orientation*tempQuat;

    const type::Vec3 trans = point + orientation.rotate(type::Vec3(0,0,-distance)) - position;
    position = position + trans;

    d_orientation.setValue(orientation);
    d_position.endEdit();

    updateOutputData();
}

void BaseCamera::rotateWorldAroundPoint(Quat &rotation, const type::Vec3 &point, Quat orientationCam)
{
    type::Vec3 tempAxis;
    SReal tempAngle;
    //Quat orientationCam = this->getOrientation();
    type::Vec3& positionCam = *d_position.beginEdit();

    rotation.quatToAxis(tempAxis, tempAngle);
    const Quat tempQuat (orientationCam.rotate(-tempAxis), tempAngle);

    const sofa::type::Transform<SReal> world_H_cam(positionCam, orientationCam);
    const sofa::type::Transform<SReal> world_H_pivot(point, Quat());
    const sofa::type::Transform<SReal> pivotBefore_R_pivotAfter(type::Vec3(0.0,0.0,0.0), tempQuat);
    const sofa::type::Transform<SReal> camera_H_WorldAfter = world_H_cam.inversed() * world_H_pivot * pivotBefore_R_pivotAfter * world_H_pivot.inversed();
    //defaulttype::SolidTypes<double>::Transform camera_H_WorldAfter = worldBefore_H_cam.inversed()*worldBefore_R_worldAfter;

    positionCam = camera_H_WorldAfter.inversed().getOrigin();
    orientationCam = camera_H_WorldAfter.inversed().getOrientation();

    if ( !d_fixedLookAtPoint.getValue() )
    {
        d_lookAt.setValue(getLookAtFromOrientation(positionCam, d_distance.getValue(), orientationCam));
        currentLookAt = d_lookAt.getValue();
    }

    d_orientation.setValue(orientationCam);
    d_position.endEdit();

    updateOutputData();
}





type::Vec3 BaseCamera::screenToViewportPoint(const type::Vec3& p) const
{
    if (d_widthViewport.getValue() == 0 || d_heightViewport.getValue() == 0)
        return type::Vec3(0, 0, p.z());
    return type::Vec3(p.x() / this->d_widthViewport.getValue(),
                p.y() / this->d_heightViewport.getValue(),
                p.z());
}
type::Vec3 BaseCamera::screenToWorldPoint(const type::Vec3& p)
{
    const type::Vec3 vP = screenToViewportPoint(p);
    return viewportToWorldPoint(vP);
}

type::Vec3 BaseCamera::viewportToScreenPoint(const type::Vec3& p) const
{
    return type::Vec3(p.x() * d_widthViewport.getValue(), p.y() * d_heightViewport.getValue(), p.z());
}
type::Vec3 BaseCamera::viewportToWorldPoint(const type::Vec3& p)
{
    const type::Vec3 nsPosition{ p.x() * 2.0 - 1.0, (1.0 - p.y()) * 2.0 - 1.0, p.z() * 2.0 - 1.0 };

    sofa::type::Mat4x4d glP, glM, invertglP, invertglM;
    getOpenGLProjectionMatrix(glP.ptr());
    getOpenGLModelViewMatrix(glM.ptr());

    const bool canInvert1 = invertglP.invert(glP);
    assert(canInvert1);
    SOFA_UNUSED(canInvert1);
    const bool canInvert2 = invertglM.invert(glM);
    assert(canInvert2);
    SOFA_UNUSED(canInvert2);

    type::Vec4 vsPosition = invertglP.transposed() * type::Vec4(nsPosition, 1.0);
    if(isEqual(vsPosition.w(), SReal(0.0)))
    {
        return type::Vec3(std::nan(""), std::nan(""), std::nan(""));
    }
    vsPosition /= vsPosition.w();
    type::Vec4 v = (invertglM.transposed() * vsPosition);

    return type::Vec3(v[0],v[1],v[2]);
}

type::Vec3 BaseCamera::worldToScreenPoint(const type::Vec3& p)
{
    sofa::type::Mat4x4d glP, glM;
    getOpenGLProjectionMatrix(glP.ptr());
    getOpenGLModelViewMatrix(glM.ptr());

    type::Vec4 nsPosition = (glP.transposed() * glM.transposed() * type::Vec4(p, 1.0));

    if(isEqual(nsPosition.w(), SReal(0.0)))
    {
        return type::Vec3(std::nan(""), std::nan(""), std::nan(""));
    }

    nsPosition /= nsPosition.w();
    return type::Vec3((nsPosition.x() * 0.5 + 0.5) * d_widthViewport.getValue() + 0.5,
                      d_heightViewport.getValue() - (nsPosition.y() * 0.5 + 0.5) * d_heightViewport.getValue() + 0.5,
                      (nsPosition.z() * 0.5 + 0.5));
}
type::Vec3 BaseCamera::worldToViewportPoint(const type::Vec3& p)
{
    type::Vec3 ssPoint = worldToScreenPoint(p);
    return type::Vec3(ssPoint.x() / d_widthViewport.getValue(), ssPoint.y() / d_heightViewport.getValue(), ssPoint.z());
}

type::Ray BaseCamera::viewportPointToRay(const type::Vec3& p)
{
    return type::Ray(this->d_position.getValue(), (viewportToWorldPoint(p) - this->d_position.getValue()));
}
type::Ray BaseCamera::screenPointToRay(const type::Vec3& p)
{
    return type::Ray(this->d_position.getValue(), (screenToWorldPoint(p) - this->d_position.getValue()));
}

type::Ray BaseCamera::toRay() const
{
    return type::Ray(this->d_position.getValue(), this->d_lookAt.getValue());
}



void BaseCamera::computeZ()
{
    if (d_computeZClip.getValue())
    {
        //modelview transform
        sofa::type::Transform<SReal> world_H_cam(d_position.getValue(), this->getOrientation());

        //double distanceCamToCenter = fabs((world_H_cam.inversed().projectPoint(sceneCenter))[2]);
        const double distanceCamToCenter = (d_position.getValue() - sceneCenter).norm();

        const double zClippingCoeff = 5;
        const double zNearCoeff = 0.01;

        double zNear = distanceCamToCenter - sceneRadius;
        const double zFar = (zNear + 2 * sceneRadius) * 1.1;
        zNear = zNear * zNearCoeff;

        const double zMin = zNearCoeff * zClippingCoeff * sceneRadius;

        if (zNear < zMin)
            zNear = zMin;

        currentZNear = zNear;
        currentZFar = zFar;
    }
    else
    {
        if (d_zNear.getValue() >= d_zFar.getValue())
        {
            msg_error() << "ZNear > ZFar !";
        }
        else if (d_zNear.getValue() <= 0.0)
        {
            msg_error() << "ZNear is negative!";
        }
        else if (d_zFar.getValue() <= 0.0)
        {
            msg_error() << "ZFar is negative!";
        }
        else
        {
            currentZNear = d_zNear.getValue();
            currentZFar = d_zFar.getValue();
        }
    }
}

void BaseCamera::fitSphere(const type::Vec3 &center, SReal radius)
{
    const SReal fov_radian = getFieldOfView() * (M_PI/180);
    const SReal hor_fov_radian = getHorizontalFieldOfView() * (M_PI/180);
    const SReal yview = radius / sin(fov_radian/2.0);
    const SReal xview = radius / sin(hor_fov_radian/2.0);
    const SReal distance = std::max(xview,yview);
    const Quat& orientation = d_orientation.getValue();
    const type::Vec3 viewDirection = orientation.rotate(type::Vec3(0.0, 0.0, -1.0));

    const type::Vec3 newPos = center - viewDirection*distance;
    d_position.setValue(newPos);
}

void BaseCamera::fitBoundingBox(const type::Vec3 &min, const type::Vec3 &max)
{
    SReal diameter = std::max(fabs(max[1]-min[1]), fabs(max[0]-min[0]));
    diameter = std::max((SReal)fabs(max[2]-min[2]), diameter);
    const type::Vec3 center = (min + max)*0.5;

    fitSphere(center,0.5*diameter);

}

void BaseCamera::setView(const type::Vec3& position, const Quat &orientation)
{
    d_position.setValue(position);
    d_orientation.setValue(orientation);
    computeZ();
}

void BaseCamera::setDefaultView(const type::Vec3 & gravity)
{
    const type::Vec3 & minBBox = d_minBBox.getValue();
    const type::Vec3 & maxBBox = d_maxBBox.getValue();
    sceneCenter = (minBBox + maxBBox)*0.5;

    if (b_setDefaultParameters)
    {
        //LookAt
        d_lookAt.setValue(sceneCenter);
        currentLookAt = d_lookAt.getValue();

        //Orientation
        type::Vec3 xAxis(1.0, 0.0, 0.0);
        type::Vec3 yAxis = -gravity;
        // If no gravity defined set the yAxis as 0 1 0;
        if (gravity == type::Vec3(0.0, 0.0, 0.0))
        {
            yAxis = type::Vec3(0.0, 1.0, 0.0);
        }
        yAxis.normalize();

        if (1.0 - fabs(dot(xAxis, yAxis)) < 0.001)
            xAxis = type::Vec3(0.0, 1.0, 0.0);

        type::Vec3 zAxis = xAxis.cross(yAxis);
        zAxis.normalize();
        xAxis = yAxis.cross(zAxis);
        xAxis.normalize();
        Quat q = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
        q.normalize();
        d_orientation.setValue(q);

        //Distance
        const double coeff = 3.0;
        const double dist = (minBBox - sceneCenter).norm() * coeff;
        d_distance.setValue(dist);
        currentDistance = dist;

        //Position
        const type::Vec3 pos = currentLookAt + zAxis*dist;
        d_position.setValue(pos);
    }

    computeZ();
}

void BaseCameraXMLExportSingleParameter(tinyxml2::XMLElement* root, core::objectmodel::BaseData& data, const std::string& comment)
{
    tinyxml2::XMLElement* node = root->GetDocument()->NewElement( data.getName().c_str() );
    node->SetAttribute("value", data.getValueString().c_str() );
    if(!comment.empty())
    {
        tinyxml2::XMLComment* com = root->GetDocument()->NewComment( comment.c_str() );
        root->LinkEndChild(com);
    }
    root->LinkEndChild(node);
}

bool BaseCamera::exportParametersInFile(const std::string& viewFilename)
{
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLDeclaration* decl = doc.NewDeclaration();
    doc.LinkEndChild( decl );

    tinyxml2::XMLElement* root = doc.NewElement( "Camera" );
    root->SetAttribute("version", "1.0" );
    doc.LinkEndChild( root );

    BaseCameraXMLExportSingleParameter(root, d_position, "Vector of 3 reals (x, y, z)");
    BaseCameraXMLExportSingleParameter(root, d_orientation, "Quaternion (x, y, z, w)");
    BaseCameraXMLExportSingleParameter(root, d_fieldOfView, "Real");
    BaseCameraXMLExportSingleParameter(root, d_distance, "Real");
    BaseCameraXMLExportSingleParameter(root, d_zNear, "Real");
    BaseCameraXMLExportSingleParameter(root, d_zFar, "Real");
    BaseCameraXMLExportSingleParameter(root, d_type, "Int (0 -> Perspective, 1 -> Orthographic)");

    return (doc.SaveFile(viewFilename.c_str()) == tinyxml2::XML_SUCCESS);
}

bool BaseCameraXMLImportSingleParameter(tinyxml2::XMLElement* root, core::objectmodel::BaseData& data, BaseCamera* c)
{
    if(root)
    {
        tinyxml2::XMLNode* node = root->FirstChildElement( data.getName().c_str() );
        if(node)
        {
            const tinyxml2::XMLElement* element = node->ToElement();
            if(element)
            {
                const char* attrValue;
                attrValue = element->Attribute("value");
                if(attrValue)
                {
                    std::string m_string; m_string.assign(attrValue);
                    const bool retvalue = data.read(m_string);
                    if(!retvalue)
                        msg_error(c) << "Unreadable value for " << data.getName() << " field.";
                    return retvalue;
                }
                else
                {
                    msg_error(c) << "Attribute value has not been found for " << data.getName() << " field.";
                    return false;
                }
            }
            else
            {
                msg_error(c) << "Unknown error occurred for " << data.getName() << " field.";
                return false;
            }
        }
        else
        {
            msg_error(c) << "Field " << data.getName() << " has not been found.";
            return false;
        }
    }
    else return false;
}

bool BaseCamera::importParametersFromFile(const std::string& viewFilename)
{
    bool result = true;

    msg_info() << "Reading " << viewFilename << " for view parameters.";
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(viewFilename.c_str()) != tinyxml2::XML_SUCCESS)
    {
        result = false;
    }

    tinyxml2::XMLHandle hDoc(&doc);
    tinyxml2::XMLElement* root = hDoc.FirstChildElement().ToElement();

    if (!root)
        result = false;

    if(result)
    {
        BaseCameraXMLImportSingleParameter(root, d_position, this);
        BaseCameraXMLImportSingleParameter(root, d_orientation, this);
        BaseCameraXMLImportSingleParameter(root, d_fieldOfView, this);
        BaseCameraXMLImportSingleParameter(root, d_distance, this);
        BaseCameraXMLImportSingleParameter(root, d_zNear, this);
        BaseCameraXMLImportSingleParameter(root, d_zFar, this);
        BaseCameraXMLImportSingleParameter(root, d_type, this);
    }
    else
    {
        msg_info() << "Error while reading " << viewFilename << ".";
    }
    return result;
}

void BaseCamera::updateOutputData()
{
    //Matrices
    type::vector<SReal>& wModelViewMatrix = *d_modelViewMatrix.beginEdit();
    type::vector<SReal>& wProjectionMatrix = *d_projectionMatrix.beginEdit();

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

    d_modelViewMatrix.endEdit();
    d_projectionMatrix.endEdit();

    //TODO: other info to update
    d_minBBox.setValue(getContext()->f_bbox.getValue().minBBox());
    d_maxBBox.setValue(getContext()->f_bbox.getValue().maxBBox());

    d_zNear.setValue(currentZNear);
    d_zFar.setValue(currentZFar);
}

void BaseCamera::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
        updateOutputData();
}

void BaseCamera::draw(const sofa::core::visual::VisualParams* /*params*/)
{
}

void BaseCamera::drawCamera(const core::visual::VisualParams* vparams)
{
    const auto dt = (vparams->drawTool());
    dt->setPolygonMode(0, true);
    dt->setLightingEnabled(false);

    type::Vec3 camPos = getPosition();
    sofa::type::Vec3 p1, p2, p3, p4;
    p1 = viewportToWorldPoint(type::Vec3(0,0,0.994));
    p2 = viewportToWorldPoint(type::Vec3(1,0,0.994));
    p3 = viewportToWorldPoint(type::Vec3(1,1,0.994));
    p4 = viewportToWorldPoint(type::Vec3(0,1,0.994));

    dt->drawLine(camPos, p1, sofa::type::RGBAColor::black());
    dt->drawLine(camPos, p2, sofa::type::RGBAColor::black());
    dt->drawLine(camPos, p3, sofa::type::RGBAColor::black());
    dt->drawLine(camPos, p4, sofa::type::RGBAColor::black());

    dt->drawLine(p1, p2, sofa::type::RGBAColor::black());
    dt->drawLine(p2, p3, sofa::type::RGBAColor::black());
    dt->drawLine(p3, p4, sofa::type::RGBAColor::black());
    dt->drawLine(p4, p1, sofa::type::RGBAColor::black());

    dt->setPolygonMode(0, false);
    dt->drawTriangles({camPos, p1, p2}, RGBAColor::black());
    dt->setLightingEnabled(true);
}

} // namespace sofa::component::visual
