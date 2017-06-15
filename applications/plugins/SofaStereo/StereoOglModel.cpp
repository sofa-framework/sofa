/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include "StereoOglModel.h"
#include <sofa/core/ObjectFactory.h>
#include <iostream>

#include <sstream>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <limits>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(StereoOglModel)

int StereoOglModelClass = core::RegisterObject("StereoOglModel")
        .add< StereoOglModel >()
        ;


StereoOglModel::StereoOglModel()
    : textureleft(initData(&textureleft, "textureleft", "Name of the Left Texture"))
    , textureright(initData(&textureright, "textureright", "Name of the Right Texture"))
    , fileMesh(initData(&fileMesh, "fileMesh"," Path to the model"))
    , m_translation     (initData   (&m_translation, Vec3Real(), "translation", "Initial Translation of the object"))
    , m_rotation        (initData   (&m_rotation, Vec3Real(), "rotation", "Initial Rotation of the object"))
    , m_scale           (initData   (&m_scale, Vec3Real(1.0,1.0,1.0), "scale3d", "Initial Scale of the object"))
    , m_scaleTex        (initData   (&m_scaleTex, TexCoord(1.0,1.0), "scaleTex", "Scale of the texture"))
    , m_translationTex  (initData   (&m_translationTex, TexCoord(0.0,0.0), "translationTex", "Translation of the texture"))
{

}

StereoOglModel::~StereoOglModel()
{
}

void StereoOglModel::drawVisual(const core::visual::VisualParams* vparams)
{
    if(!m_camera) return;
    if(m_camera->getCurrentSide() == BaseCamera::LEFT)
    {
        leftModel->drawVisual(vparams);
    }
    if(m_camera->getCurrentSide() == BaseCamera::RIGHT)
    {
        rightModel->drawVisual(vparams);
    }
}

void StereoOglModel::drawTransparent(const core::visual::VisualParams* vparams)
{
    if(!m_camera) return;
    if(m_camera->getCurrentSide() == BaseCamera::LEFT)
    {
        leftModel->drawTransparent(vparams);
    }
    if(m_camera->getCurrentSide() == BaseCamera::RIGHT)
    {
        rightModel->drawTransparent(vparams);
    }
}
void StereoOglModel::drawShadow(const core::visual::VisualParams* vparams)
{
    if(!m_camera) return;
    if(m_camera->getCurrentSide() == BaseCamera::LEFT)
    {
        leftModel->drawShadow(vparams);
    }
    if(m_camera->getCurrentSide() == BaseCamera::RIGHT)
    {
        rightModel->drawShadow(vparams);
    }
}
void StereoOglModel::updateVisual()
{
    if(!m_camera) return;
    if(m_camera->getCurrentSide() == BaseCamera::LEFT)
    {
        leftModel->updateVisual();
    }
    if(m_camera->getCurrentSide() == BaseCamera::RIGHT)
    {
        rightModel->updateVisual();
    }
}
void StereoOglModel::init()
{
    leftModel = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
    leftModel->texturename.setValue(textureleft.getValue());
    leftModel->fileMesh.setValue(fileMesh.getValue());
    leftModel->m_translation.setValue(m_translation.getValue());
    leftModel->m_rotation.setValue(m_rotation.getValue());
    leftModel->m_scale.setValue(m_scale.getValue());
    leftModel->m_scaleTex.setValue(m_scaleTex.getValue());
    leftModel->m_translationTex.setValue(m_translationTex.getValue());
    rightModel = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
    rightModel->texturename.setValue(textureright.getValue());
    rightModel->fileMesh.setValue(fileMesh.getValue());
    rightModel->m_translation.setValue(m_translation.getValue());
    rightModel->m_rotation.setValue(m_rotation.getValue());
    rightModel->m_scale.setValue(m_scale.getValue());
    rightModel->m_scaleTex.setValue(m_scaleTex.getValue());
    rightModel->m_translationTex.setValue(m_translationTex.getValue());

    leftModel->init();
    rightModel->init();

    //getting pointer to the camera
    this->getContext()->get(m_camera,sofa::core::objectmodel::BaseContext::SearchRoot);
    if(!m_camera)
    {
        std::cerr << "StereoCamera not found."<< std::endl;
    }
}
void StereoOglModel::initVisual()
{
    leftModel->initVisual();
    rightModel->initVisual();
}
void StereoOglModel::exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex, int& count)
{
    leftModel->exportOBJ(name,out,mtl,vindex,nindex,tindex,count);
    rightModel->exportOBJ(name,out,mtl,vindex,nindex,tindex,count);
}

void StereoOglModel::computeBBox(const core::ExecParams * params, bool /*onlyVisible*/)
{
    if ( leftModel && rightModel)
    {
        const SReal max_real = std::numeric_limits<SReal>::max();
        const SReal min_real = std::numeric_limits<SReal>::min();

        SReal maxBBox[3] = {min_real,min_real,min_real};
        SReal minBBox[3] = {max_real,max_real,max_real};

        leftModel->computeBBox(params);
        rightModel->computeBBox(params);

        const sofa::defaulttype::Vector3& leftMinBBox = leftModel->f_bbox.getValue().minBBox();
        const sofa::defaulttype::Vector3& leftMaxBBox = leftModel->f_bbox.getValue().maxBBox();
        const sofa::defaulttype::Vector3& rightMinBBox = rightModel->f_bbox.getValue().minBBox();
        const sofa::defaulttype::Vector3& rightMaxBBox = rightModel->f_bbox.getValue().maxBBox();

        for(unsigned int i=0 ; i<3 ; i++)
        {
            minBBox[i] = (leftMinBBox[i] <= rightMinBBox[i]) ? leftMinBBox[i] : rightMinBBox[i];
            maxBBox[i] = (leftMaxBBox[i] >= rightMaxBBox[i]) ? leftMaxBBox[i] : rightMaxBBox[i];
        }


        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
    }
}

} // namespace visual

} // namespace component

} // namespace sofa
