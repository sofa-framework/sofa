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
#ifndef SOFA_STEREO_PLUGIN_STEREOOGLMODEL_H
#define SOFA_STEREO_PLUGIN_STEREOOGLMODEL_H

#include <sofa/core/visual/VisualModel.h>
#include <SofaOpenglVisual/OglModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaStereo/StereoCamera.h>

#define   NB_MAX_TEXTURES 16

namespace sofa
{

namespace component
{

namespace visualmodel
{

class StereoOglModel : public core::visual::VisualModel
{

protected:
    sofa::core::objectmodel::DataFileName textureleft;
    sofa::core::objectmodel::DataFileName textureright;
    sofa::core::objectmodel::DataFileName fileMesh;

    typedef sofa::defaulttype::ExtVec3fTypes DataTypes;
    typedef DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<2, float> TexCoord;
    typedef sofa::defaulttype::Vec<3,Real> Vec3Real;
    Data< Vec3Real > m_translation; ///< Initial Translation of the object
    Data< Vec3Real > m_rotation; ///< Initial Rotation of the object
    Data< Vec3Real > m_scale; ///< Initial Scale of the object

    Data< TexCoord > m_scaleTex; ///< Scale of the texture
    Data< TexCoord > m_translationTex; ///< Translation of the texture

public:

    StereoOglModel();
    ~StereoOglModel();

    //VisualModel functions
    void drawVisual(const core::visual::VisualParams* vparams);
    void drawTransparent(const core::visual::VisualParams* vparams);
    void drawShadow(const core::visual::VisualParams* vparams);
    void updateVisual();
    void init();
    void initVisual();
    void exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex, int& count);
    void computeBBox(const core::ExecParams * params, bool /*onlyVisible*/);

private:
    sofa::component::visualmodel::OglModel::SPtr leftModel;
    sofa::component::visualmodel::OglModel::SPtr rightModel;
    sofa::component::visualmodel::StereoCamera::SPtr m_camera;

};

} // namespace visual

} // namespace component

} // namespace sofa

#endif // SOFA_STEREO_PLUGIN_TEST_H
