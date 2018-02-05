/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
/*
 * GlslModel.h
 *
 *  Created on: 9 févr. 2009
 *      Author: froy
 */

#ifndef OGLSHADERVISUALMODEL_H_
#define OGLSHADERVISUALMODEL_H_
#include "config.h"

#include <SofaOpenglVisual/OglModel.h>
#include <SofaOpenglVisual/OglShader.h>
#include <SofaOpenglVisual/OglAttribute.h>
#include <SofaOpenglVisual/OglVariable.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{


class SOFA_OPENGL_VISUAL_API OglShaderVisualModel : public OglModel
{
public:
    SOFA_CLASS(OglShaderVisualModel, OglModel);

protected:

    typedef sofa::defaulttype::ExtVec3fTypes::Coord Coord;
    typedef sofa::defaulttype::ExtVec3fTypes::VecCoord VecCoord;

    GLuint abo;
    OglShader* shader;
    int restPosition_lastUpdate;
public:
    // These attributes are public due to dynamic topologies updates.
    OglFloat3Attribute* vrestpositions;
    OglFloat3Attribute* vrestnormals;

    OglMatrix4Variable* modelMatrixUniform;
protected:
    OglShaderVisualModel();
    virtual ~OglShaderVisualModel();
public:
    void init() override;
    void initVisual() override;

    void updateVisual() override;

    //void putRestPositions(const Vec3fTypes::VecCoord& positions);

    virtual void bwdDraw(core::visual::VisualParams*) override;
    virtual void fwdDraw(core::visual::VisualParams*) override;

    // handle topological changes
    virtual void handleTopologyChange() override;
    void computeRestPositions();
    void computeRestNormals();

private:
    virtual void pushTransformMatrix(float* matrix) override;
    virtual void popTransformMatrix() override;


};

} //namespace visualmodel

} //namespace component

} //namespace sofa

#endif /* OGLSHADERVISUALMODEL_H_ */
