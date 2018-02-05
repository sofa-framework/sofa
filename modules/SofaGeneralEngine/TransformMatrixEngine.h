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
#ifndef SOFA_COMPONENT_ENGINE_TRANSFORMMATRIXENGINE_H
#define SOFA_COMPONENT_ENGINE_TRANSFORMMATRIXENGINE_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>

#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_GENERAL_ENGINE_API AbstractTransformMatrixEngine : public core::DataEngine
{
public:
    SOFA_ABSTRACT_CLASS(AbstractTransformMatrixEngine, core::DataEngine);

protected:
    AbstractTransformMatrixEngine();
    ~AbstractTransformMatrixEngine() {}

    /**
     * Update the transformation, to be implemented in herited classes
     */
    virtual void update() override = 0;

public:
    virtual void init() override;
    virtual void reinit() override;

protected:
    Data<defaulttype::Matrix4> d_inT; // input transformation
    Data<defaulttype::Matrix4> d_outT; // input transformation
};

/**
 * This engine inverts the input transform and outputs the resulting transformation matrix.
 * T_output = T_input^-1
 */
class SOFA_GENERAL_ENGINE_API InvertTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(InvertTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    InvertTransformMatrixEngine() {}
    ~InvertTransformMatrixEngine() {}
    virtual void update() override;
};

/**
 * This engine compose the input transform (if any) with the given translation and outputs the resulting transformation matrix.
 * T_output = T_input * T_translate
 */
class SOFA_GENERAL_ENGINE_API TranslateTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(TranslateTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    TranslateTransformMatrixEngine();
    ~TranslateTransformMatrixEngine() {}
    virtual void update() override;

public:
    virtual void init() override;

protected:
    /// translation
    Data<defaulttype::Vector3> d_translation;

};

/**
 * This engine produces a rotation transformation matrix. It is composed with the input transform if any.
 * T_output = T_input * T_rotate
 */
class SOFA_GENERAL_ENGINE_API RotateTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(RotateTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    RotateTransformMatrixEngine();
    ~RotateTransformMatrixEngine() {}
    virtual void update() override;

public:
    virtual void init() override;

protected:
    /// rotation
    Data<defaulttype::Vector3> d_rotation;

};

/**
 * This engine compose the input transform (if any) with the given scale transformation and outputs the resulting transformation matrix.
 * T_output = T_input * T_scale
 */
class SOFA_GENERAL_ENGINE_API ScaleTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(ScaleTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    ScaleTransformMatrixEngine();
    ~ScaleTransformMatrixEngine() {}
    virtual void update() override;

public:
    virtual void init() override;

protected:
    Data<defaulttype::Vector3> d_scale; // scale
};

} // namespace engine

} // namespace component

} // namespace sofa

#endif
