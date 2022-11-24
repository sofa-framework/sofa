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
#pragma once
#include <sofa/component/engine/transform/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::engine::transform
{

class SOFA_COMPONENT_ENGINE_TRANSFORM_API AbstractTransformMatrixEngine : public core::DataEngine
{
public:
    SOFA_ABSTRACT_CLASS(AbstractTransformMatrixEngine, core::DataEngine);

protected:
    AbstractTransformMatrixEngine();
    ~AbstractTransformMatrixEngine() override {}

    /**
     * Update the transformation, to be implemented in herited classes
     */
    void doUpdate() override = 0;

public:
    void init() override;
    void reinit() override;

protected:
    Data<type::Matrix4> d_inT; ///< input transformation
    Data<type::Matrix4> d_outT; ///< input transformation
};

/**
 * This engine inverts the input transform and outputs the resulting transformation matrix.
 * T_output = T_input^-1
 */
class SOFA_COMPONENT_ENGINE_TRANSFORM_API InvertTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(InvertTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    InvertTransformMatrixEngine() {}
    ~InvertTransformMatrixEngine() override {}
    void doUpdate() override;
};

/**
 * This engine compose the input transform (if any) with the given translation and outputs the resulting transformation matrix.
 * T_output = T_input * T_translate
 */
class SOFA_COMPONENT_ENGINE_TRANSFORM_API TranslateTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(TranslateTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    TranslateTransformMatrixEngine();
    ~TranslateTransformMatrixEngine() override {}
    void doUpdate() override;

public:
    void init() override;

protected:
    /// translation
    Data<type::Vec3> d_translation;

};

/**
 * This engine produces a rotation transformation matrix. It is composed with the input transform if any.
 * T_output = T_input * T_rotate
 */
class SOFA_COMPONENT_ENGINE_TRANSFORM_API RotateTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(RotateTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    RotateTransformMatrixEngine();
    ~RotateTransformMatrixEngine() override {}
    void doUpdate() override;

public:
    void init() override;

protected:
    /// rotation
    Data<type::Vec3> d_rotation;

};

/**
 * This engine compose the input transform (if any) with the given scale transformation and outputs the resulting transformation matrix.
 * T_output = T_input * T_scale
 */
class SOFA_COMPONENT_ENGINE_TRANSFORM_API ScaleTransformMatrixEngine : public AbstractTransformMatrixEngine
{
public:
    SOFA_CLASS(ScaleTransformMatrixEngine, AbstractTransformMatrixEngine);

protected:
    ScaleTransformMatrixEngine();
    ~ScaleTransformMatrixEngine() override {}
    void doUpdate() override;

public:
    void init() override;

protected:
    Data<type::Vec3> d_scale; ///< scale
};

} //namespace sofa::component::engine::transform
