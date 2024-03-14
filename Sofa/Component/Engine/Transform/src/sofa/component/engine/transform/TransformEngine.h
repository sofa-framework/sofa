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

#include <sofa/type/Quat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::transform
{

/**
 * This class transforms the positions of one DataFields into new positions after applying a transformation
This transformation can be either translation, rotation, scale
 */
template <class DataTypes>
class TransformEngine : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TransformEngine,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

protected:

    TransformEngine();

    ~TransformEngine() override {}
public:
    void init() override;
    void reinit() override;

protected:
    void doUpdate() override;

    Data<VecCoord> f_inputX; ///< input position
    Data<VecCoord> f_outputX; ///< ouput position
    Data<type::Vec3> translation; ///< translation
    Data<type::Vec3> rotation; ///< rotation
    Data<type::Quat<SReal>> quaternion; ///< quaternion rotation
    Data<type::Vec3> scale; ///< scale
    Data<bool> inverse; ///< true to apply inverse transformation
};

#if !defined(SOFA_COMPONENT_ENGINE_TRANSFORMENGINE_CPP)
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Rigid2Types>;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API TransformEngine<defaulttype::Rigid3Types>;
#endif

} //namespace sofa::component::engine::transform
