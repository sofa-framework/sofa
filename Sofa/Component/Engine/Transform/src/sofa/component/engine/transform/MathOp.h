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

#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::engine::transform
{

/**
 * Apply a math operation to combine several inputs
 */
template <class VecT>
class MathOp : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MathOp,VecT),core::DataEngine);
    typedef VecT VecValue;
    typedef typename VecValue::value_type Value;

protected:
    MathOp();

    ~MathOp() override;
public:
    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str ) override;

    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data<unsigned int> f_nbInputs; ///< Number of input values
    type::vector<Data<VecValue>*> vf_inputs;
    sofa::core::objectmodel::Data< sofa::helper::OptionsGroup > f_op; ///< Selected operation to apply
    Data<VecValue> f_output; ///< Output values

protected:
    void createInputs(int nb = -1);
};

#if !defined(SOFA_COMPONENT_ENGINE_MATHOP_CPP)

extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< type::vector<int> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< type::vector<bool> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< type::vector<SReal> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< type::vector<type::Vec2> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< type::vector<type::Vec3> >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< defaulttype::Rigid2Types::VecCoord >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< defaulttype::Rigid2Types::VecDeriv >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< defaulttype::Rigid3Types::VecCoord >;
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API MathOp< defaulttype::Rigid3Types::VecDeriv >;

#endif

} //namespace sofa::component::engine::transform
