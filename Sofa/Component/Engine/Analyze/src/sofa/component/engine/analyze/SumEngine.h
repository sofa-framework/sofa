 
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
#include <sofa/component/engine/analyze/config.h>
#include <sofa/core/DataEngine.h>
#include <sofa/type/Vec.h>

namespace sofa::component::engine::analyze
{

/// Computing the Sum between two vector of dofs
/// output = input - substractor
template <class TDataType>
class SumEngine : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SumEngine,TDataType),core::DataEngine);

    typedef TDataType DataType;
    typedef type::vector<DataType> VecData;


    SumEngine();

    ~SumEngine() override {}

    void init() override;
    void reinit() override;
    void doUpdate() override;

protected:
    Data<VecData> d_input; ///< input vector
    Data<DataType> d_output; ///< output sum
};

#if !defined(SOFA_COMPONENT_ENGINE_SumEngine_CPP)
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API SumEngine<type::Vec1>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API SumEngine<type::Vec3>;
#endif

} //namespace sofa::component::engine::analyze
