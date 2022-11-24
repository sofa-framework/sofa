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

#include <MultiThreading/config.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::component::engine
{

/**
* This class merge 2 coordinate vectors.
*/
template <class DataTypes>
class MeanComputation : public virtual core::objectmodel::BaseObject
{
    typedef typename DataTypes::Coord         Coord;
    typedef typename DataTypes::VecCoord      VecCoord;
    typedef typename DataTypes::Real          Real;
    //typedef sofa::type::Vec<1, Real>                       Coord1D;
    //typedef sofa::type::Vec<2, Real>                       Coord2D;
    //typedef sofa::type::Vec<3, Real>                       Coord3D;
    typedef sofa::type::vector<Coord>       VectorCoord;
    typedef sofa::type::vector<VecCoord>    VectorVecCoord;
    //typedef sofa::type::vector<Coord3D>    VecCoord3D;

public:
    SOFA_CLASS(SOFA_TEMPLATE(MeanComputation, DataTypes), core::objectmodel::BaseObject);
    //typedef typename DataTypes::VecCoord VecCoord;

protected:

    MeanComputation();

    ~MeanComputation() override {}

    void compute();

public:
    void init() override;

    void reinit() override;

    void handleEvent(core::objectmodel::Event* event) override;

private:

    Data<VecCoord> d_result; ///< Result: mean computed from the input values

    //std::vector<component::container::MechanicalObject<DataTypes>*> _inputMechObjs;

    sofa::type::vector< Data< VecCoord >* > _inputs;

    size_t _resultSize;

};

} // namespace sofa::component::engine

