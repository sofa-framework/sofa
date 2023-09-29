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
#include <sofa/type/vector.h>
#include <sofa/type/Quat.h>
#include <sofa/defaulttype/RigidTypes.h>



namespace sofa::component::engine::transform
{

/*
 * Engine which converts a (vector of) Rigid to give a (vectors of) Vec3 + Quaternion
 *
 */

template <class DataTypes>
class RigidToQuatEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(RigidToQuatEngine,sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef sofa::type::Vec<3,Real> Vec3;
    typedef sofa::type::Quat<Real> Quat;
    typedef typename sofa::defaulttype::StdRigidTypes<3,Real>::Coord RigidVec3;
protected:
    RigidToQuatEngine();
    ~RigidToQuatEngine() override;
public:
    void doUpdate() override;
    void init() override;
    void reinit() override;

    Data<type::vector<Vec3 > > f_positions; ///< Positions (Vector of 3)
    Data<type::vector<Quat> > f_orientations; ///< Orientations (Quaternion)
    Data<type::vector<Vec3> > f_orientationsEuler; ///< Orientation (Euler angle)
    Data<type::vector<RigidVec3> > f_rigids; ///< Rigid (Position + Orientation)
};

#if !defined(RIGIDTOQUATENGINE_CPP)
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API RigidToQuatEngine<defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::engine::transform
