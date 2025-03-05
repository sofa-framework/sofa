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
#include <sofa/component/statecontainer/config.h>

#include <sofa/core/State.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::statecontainer
{
//using core::objectmodel::Data;

/// This class can be overridden if needed for additional storage within template specializations.
template<class DataTypes>
class MappedObjectInternalData
{
public:
};

template <class DataTypes>
class MappedObject : public core::State<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MappedObject,DataTypes), SOFA_TEMPLATE(core::State,DataTypes));
    typedef core::State<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

protected:

    MappedObjectInternalData<DataTypes> data;



    MappedObject();

    virtual ~MappedObject();
public:
    void init() override;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_STATECONTAINER()
    sofa::core::objectmodel::lifecycle::RenamedData<VecCoord> f_X;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_STATECONTAINER()
    sofa::core::objectmodel::lifecycle::RenamedData<VecDeriv> f_V;

    Data<VecCoord> d_X; ///< position vector
    Data<VecDeriv> d_V; ///< velocity vector

    void resize(Size vsize) override { d_X.beginEdit()->resize(vsize); d_X.endEdit(); d_V.beginEdit()->resize(vsize); d_V.endEdit(); }

    VecCoord* getX()  { return d_X.beginEdit(); }
    VecDeriv* getV()  { return d_V.beginEdit(); }

    const VecCoord* getX()  const { return &d_X.getValue();  }
    const VecDeriv* getV()  const { return &d_V.getValue();  }

    Size getSize() const override
    {
        return Size(d_X.getValue().size());
    }

    Data< VecCoord >* write(core::VecCoordId v) override
    {
        if(v == core::vec_id::write_access::position)
            return &d_X;

        return nullptr;
    }

    const Data< VecCoord >* read(core::ConstVecCoordId v) const override
    {
        if(v == core::vec_id::read_access::position)
            return &d_X;
        else
            return nullptr;
    }

    Data< VecDeriv >* write(core::VecDerivId v) override
    {
        if(v == core::vec_id::write_access::velocity)
            return &d_V;
        else
            return nullptr;
    }

    const Data< VecDeriv >* read(core::ConstVecDerivId v) const override
    {
        if(v == core::vec_id::read_access::velocity)
            return &d_V;
        else
            return nullptr;
    }

    Data< MatrixDeriv >* write(core::MatrixDerivId /*v*/) override
    {
        return nullptr;
    }

    const Data< MatrixDeriv >* read(core::ConstMatrixDerivId /*v*/) const override
    {
        return nullptr;
    }
};

#if !defined(SOFA_COMPONENT_CONTAINER_MAPPEDOBJECT_CPP)
extern template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_STATECONTAINER_API MappedObject<defaulttype::Rigid2Types>;
#endif

} // namespace sofa::component::statecontainer
