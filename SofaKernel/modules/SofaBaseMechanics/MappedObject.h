/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MAPPEDOBJECT_H
#define SOFA_COMPONENT_MAPPEDOBJECT_H
#include "config.h"

#include <sofa/core/State.h>
#include <vector>
#include <assert.h>
#include <fstream>

namespace sofa
{

namespace component
{

namespace container
{
//using core::objectmodel::Data;

/// This class can be overridden if needed for additionnal storage within template specializations.
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
    virtual void init();

    Data<VecCoord> f_X;
    Data<VecDeriv> f_V;

    virtual void resize(size_t vsize) { f_X.beginEdit()->resize(vsize); f_X.endEdit(); f_V.beginEdit()->resize(vsize); f_V.endEdit(); }

    VecCoord* getX()  { return f_X.beginEdit(); }
    VecDeriv* getV()  { return f_V.beginEdit(); }

    const VecCoord* getX()  const { return &f_X.getValue();  }
    const VecDeriv* getV()  const { return &f_V.getValue();  }

    size_t getSize() const
    {
        return f_X.getValue().size();
    }

    Data< VecCoord >* write(core::VecCoordId v)
    {
        if(v == core::VecCoordId::position())
            return &f_X;

        return NULL;
    }

    const Data< VecCoord >* read(core::ConstVecCoordId v) const
    {
        if(v == core::ConstVecCoordId::position())
            return &f_X;
        else
            return NULL;
    }

    Data< VecDeriv >* write(core::VecDerivId v)
    {
        if(v == core::VecDerivId::velocity())
            return &f_V;
        else
            return NULL;
    }

    const Data< VecDeriv >* read(core::ConstVecDerivId v) const
    {
        if(v == core::ConstVecDerivId::velocity())
            return &f_V;
        else
            return NULL;
    }

    Data< MatrixDeriv >* write(core::MatrixDerivId /*v*/)
    {
        return NULL;
    }

    const Data< MatrixDeriv >* read(core::ConstMatrixDerivId /*v*/) const
    {
        return NULL;
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONTAINER_MAPPEDOBJECT_CPP)
#ifndef SOFA_FLOAT
extern template class MappedObject<defaulttype::Vec3dTypes>;
extern template class MappedObject<defaulttype::Vec2dTypes>;
extern template class MappedObject<defaulttype::Vec1dTypes>;
extern template class MappedObject<defaulttype::Vec6dTypes>;
extern template class MappedObject<defaulttype::Rigid3dTypes>;
extern template class MappedObject<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class MappedObject<defaulttype::Vec3fTypes>;
extern template class MappedObject<defaulttype::Vec2fTypes>;
extern template class MappedObject<defaulttype::Vec1fTypes>;
extern template class MappedObject<defaulttype::Vec6fTypes>;
extern template class MappedObject<defaulttype::Rigid3fTypes>;
extern template class MappedObject<defaulttype::Rigid2fTypes>;
#endif
extern template class MappedObject<defaulttype::LaparoscopicRigid3Types>;
#endif

} // namespace container

} // namespace component

} // namespace sofa

#endif
