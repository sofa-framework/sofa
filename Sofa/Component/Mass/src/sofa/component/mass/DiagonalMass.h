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

#include <sofa/component/mass/config.h>

#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/component/mass/VecMassType.h>
#include <sofa/component/mass/RigidMassType.h>

#include <type_traits>

#include <sofa/component/mass/MeshMatrixMass.h>


#define SOFA_ATTRIBUTE_DEPRECATED__DIAGONALMASS() \
    SOFA_ATTRIBUTE_DISABLED( \
        "v22.12 (PR#XXXX)", "v23.06 (PR#2137)", "DiagonalMass is deprecated, it can simply be replaced with MeshMatrixMass..")


namespace sofa::component::mass
{

template<class DataTypes, class TMassType, class GeometricalTypes>
class DiagonalMassInternalData
{
public :
    typedef typename DataTypes::Real Real;
    typedef type::vector<TMassType> MassVector;
    typedef sofa::core::topology::PointData<MassVector> VecMass;

    // In case of non 3D template
    typedef sofa::type::Vec<3,Real> Vec3;
};

/**
* @class    DiagonalMass
* @brief    This component computes the integral of this mass density over the volume of the object geometry but it supposes that the Mass matrix is diagonal.
* @remark   Similar to MeshMatrixMass but it does not simplify the Mass Matrix as diagonal.
* @remark   https://www.sofa-framework.org/community/doc/components/masses/diagonalmass/
* @tparam   DataTypes type of the state associated with this mass
* @tparam   GeometricalTypes type of the geometry, i.e type of the state associated with the topology (if the topology and the mass relates to the same state, this will be the same as DataTypes)
*/
template <class DataTypes, class GeometricalTypes = DataTypes>
class SOFA_ATTRIBUTE_DEPRECATED__DIAGONALMASS() DiagonalMass final : public sofa::core::objectmodel::BaseObject
{
public:

    using TMassType = typename sofa::component::mass::MassType<DataTypes>::type;

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef TMassType MassType;

    typedef typename DiagonalMassInternalData<DataTypes,TMassType,GeometricalTypes>::VecMass VecMass;
    typedef typename DiagonalMassInternalData<DataTypes,TMassType,GeometricalTypes>::MassVector MassVector;

    void init() override;

    /// Construction method called by ObjectFactory
    template<class T>
    static typename T::SPtr create(T*, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();

        using TMeshMatrixMass = typename sofa::component::mass::MeshMatrixMass<DataTypes,GeometricalTypes> ;
        typename TMeshMatrixMass::SPtr meshMatrixMass = sofa::core::objectmodel::New< TMeshMatrixMass >();
        meshMatrixMass->setName(context->getNameHelper().resolveName(meshMatrixMass->getClassName(), core::ComponentNameHelper::Convention::python));

        if (context)
        {
            context->addObject(meshMatrixMass);
        }

        arg->setAttribute("lumping","true");
        meshMatrixMass->parse(arg);

        return obj;
    }

    DiagonalMass() = default;
protected:
    ~DiagonalMass() override = default;
};

#if  !defined(SOFA_COMPONENT_MASS_DIAGONALMASS_CPP)
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec2Types, defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1Types, defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MASS_API DiagonalMass<defaulttype::Vec1Types, defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::mass
