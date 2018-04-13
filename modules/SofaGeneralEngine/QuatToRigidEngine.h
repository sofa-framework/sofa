/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_QUATTORIGIDENGINE_H
#define SOFA_COMPONENT_ENGINE_QUATTORIGIDENGINE_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/Quater.h>
#include <sofa/defaulttype/RigidTypes.h>



namespace sofa
{

namespace component
{

namespace engine
{

/*
 * Engine which converts (vectors of) Vec3 + Quaternion to give a (vector of) Rigid
 *
 */

template <class DataTypes>
class QuatToRigidEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(QuatToRigidEngine,sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef sofa::helper::Quater<Real> Quat;
    typedef typename sofa::defaulttype::StdRigidTypes<3,Real>::Coord RigidVec3;
protected:
    QuatToRigidEngine();
    virtual ~QuatToRigidEngine();
public:
    void update() override;
    void init() override;
    void reinit() override;

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* tObj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::objectmodel::BaseObject::create(tObj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const QuatToRigidEngine<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //
    Data<helper::vector<Vec3 > > f_positions; ///< Positions (Vector of 3)
    Data<helper::vector<Quat> > f_orientations; ///< Orientations (Quaternion)
    Data<helper::vector<Vec3 > > f_colinearPositions; ///< Optional positions to restrict output to be colinear in the quaternion Z direction
    Data<helper::vector<RigidVec3> > f_rigids; ///< Rigid (Position + Orientation)
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(QUATTORIGIDENGINE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API QuatToRigidEngine<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API QuatToRigidEngine<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ENGINE_QUATTORIGIDENGINE_H
