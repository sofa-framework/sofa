/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_ENGINE_RIGIDTOQUATENGINE_H
#define SOFA_COMPONENT_ENGINE_RIGIDTOQUATENGINE_H
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
 * Engine which converts a (vector of) Rigid to give a (vectors of) Vec3 + Quaternion
 *
 */

template <class DataTypes>
class RigidToQuatEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(RigidToQuatEngine,sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef sofa::helper::Quater<Real> Quat;
    typedef typename sofa::defaulttype::StdRigidTypes<3,Real>::Coord RigidVec3;
protected:
    RigidToQuatEngine();
    virtual ~RigidToQuatEngine();
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

    static std::string templateName(const RigidToQuatEngine<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //
    Data<helper::vector<Vec3 > > f_positions;
    Data<helper::vector<Quat> > f_orientations;
    Data<helper::vector<RigidVec3> > f_rigids;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(RIGIDTOQUATENGINE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API RigidToQuatEngine<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API RigidToQuatEngine<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ENGINE_RIGIDTOQUATENGINE_H
