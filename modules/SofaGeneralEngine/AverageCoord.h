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
#ifndef SOFA_COMPONENT_ENGINE_AverageCoord_H
#define SOFA_COMPONENT_ENGINE_AverageCoord_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class computes the average of a set of Coordinates
 */
template <class DataTypes>
class AverageCoord : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AverageCoord,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef unsigned int Index;
    typedef sofa::helper::vector<Index> VecIndex;

protected:

    AverageCoord();

    virtual ~AverageCoord() {}
public:
    void init();

    void reinit();

    void update();

    Data<VecIndex> d_indices;    ///< indices of the coordinates to average
    Data<unsigned> d_vecId;  ///< index of the vector (default value corresponds to core::VecCoordId::position() )
    Data<Coord> d_average;       ///< result

    void handleEvent(core::objectmodel::Event *event);
    void onBeginAnimationStep(const double /*dt*/);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }


    static std::string templateName(const AverageCoord<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }



protected:
    sofa::core::behavior::MechanicalState<DataTypes> *mstate;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_AverageCoord_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Vec2dTypes>;
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Rigid2dTypes>;
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Vec2fTypes>;
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Rigid2fTypes>;
extern template class SOFA_GENERAL_ENGINE_API AverageCoord<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
