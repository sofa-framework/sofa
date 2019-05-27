/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_NearestPointROI_H
#define SOFA_COMPONENT_ENGINE_NearestPointROI_H
#include "config.h"

#include <sofa/core/behavior/PairInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <set>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/core.h>



namespace sofa
{

namespace component
{

namespace engine
{
/** Attach given pair of particles, projecting the positions of the second particles to the first ones.
*/

using sofa::core::behavior::MechanicalState ;

template <class DataTypes>
class NearestPointROI : public sofa::core::DataEngine
{
public:    
    SOFA_CLASS(SOFA_TEMPLATE(NearestPointROI, DataTypes), sofa::core::DataEngine);

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;

public:
    SetIndex f_indices1; ///< Indices of the source points on the first model
    SetIndex f_indices2; ///< Indices of the fixed points on the second model
    Data<Real> f_radius; ///< Radius to search corresponding fixed point if no indices are given
    SingleLink<NearestPointROI<DataTypes>, MechanicalState<DataTypes>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate1;
    SingleLink<NearestPointROI<DataTypes>, MechanicalState<DataTypes>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate2;

public:
    NearestPointROI();
    ~NearestPointROI() override;

    void init() override;
    void reinit() override;

    void doUpdate() override;
    void handleEvent(core::objectmodel::Event *event) override;

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const NearestPointROI<DataTypes>* = nullptr)
    {
        return DataTypes::Name();
    }
};


#if  !defined(SOFA_COMPONENT_ENGINE_NearestPointROI_CPP)
extern template class SOFA_GENERAL_ENGINE_API NearestPointROI<defaulttype::Vec3Types>;
extern template class SOFA_GENERAL_ENGINE_API NearestPointROI<defaulttype::Vec2Types>;
extern template class SOFA_GENERAL_ENGINE_API NearestPointROI<defaulttype::Vec1Types>;
extern template class SOFA_GENERAL_ENGINE_API NearestPointROI<defaulttype::Rigid3Types>;
extern template class SOFA_GENERAL_ENGINE_API NearestPointROI<defaulttype::Rigid2Types>;
#endif

} // namespace engine

} // namespace component

} // namespace sofa


#endif
