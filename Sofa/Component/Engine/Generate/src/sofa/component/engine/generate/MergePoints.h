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
#include <sofa/component/engine/generate/config.h>

#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::generate
{

/**
 * This class merge 2 coordinate vectors.
 */
template <class DataTypes>
class MergePoints : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MergePoints,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;

protected:

    MergePoints();

    ~MergePoints() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    bool           initDone;

    Data<VecCoord> f_X1; ///< position coordinates of the degrees of freedom of the first object
    Data<VecCoord> f_X2; ///< Rest position coordinates of the degrees of freedom of the second object
    Data<SetIndex> f_X2_mapping; ///< Mapping of indices to inject position2 inside position1 vertex buffer
    Data<SetIndex> f_indices1; ///< Indices of the points of the first object
    Data<SetIndex> f_indices2; ///< Indices of the points of the second object
    Data<VecCoord> f_points; ///< position coordinates resulting from the merge
    Data<bool>     f_noUpdate; ///< do not update the output at each time step (false)
};

#if !defined(SOFA_COMPONENT_ENGINE_MERGEPOINTS_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergePoints<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergePoints<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergePoints<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergePoints<defaulttype::Rigid2Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MergePoints<defaulttype::Rigid3Types>;
 
#endif

} //namespace sofa::component::engine::generate
