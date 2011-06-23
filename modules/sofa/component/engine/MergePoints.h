/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_MERGEPOINTS_H
#define SOFA_COMPONENT_ENGINE_MERGEPOINTS_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace engine
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
    typedef topology::PointSubset SetIndex;

public:

    MergePoints();

    ~MergePoints() {}

    void init();

    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MergePoints<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    Data<VecCoord> f_X1;
    Data<VecCoord> f_X2;
    Data<SetIndex> f_indices1;
    Data<SetIndex> f_indices2;
    Data<VecCoord> f_points;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_ENGINE_MERGEPOINTS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Vec1dTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Vec2dTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Vec3dTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Rigid2dTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Vec1fTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Vec2fTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Vec3fTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Rigid2fTypes>;
template class SOFA_COMPONENT_ENGINE_API MergePoints<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
