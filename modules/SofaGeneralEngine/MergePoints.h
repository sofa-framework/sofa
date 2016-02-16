/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "config.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

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
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;

protected:

    MergePoints();

    ~MergePoints() {}
public:
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

    bool           initDone;

    Data<VecCoord> f_X1;
    Data<VecCoord> f_X2;
    Data<SetIndex> f_X2_mapping;
    Data<SetIndex> f_indices1;
    Data<SetIndex> f_indices2;
    Data<VecCoord> f_points;
    Data<bool>     f_noUpdate;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MERGEPOINTS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Vec1dTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Vec2dTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Vec3dTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Rigid2dTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Vec1fTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Vec2fTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Vec3fTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Rigid2fTypes>;
extern template class SOFA_ENGINE_API MergePoints<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
