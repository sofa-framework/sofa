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
#ifndef SOFA_COMPONENT_ENGINE_MERGEVECTORS_H
#define SOFA_COMPONENT_ENGINE_MERGEVECTORS_H
#include "config.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/vectorData.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * Apply a merge operation to combine several inputs
 */
template <class VecT>
class MergeVectors : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MergeVectors,VecT),core::DataEngine);
    typedef VecT VecValue;
    typedef typename VecValue::value_type Value;

protected:
    MergeVectors();

    virtual ~MergeVectors();
public:
    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse( sofa::core::objectmodel::BaseObjectDescription* arg );

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields( const std::map<std::string,std::string*>& str );

    void init();

    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MergeVectors<VecT>* = NULL)
    {
        return Data<Value>::templateName();
    }

    Data<unsigned int> f_nbInputs;
    helper::vectorData<VecValue> vf_inputs;
    Data<VecValue> f_output;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MERGEVECTORS_CPP)

extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<int> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<bool> >;
//extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<std::string> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<defaulttype::Vec2u> >;
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<double> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<defaulttype::Vec2d> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<defaulttype::Vec3d> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<defaulttype::Vec4d> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid2dTypes::VecCoord >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid2dTypes::VecDeriv >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid3dTypes::VecCoord >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid3dTypes::VecDeriv >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<float> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<defaulttype::Vec2f> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<defaulttype::Vec3f> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< helper::vector<defaulttype::Vec4f> >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid2fTypes::VecCoord >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid2fTypes::VecDeriv >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid3fTypes::VecCoord >;
extern template class SOFA_GENERAL_ENGINE_API MergeVectors< defaulttype::Rigid3fTypes::VecDeriv >;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
