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
#ifndef SOFA_COMPONENT_ENGINE_NormalsFromPoints_H
#define SOFA_COMPONENT_ENGINE_NormalsFromPoints_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec3Types.h>


namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class compute vertex normals by averaging face normals
 */
template <class DataTypes>
class NormalsFromPoints : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NormalsFromPoints,DataTypes),core::DataEngine);
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

protected:

    NormalsFromPoints();

    virtual ~NormalsFromPoints() {}
public:
    void init() override;

    void reinit() override;

    void update() override;

    Data< VecCoord > position;
    Data< helper::vector< helper::fixed_array <unsigned int,3> > > triangles;
    Data< helper::vector< helper::fixed_array <unsigned int,4> > > quads;
    Data< VecCoord > normals;       ///< result
    Data<bool> invertNormals;
    Data<bool> useAngles;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const NormalsFromPoints<DataTypes>* = NULL) { return DataTypes::Name();    }

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_NormalsFromPoints_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API NormalsFromPoints<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API NormalsFromPoints<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
