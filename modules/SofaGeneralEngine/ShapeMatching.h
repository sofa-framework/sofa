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
#ifndef SOFA_COMPONENT_ENGINE_SHAPEMATCHING_H
#define SOFA_COMPONENT_ENGINE_SHAPEMATCHING_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/SVector.h>


namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class computes target positions using shape matching deformation [Muller05][Muller11]
 */
template <class DataTypes>
class ShapeMatching : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ShapeMatching,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename Coord::value_type Real;
    typedef defaulttype::Mat<3,3,Real> Mat3x3;

    typedef core::topology::BaseMeshTopology::PointID ID;
    typedef helper::vector<ID> VI;
    typedef helper::vector<VI> VVI;
	typedef helper::vector<Real> VD;	

public:

    ShapeMatching();

    virtual ~ShapeMatching() {}

    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams* vparams);

    Data<unsigned int> iterations;
    Data< Real > affineRatio;
    Data< Real > fixedweight;
    Data< VecCoord > fixedPosition0;
    Data< VecCoord > fixedPosition;
    Data< VecCoord > position; ///< input (current mstate position)
    Data< VVI > cluster; ///< input2 (clusters)
    Data< VecCoord > targetPosition;       ///< result

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ShapeMatching<DataTypes>* = NULL)    {    return DataTypes::Name();    }

private:
    sofa::core::behavior::MechanicalState<DataTypes>* mstate;
    sofa::core::topology::BaseMeshTopology* topo;

    //rest data
    unsigned int oldRestPositionSize;
    Real oldfixedweight;
    VecCoord Xcm0;
    VecCoord Xcm;
    helper::vector<unsigned int> nbClust;
    VD W;

    helper::vector<Mat3x3> Qxinv; // Qx = sum(X0-Xcm0)(X0-Xcm0)^T
    helper::vector<Mat3x3> T;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_SHAPEMATCHING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API ShapeMatching<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API ShapeMatching<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API ShapeMatching<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API ShapeMatching<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
