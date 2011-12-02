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
#ifndef SOFA_COMPONENT_ENGINE_SHAPEMATCHING_H
#define SOFA_COMPONENT_ENGINE_SHAPEMATCHING_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/SVector.h>

#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace core::behavior;
using namespace core::topology;
using namespace core::objectmodel;

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

public:

    ShapeMatching();

    virtual ~ShapeMatching() {}

    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams* vparams);

    Data<unsigned int> iterations;
    Data<Real> affineRatio;
    Data< VecCoord > position; ///< input (current mstate position)
    Data< VVI > cluster; ///< input2 (clusters)
    Data< VecCoord > targetPosition;       ///< result

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ShapeMatching<DataTypes>* = NULL)    {    return DataTypes::Name();    }

private:
    MechanicalState<DataTypes>* mstate;
    BaseMeshTopology* topo;

    //rest data
    unsigned int oldRestPositionSize;
    unsigned int oldClusterSize;
    unsigned int oldClusterSize0;
    VecCoord Xcm0;
    VecCoord Xcm;
    helper::vector<unsigned int> nbClust;

    helper::vector<Mat3x3> Qxinv; // Qx = sum(X0-Xcm0)(X0-Xcm0)^T
    helper::vector<Mat3x3> T;

    template<class Real, int Dim>
    inline Mat<Dim, Dim, Real> covNN(const Vec<Dim,Real>& v1, const Vec<Dim,Real>& v2)
    {
        Mat<Dim, Dim, Real> res;
        for ( unsigned int i = 0; i < Dim; ++i)
            for ( unsigned int j = 0; j < Dim; ++j)
            {
                res[i][j] = v1[i] * v2[j];
            }
        return res;
    }

};


#if defined(WIN32) && !defined(SOFA_COMPONENT_ENGINE_SHAPEMATCHING_CPP)
#ifndef SOFA_FLOAT
template class SOFA_ENGINE_API ShapeMatching<defaulttype::Vec3dTypes>;
template class SOFA_ENGINE_API ShapeMatching<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_ENGINE_API ShapeMatching<defaulttype::Vec3fTypes>;
template class SOFA_ENGINE_API ShapeMatching<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
