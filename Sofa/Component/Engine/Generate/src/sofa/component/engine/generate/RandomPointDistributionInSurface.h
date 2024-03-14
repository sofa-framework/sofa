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
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/TriangleOctree.h>
#include <sofa/helper/RandomGenerator.h>

namespace sofa::component::engine::generate
{

/**
 * This class extrudes a surface
 */
template <class DataTypes>
class RandomPointDistributionInSurface : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RandomPointDistributionInSurface,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef type::Vec<3,Real> Vec3;

protected:

    RandomPointDistributionInSurface();

    ~RandomPointDistributionInSurface() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    void draw(const core::visual::VisualParams* vparams) override;

    bool initialized;
    helper::RandomGenerator rg;
    Data<unsigned int> randomSeed; ///< Set a specified seed for random generation (0 for "true pseudo-randomness" 
    Data<bool> isVisible; ///< is Visible ?
    Data<bool> drawOutputPoints; ///< Output points visible ?
    Data<Real> minDistanceBetweenPoints; ///< Min Distance between 2 points (-1 for true randomness)
    Data<unsigned int> numberOfInPoints; ///< Number of points inside
    Data<unsigned int> numberOfTests; ///< Number of tests to find if the point is inside or not (odd number)

    Data<VecCoord> f_vertices; ///< Vertices
    Data< type::vector<sofa::core::topology::BaseMeshTopology::Triangle> > f_triangles; ///< Triangles indices
    Data<VecCoord> f_inPoints; ///< Points inside the surface
    Data<VecCoord> f_outPoints; ///< Points outside the surface

    unsigned int safeCounter;
    unsigned int safeLimit;

protected:
    VecCoord directions;
    helper::TriangleOctreeRoot trianglesOctree;
    void getBBox(Vec3 &minBBox, Vec3 &maxBBox);
    void generateRandomDirections();
    Vec3 generateRandomPoint(const Vec3 &minBBox, const Vec3 &maxBBox);
    bool isInside(Coord p);
    bool testDistance(Coord p);
};

#if !defined(SOFA_COMPONENT_ENGINE_RANDOMPOINTDISTRIBUTIONINSURFACE_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API RandomPointDistributionInSurface<defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::engine::generate
