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
/*
 * FlowVisualModel.h
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#ifndef FLOWVISUALMODEL_H_
#define FLOWVISUALMODEL_H_

#include <sofa/component/component.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/ManifoldTriangleSetTopologyContainer.h>
#include <sofa/component/topology/ManifoldTetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.inl>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/core/visual/Shader.h>
namespace sofa
{

namespace component
{

namespace visualmodel
{

//static unsigned int COLORMAP_SIZE;

static defaulttype::Vec3f ColorMap[64] =
{
    defaulttype::Vec3f( 0.0,        0.0,       0.5625 ),
    defaulttype::Vec3f( 0.0,        0.0,       0.625  ),
    defaulttype::Vec3f( 0.0,        0.0,       0.6875 ),
    defaulttype::Vec3f( 0.0,        0.0,         0.75 ),
    defaulttype::Vec3f( 0.0,        0.0,       0.8125 ),
    defaulttype::Vec3f( 0.0,        0.0,        0.875 ),
    defaulttype::Vec3f( 0.0,        0.0,       0.9375 ),
    defaulttype::Vec3f( 0.0,        0.0,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.0625,          1.0 ),
    defaulttype::Vec3f( 0.0,      0.125,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.1875,          1.0 ),
    defaulttype::Vec3f( 0.0,       0.25,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.3125,          1.0 ),
    defaulttype::Vec3f( 0.0,      0.375,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.4375,          1.0 ),
    defaulttype::Vec3f( 0.0,        0.5,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.5625,          1.0 ),
    defaulttype::Vec3f( 0.0,      0.625,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.6875,          1.0 ),
    defaulttype::Vec3f( 0.0,       0.75,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.8125,          1.0 ),
    defaulttype::Vec3f( 0.0,     0.875,           1.0 ),
    defaulttype::Vec3f( 0.0,     0.9375,          1.0 ),
    defaulttype::Vec3f( 0.0,        1.0,          1.0 ),
    defaulttype::Vec3f( 0.0625,     1.0,          1.0 ),
    defaulttype::Vec3f( 0.125,      1.0,       0.9375 ),
    defaulttype::Vec3f( 0.1875,     1.0,        0.875 ),
    defaulttype::Vec3f( 0.25,       1.0,       0.8125 ),
    defaulttype::Vec3f( 0.3125,     1.0,         0.75 ),
    defaulttype::Vec3f( 0.375,      1.0,       0.6875 ),
    defaulttype::Vec3f( 0.4375,     1.0,        0.625 ),
    defaulttype::Vec3f( 0.5,        1.0,       0.5625 ),
    defaulttype::Vec3f( 0.5625,     1.0,          0.5 ),
    defaulttype::Vec3f( 0.625,      1.0,       0.4375 ),
    defaulttype::Vec3f( 0.6875,     1.0,        0.375 ),
    defaulttype::Vec3f( 0.75,       1.0,       0.3125 ),
    defaulttype::Vec3f( 0.8125,     1.0,         0.25 ),
    defaulttype::Vec3f( 0.875,      1.0,       0.1875 ),
    defaulttype::Vec3f( 0.9375,     1.0,        0.125 ),
    defaulttype::Vec3f( 1.0,        1.0,       0.0625 ),
    defaulttype::Vec3f( 1.0,        1.0,          0.0 ),
    defaulttype::Vec3f( 1.0,       0.9375,        0.0 ),
    defaulttype::Vec3f( 1.0,        0.875,        0.0 ),
    defaulttype::Vec3f( 1.0,       0.8125,        0.0 ),
    defaulttype::Vec3f( 1.0,         0.75,        0.0 ),
    defaulttype::Vec3f( 1.0,       0.6875,        0.0 ),
    defaulttype::Vec3f( 1.0,        0.625,        0.0 ),
    defaulttype::Vec3f( 1.0,       0.5625,        0.0 ),
    defaulttype::Vec3f( 1.0,          0.5,        0.0 ),
    defaulttype::Vec3f( 1.0,       0.4375,        0.0 ),
    defaulttype::Vec3f( 1.0,        0.375,        0.0 ),
    defaulttype::Vec3f( 1.0,       0.3125,        0.0 ),
    defaulttype::Vec3f( 1.0,         0.25,        0.0 ),
    defaulttype::Vec3f( 1.0,       0.1875,        0.0 ),
    defaulttype::Vec3f( 1.0,        0.125,        0.0 ),
    defaulttype::Vec3f( 1.0,       0.0625,        0.0 ),
    defaulttype::Vec3f( 1.0,          0.0,        0.0 ),
    defaulttype::Vec3f( 0.9375,       0.0,        0.0 ),
    defaulttype::Vec3f( 0.875,        0.0,        0.0 ),
    defaulttype::Vec3f( 0.8125,       0.0,        0.0 ),
    defaulttype::Vec3f( 0.75,         0.0,        0.0 ),
    defaulttype::Vec3f( 0.6875,       0.0,        0.0 ),
    defaulttype::Vec3f( 0.625,        0.0,        0.0 ),
    defaulttype::Vec3f( 0.5625,       0.0,        0.0 )
};

template <class DataTypes>
class SOFA_COMPONENT_VISUALMODEL_API FlowVisualModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FlowVisualModel,DataTypes), core::visual::VisualModel);

    typedef typename core::behavior::MechanicalState<DataTypes> FluidState;
    typedef typename core::behavior::MechanicalState<DataTypes> TetraGeometry;
    typedef typename core::behavior::MechanicalState<DataTypes> TriangleGeometry;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    struct StreamLine
    {
        helper::vector<Coord> positions;
        unsigned int currentPrimitiveID;
        helper::set<unsigned int> primitivesAroundLastPoint;
    };

protected:
    TriangleGeometry* triangleGeometry;
    TetraGeometry* tetraGeometry;
    FluidState* tetraCenters;
    core::behavior::MechanicalState<DataTypes>* surfaceVolume;

    topology::ManifoldTriangleSetTopologyContainer* m_triTopo;
    topology::TriangleSetGeometryAlgorithms<DataTypes>* m_triGeo;
    topology::ManifoldTetrahedronSetTopologyContainer* m_tetraTopo;
    topology::TetrahedronSetGeometryAlgorithms<DataTypes>* m_tetraGeo;
    //draw tetrahedra
    core::visual::Shader* shader;

    VecCoord triangleCenters;
    VecDeriv velocityAtTriangleVertex;
    VecDeriv velocityAtTetraVertex;
    helper::vector<bool> isPointInTetra;
    helper::vector< helper::vector<unsigned int> >  tetraShellPerTriangleVertex;
    helper::vector< helper::vector<unsigned int> >  tetraShellPerTetraVertex;
    helper::vector< float > tetraSize;
    helper::vector<double> normAtTriangleVertex;
    helper::vector<double> normAtTetraVertex;
    helper::vector<StreamLine> streamLines;
    double meanEdgeLength;
    double maximumVelocityAtTriangleVertex, minimumVelocityAtTriangleVertex;
    double maximumVelocityAtTetraVertex, minimumVelocityAtTetraVertex;

    unsigned int getIndexClosestPoint(const VecCoord &x, Coord p);
    bool isInDomain(unsigned int index,Coord p);
    bool isInDomainT(unsigned int index,Coord p);
    Coord interpolateVelocity(unsigned int index, Coord p, bool &atEnd);
    void interpolateVelocityAtTriangleVertices();
    void interpolateVelocityAtTetraVertices();

public:
    static const double STREAMLINE_NUMBER_OF_POINTS_BY_TRIANGLE;

    Data<std::string> m_tag2D, m_tag3D;

    Data<bool> showVelocityLines;
    Data<double> viewVelocityFactor;
    Data<double> velocityMin;
    Data<double> velocityMax;
    Data<bool> showStreamLines;
    Data<helper::vector<Coord> > streamlineSeeds;
    Data<unsigned int> streamlineMaxNumberOfPoints;
    Data<double> streamlineDtNumberOfPointsPerTriangle;
    Data<bool> showColorScale;
    Data<bool> showTetrahedra;
    Data<float> minAlpha;
    Data<float> maxAlpha;
    FlowVisualModel();
    virtual ~FlowVisualModel();

    void init();
    void reinit();
    void initVisual();
    void draw(const core::visual::VisualParams*);
    void drawTransparent(const core::visual::VisualParams*);
    void drawTetra();
    void computeStreamLine(unsigned int index, unsigned int maxNbPoints, double dt);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }
    static std::string templateName(const FlowVisualModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

};
/*
#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_FLUIDSOLIDINTERACTIONFORCEFIELD_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class FlowVisualModel<defaulttype::Vec3dTypes>;
extern template class FlowVisualModel<defaulttype::Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class FlowVisualModel<defaulttype::Vec3fTypes>;
extern template class FlowVisualModel<defaulttype::Vec2fTypes>;
#endif
#endif
*/
}

}

}

#endif /* FLOWVISUALMODEL_H_ */
