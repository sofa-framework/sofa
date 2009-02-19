/*
 * FlowVisualModel.h
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#ifndef FLOWVISUALMODEL_H_
#define FLOWVISUALMODEL_H_

#include <sofa/component/component.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/ManifoldTriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/helper/gl/BasicShapes.h>
namespace sofa
{

namespace component
{

namespace visualmodel
{

static unsigned int COLORMAP_SIZE=64;

static defaulttype::Vec3d ColorMap[64] =
{
    defaulttype::Vec3d( 0.0,        0.0,       0.5625 ),
    defaulttype::Vec3d( 0.0,        0.0,       0.625  ),
    defaulttype::Vec3d( 0.0,        0.0,       0.6875 ),
    defaulttype::Vec3d( 0.0,        0.0,         0.75 ),
    defaulttype::Vec3d( 0.0,        0.0,       0.8125 ),
    defaulttype::Vec3d( 0.0,        0.0,        0.875 ),
    defaulttype::Vec3d( 0.0,        0.0,       0.9375 ),
    defaulttype::Vec3d( 0.0,        0.0,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.0625,          1.0 ),
    defaulttype::Vec3d( 0.0,      0.125,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.1875,          1.0 ),
    defaulttype::Vec3d( 0.0,       0.25,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.3125,          1.0 ),
    defaulttype::Vec3d( 0.0,      0.375,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.4375,          1.0 ),
    defaulttype::Vec3d( 0.0,        0.5,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.5625,          1.0 ),
    defaulttype::Vec3d( 0.0,      0.625,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.6875,          1.0 ),
    defaulttype::Vec3d( 0.0,       0.75,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.8125,          1.0 ),
    defaulttype::Vec3d( 0.0,     0.875,           1.0 ),
    defaulttype::Vec3d( 0.0,     0.9375,          1.0 ),
    defaulttype::Vec3d( 0.0,        1.0,          1.0 ),
    defaulttype::Vec3d( 0.0625,     1.0,          1.0 ),
    defaulttype::Vec3d( 0.125,      1.0,       0.9375 ),
    defaulttype::Vec3d( 0.1875,     1.0,        0.875 ),
    defaulttype::Vec3d( 0.25,       1.0,       0.8125 ),
    defaulttype::Vec3d( 0.3125,     1.0,         0.75 ),
    defaulttype::Vec3d( 0.375,      1.0,       0.6875 ),
    defaulttype::Vec3d( 0.4375,     1.0,        0.625 ),
    defaulttype::Vec3d( 0.5,        1.0,       0.5625 ),
    defaulttype::Vec3d( 0.5625,     1.0,          0.5 ),
    defaulttype::Vec3d( 0.625,      1.0,       0.4375 ),
    defaulttype::Vec3d( 0.6875,     1.0,        0.375 ),
    defaulttype::Vec3d( 0.75,       1.0,       0.3125 ),
    defaulttype::Vec3d( 0.8125,     1.0,         0.25 ),
    defaulttype::Vec3d( 0.875,      1.0,       0.1875 ),
    defaulttype::Vec3d( 0.9375,     1.0,        0.125 ),
    defaulttype::Vec3d( 1.0,        1.0,       0.0625 ),
    defaulttype::Vec3d( 1.0,        1.0,          0.0 ),
    defaulttype::Vec3d( 1.0,       0.9375,        0.0 ),
    defaulttype::Vec3d( 1.0,        0.875,        0.0 ),
    defaulttype::Vec3d( 1.0,       0.8125,        0.0 ),
    defaulttype::Vec3d( 1.0,         0.75,        0.0 ),
    defaulttype::Vec3d( 1.0,       0.6875,        0.0 ),
    defaulttype::Vec3d( 1.0,        0.625,        0.0 ),
    defaulttype::Vec3d( 1.0,       0.5625,        0.0 ),
    defaulttype::Vec3d( 1.0,          0.5,        0.0 ),
    defaulttype::Vec3d( 1.0,       0.4375,        0.0 ),
    defaulttype::Vec3d( 1.0,        0.375,        0.0 ),
    defaulttype::Vec3d( 1.0,       0.3125,        0.0 ),
    defaulttype::Vec3d( 1.0,         0.25,        0.0 ),
    defaulttype::Vec3d( 1.0,       0.1875,        0.0 ),
    defaulttype::Vec3d( 1.0,        0.125,        0.0 ),
    defaulttype::Vec3d( 1.0,       0.0625,        0.0 ),
    defaulttype::Vec3d( 1.0,          0.0,        0.0 ),
    defaulttype::Vec3d( 0.9375,       0.0,        0.0 ),
    defaulttype::Vec3d( 0.875,        0.0,        0.0 ),
    defaulttype::Vec3d( 0.8125,       0.0,        0.0 ),
    defaulttype::Vec3d( 0.75,         0.0,        0.0 ),
    defaulttype::Vec3d( 0.6875,       0.0,        0.0 ),
    defaulttype::Vec3d( 0.625,        0.0,        0.0 ),
    defaulttype::Vec3d( 0.5625,       0.0,        0.0 )
};

template <class DataTypes>
class SOFA_COMPONENT_VISUALMODEL_API FlowVisualModel : public core::VisualModel
{
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> FluidState;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
protected:
    FluidState* fstate;
    topology::ManifoldTriangleSetTopologyContainer* m_triTopo;
    topology::TriangleSetGeometryAlgorithms<DataTypes>* m_triGeo;

public:
    Data<double> viewVelocityFactor;
    FlowVisualModel();
    virtual ~FlowVisualModel();

    void init();
    void initVisual();
    void draw();
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
