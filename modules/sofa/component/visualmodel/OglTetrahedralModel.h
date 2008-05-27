#ifndef OGLTETRAHEDRALMODEL_H_
#define OGLTETRAHEDRALMODEL_H_

#include <sofa/core/VisualModel.h>
#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

template<class DataTypes>
class OglTetrahedralModel : public core::VisualModel
{
public:
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

private:
    topology::TetrahedronSetTopology<DataTypes>* topo;
    component::MechanicalObject<DataTypes>* nodes;

    Data<bool> depthTest;
    Data<bool> blending;

public:
    OglTetrahedralModel();
    virtual ~OglTetrahedralModel();

    void init();
    void drawVisual();
    bool addBBox(double* minBBox, double* maxBBox);

};

#endif /*OGLTETRAHEDRALMODEL_H_*/
}
}
}
