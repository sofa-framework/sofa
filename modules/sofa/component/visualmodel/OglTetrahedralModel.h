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

class OglTetrahedralModel : public core::VisualModel
{
private:
    topology::TetrahedronSetTopology<defaulttype::Vec3fTypes>* topo;
    component::MechanicalObject<defaulttype::Vec3fTypes>* nodes;

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
