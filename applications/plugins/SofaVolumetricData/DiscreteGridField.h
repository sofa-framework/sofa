#ifndef DISTANCE_GRID_COMPONENT_H
#define DISTANCE_GRID_COMPONENT_H

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/ObjectFactory.h>
#include "ScalarField.h"
#include "../../DistanceGrid.h"

namespace sofa
{

namespace component
{

namespace implicit
{

using sofa::core::objectmodel::DataFileName ;
using sofa::component::container::DistanceGrid ;
using sofa::component::implicit::Vector3 ;

class DiscreteGridField : public ScalarField {
public:
    SOFA_CLASS(DiscreteGridField, ScalarField);

    DiscreteGridField();
    virtual ~DiscreteGridField()  { }
    DistanceGrid* grid {nullptr};
    void setFilename(const std::string& name);
    void loadGrid(double scale, double sampling, int nx, int ny, int nz, Vector3 pmin, Vector3 pmax);
    virtual void init();
    virtual double eval(Vector3 p);

private:
    DataFileName in_filename;
    Vector3 pmin, pmax;
    Data<int> in_nx;
    Data<int> in_ny;
    Data<int> in_nz;
    Data<double> in_scale;
    Data<double> in_sampling;

};

} /// implicit
    using implicit::DiscreteGridField ;

} /// component

} /// sofa

#endif
