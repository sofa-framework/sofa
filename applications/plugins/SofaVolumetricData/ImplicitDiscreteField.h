#ifndef DISTANCE_GRID_COMPONENT_H
#define DISTANCE_GRID_COMPONENT_H

#include <iostream>
#include <SofaVolumetricData/ImplicitShape.h>
#include <SofaVolumetricData/DistanceGrid.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace core
{

typedef sofa::component::container::DistanceGrid DistanceGrid;
typedef sofa::defaulttype::Vector3 Coord;
using namespace sofa::core::objectmodel;

class DiscreteField : public ImplicitShape {

public:
    SOFA_CLASS(DiscreteField, BaseObject);
    DiscreteField()
        : in_filename(initData(&in_filename,"filename","filename"))
        , in_nx(initData(&in_nx,0,"nx","in_nx"))
        , in_ny(initData(&in_ny,0,"ny","in_ny"))
        , in_nz(initData(&in_nz,0,"nz","in_nz"))
        , in_scale(initData(&in_scale,0.0,"scale","in_scale"))
        , in_sampling(initData(&in_sampling,0.0,"sampling","in_sampling"))
    {
    }
    virtual ~DiscreteField()  { }
    DistanceGrid* grid {nullptr};
    void setFilename(const std::string& name);
    void loadGrid(double scale, double sampling, int nx, int ny, int nz, Coord pmin, Coord pmax);
    virtual void init();
    virtual double eval(Coord p);

private:
    DataFileName in_filename;
    Coord pmin, pmax;
    Data<int> in_nx; ///< in_nx
    Data<int> in_ny; ///< in_ny
    Data<int> in_nz; ///< in_nz
    Data<double> in_scale; ///< in_scale
    Data<double> in_sampling; ///< in_sampling

};


}

}

#endif
