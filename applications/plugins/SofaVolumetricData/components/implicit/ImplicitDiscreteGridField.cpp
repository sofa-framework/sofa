#include "DistanceGridComponent.h"

namespace sofa
{

namespace core
{

typedef sofa::component::container::DistanceGrid DistanceGrid;

//factory register
int DistanceGridComponentClass = sofa::core::RegisterObject("Use to store grid").add< DiscreteField >().addAlias("DistGrid");

//fil the grid
void DiscreteField::loadGrid(double scale, double sampling, int nx, int ny, int nz, Coord pmin, Coord pmax) {

    grid = grid->load(in_filename.getFullPath(), scale, sampling, nx, ny, nz, pmin, pmax);
    if(grid == nullptr) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "uninitialized grid";
    }
    return;

}


//used to set a name in tests
void DiscreteField::setFilename(const std::string& name) {
    this->in_filename.setValue(name);
}

//eval the grid
double DiscreteField::eval(Coord p) {

    return grid->quickeval(p);

}


void DiscreteField::init() {

    if(in_nx.getValue()==0 && in_nz.getValue()==0 && in_nz.getValue()==0) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "uninitialized grid";
    }
    else if(in_filename.isSet() == false) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "unset filename";
    }
    else {
        pmin.set(0,0,-5.0);
        pmax.set(27,27,5.0);
        loadGrid(in_scale.getValue(),in_sampling.getValue(),in_nx.getValue(),in_ny.getValue(),in_nz.getValue(),pmin,pmax);
    }

}


} //namespace core

} //namespace sofa
