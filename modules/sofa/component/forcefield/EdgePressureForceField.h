#ifndef SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H


#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/topology/EdgeSubsetData.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology;

template<class DataTypes>
class EdgePressureForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

protected:

    class EdgePressureInformation
    {
    public:
        Real length;
        Deriv force;

        EdgePressureInformation() {}
        EdgePressureInformation(const EdgePressureInformation &e)
            : length(e.length),force(e.force)
        { }
    };

    EdgeSubsetData<EdgePressureInformation> edgePressureMap;


    topology::EdgeSetTopology<DataTypes>* est;

    Data<Deriv> pressure;

    Data<std::string> edgeList;

    /// the normal used to define the edge subjected to the pressure force.
    Data<Deriv> normal;

    Data<Real> dmin; // coordinates min of the plane for the vertex selection
    Data<Real> dmax;// coordinates max of the plane for the vertex selection

public:

    EdgePressureForceField():
        est(0)
        , pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
        , edgeList(initData(&edgeList,std::string(0),"edgeList", "Indices of edges separated with commas where a pressure is applied"))
        , normal(initData(&normal,"normal", "Normal direction for the plane selection of edges"))
        , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
        , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
    {
    }

    virtual ~EdgePressureForceField();

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);
    virtual void addDForce (VecDeriv& /*df*/, const VecDeriv& /*dx*/) {}
    virtual double getPotentialEnergy(const VecCoord& x);

    // Handle topological changes
    virtual void handleTopologyChange();

    void draw();

    void setDminAndDmax(const double _dmin, const double _dmax)
    {
        dmin.setValue((Real)_dmin); dmax.setValue((Real)_dmax);
    }
    void setNormal(const Coord n) { normal.setValue(n);}
    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateEdgeInformation(); }

protected :
    void selectEdgesAlongPlane();
    void selectEdgesFromString();
    void updateEdgeInformation();
    void initEdgeInformation();
    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,normal.getValue());
        if ((d>dmin.getValue())&& (d<dmax.getValue()))
            return true;
        else
            return false;
    }
};


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif /* _EDGEPRESSUREFORCEFIELD_H_ */
