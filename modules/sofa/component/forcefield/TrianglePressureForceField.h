#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_H


#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/topology/TriangleSubsetData.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology;

template<class DataTypes>
class TrianglePressureForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

protected:

    class TrianglePressureInformation
    {
    public:
        Real area;
        Deriv force;

        TrianglePressureInformation() {}
        TrianglePressureInformation(const TrianglePressureInformation &e)
            : area(e.area),force(e.force)
        { }
    };

    TriangleSubsetData<TrianglePressureInformation> trianglePressureMap;

    topology::TriangleSetTopology<DataTypes>* tst;

    Data<Deriv> pressure;

    Data<std::string> triangleList;

    /// the normal used to define the edge subjected to the pressure force.
    Data<Deriv> normal;

    Data<Real> dmin; // coordinates min of the plane for the vertex selection
    Data<Real> dmax;// coordinates max of the plane for the vertex selection

public:

    TrianglePressureForceField():
        tst(0)
        , pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
        , triangleList(initData(&triangleList,std::string(),"triangleList", "Indices of triangles separated with commas where a pressure is applied"))
        , normal(initData(&normal,"normal", "Normal direction for the plane selection of triangles"))
        , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
        , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
    {
    }

    virtual ~TrianglePressureForceField();

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);
    virtual void addDForce (VecDeriv& /*df*/, const VecDeriv& /*dx*/) {}
    virtual double getPotentialEnergy(const VecCoord& x);

    // Handle topological changes
    virtual void handleTopologyChange();


    // -- VisualModel interface
    void draw();
    void initTextures() { };
    void update() { };


    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateTriangleInformation(); }

protected :
    void selectTrianglesAlongPlane();
    void selectTrianglesFromString();
    void updateTriangleInformation();
    void initTriangleInformation();
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

#endif /* _TRIANGLEPRESSUREFORCEFIELD_H_ */
