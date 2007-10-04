#ifndef SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H


#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


template<class DataTypes>
class EdgePressureForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef topology::MeshTopology::index_type Index;
    typedef topology::MeshTopology::Triangle Element;
    typedef topology::MeshTopology::SeqTriangles VecElement;


protected:

    class EdgePressureInformation
    {
    public:
        Index index[2];
        Real length;
        Deriv force;

        EdgePressureInformation()
        {
        }
        EdgePressureInformation(const EdgePressureInformation &e)
            : length(e.length),force(e.force)
        {
            index[0]=e.index[0];
            index[1]=e.index[1];
        }
    };

    std::vector<EdgePressureInformation> edgeInfo;

    unsigned int nbEdges;  // number of edge pressure forces

    topology::MeshTopology* _mesh;
    const VecElement *_indexedElements;
    DataField< VecCoord > _initialPoints;										///< the intial positions of the points

    Deriv pressure;

    bool usePlaneSelection; // whether the edges are defined from 2 parallel planes or not
    /// the normal used to define the edge subjected to the pressure force.
    Deriv normal;

    Real dmin; // coordinates min of the plane for the vertex selection
    Real dmax;// coordinates max of the plane for the vertex selection

public:

    EdgePressureForceField()
        : nbEdges(0)
        , _mesh(NULL)
        , _initialPoints(dataField(&_initialPoints, "initialPoints", "Initial Position"))
        , usePlaneSelection(false)
    {
    }

    virtual ~EdgePressureForceField();

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);
    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);
    virtual void addDForce (VecDeriv& /*df*/, const VecDeriv& /*dx*/) {}
    virtual double getPotentialEnergy(const VecCoord& x);


    // -- VisualModel interface
    void draw();
    void initTextures() { };
    void update() { };
    void addEdgePressure(Index ind1,Index ind2);
    void setNormal (Coord dir);
    void selectEdgesAlongPlane();
    void setDminAndDmax(const Real _dmin,const Real _dmax) {dmin=_dmin; dmax=_dmax; usePlaneSelection=true;}

    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateEdgeInformation(); }

protected :
    void updateEdgeInformation();
    void initEdgeInformation();
    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,normal);
        if ((d>dmin)&& (d<dmax))
            return true;
        else
            return false;
    }
};


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif /* _EDGEPRESSUREFORCEFIELD_H_ */
