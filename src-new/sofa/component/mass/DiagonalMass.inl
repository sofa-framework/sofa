#ifndef SOFA_COMPONENT_MASS_DIAGONALMASS_INL
#define SOFA_COMPONENT_MASS_DIAGONALMASS_INL

#include <sofa/component/mass/DiagonalMass.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/TopologyChangedEvent.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;


template<class MassType>
void MassPointCreationFunction(int ,
        void* , MassType & t,
        const std::vector< unsigned int > &,
        const std::vector< double >&)
{
    t=0;
}

template< class DataTypes, class MassType>
inline void MassEdgeCreationFunction(const std::vector<unsigned int> &edgeAdded,
        void* param, helper::vector<MassType> &masses)
{
    DiagonalMass<DataTypes, MassType> *dm= (DiagonalMass<DataTypes, MassType> *)param;
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_EDGESET)
    {
        component::topology::EdgeSetTopology<DataTypes> *est = dynamic_cast<component::topology::EdgeSetTopology<DataTypes>*>(dm->getContext()->getMainTopology());
        assert(est!=0);
        component::topology::EdgeSetTopologyContainer *container=est->getEdgeSetTopologyContainer();
        const std::vector<component::topology::Edge> &edgeArray=container->getEdgeArray();
        component::topology::EdgeSetGeometryAlgorithms<DataTypes> *ga=est->getEdgeSetGeometryAlgorithms();
        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass;
        unsigned int i;

        for (i=0; i<edgeAdded.size(); ++i)
        {
            /// get the edge to be added
            const component::topology::Edge &e=edgeArray[edgeAdded[i]];
            // compute its mass based on the mass density and the edge length
            mass=(md*ga->computeRestEdgeLength(edgeAdded[i]))/2;
            // added mass on its two vertices
            masses[e.first]+=mass;
            masses[e.second]+=mass;
        }

    }
}

template <>
inline void MassEdgeCreationFunction<RigidTypes, RigidMass>(const std::vector<unsigned int> &,
        void* , helper::vector<RigidMass> &)
{
}
template< class DataTypes, class MassType>
inline void MassEdgeDestroyFunction(const std::vector<unsigned int> &edgeRemoved,
        void* param, helper::vector<MassType> &masses)
{
    DiagonalMass<DataTypes, MassType> *dm= (DiagonalMass<DataTypes, MassType> *)param;
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_EDGESET)
    {
        component::topology::EdgeSetTopology<DataTypes> *est = dynamic_cast<component::topology::EdgeSetTopology<DataTypes>*>(dm->getContext()->getMainTopology());
        assert(est!=0);
        component::topology::EdgeSetTopologyContainer *container=est->getEdgeSetTopologyContainer();
        const std::vector<component::topology::Edge> &edgeArray=container->getEdgeArray();
        component::topology::EdgeSetGeometryAlgorithms<DataTypes> *ga=est->getEdgeSetGeometryAlgorithms();
        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass;
        unsigned int i;

        for (i=0; i<edgeRemoved.size(); ++i)
        {
            /// get the edge to be added
            const component::topology::Edge &e=edgeArray[edgeRemoved[i]];
            // compute its mass based on the mass density and the edge length
            mass=(md*ga->computeRestEdgeLength(edgeRemoved[i]))/2;
            // added mass on its two vertices
            masses[e.first]-=mass;
            masses[e.second]-=mass;
        }

    }
}

template <>
inline void MassEdgeDestroyFunction<RigidTypes, RigidMass>(const std::vector<unsigned int> &,
        void* , helper::vector<RigidMass> &)
{
}

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::DiagonalMass()
    : f_mass( dataField(&f_mass, "mass", "values of the particles masses") )
    ,m_massDensity( dataField(&m_massDensity, (Real)1.0,"massDensity", "mass density that allows to compute the  particles masses from a mesh topology and geometry") )
    , topologyType(TOPOLOGY_UNKNOWN)
{

}


template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::DiagonalMass(core::componentmodel::behavior::MechanicalState<DataTypes>* mstate, const std::string& /*name*/)
    : core::componentmodel::behavior::Mass<DataTypes>(mstate)
    , f_mass( dataField(&f_mass, "mass", "values of the particles' masses") )
    , m_massDensity( dataField(&m_massDensity, (Real)1.0,"massDensity", "mass density that allows to compute the  particles masses from a mesh topology and geometry") )
    , topologyType(TOPOLOGY_UNKNOWN)
{
}

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::~DiagonalMass()
{
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::clear()
{
    VecMass& masses = *f_mass.beginEdit();
    masses.clear();
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMass(const MassType& m)
{
    VecMass& masses = *f_mass.beginEdit();
    masses.push_back(m);
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::resize(int vsize)
{
    VecMass& masses = *f_mass.beginEdit();
    masses.resize(vsize);
    f_mass.endEdit();
}

// -- Mass interface
template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMDx(VecDeriv& res, const VecDeriv& dx)
{
    const MassVector &masses= f_mass.getValue().getArray();
    for (unsigned int i=0; i<dx.size(); i++)
    {
        res[i] += dx[i] * masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::accFromF(VecDeriv& a, const VecDeriv& f)
{
    const MassVector &masses= f_mass.getValue().getArray();
    for (unsigned int i=0; i<f.size(); i++)
    {
        a[i] = f[i] / masses[i];
    }
}

template <class DataTypes, class MassType>
double DiagonalMass<DataTypes, MassType>::getKineticEnergy( const VecDeriv& v )
{
    const MassVector &masses= f_mass.getValue().getArray();
    double e = 0;
    for (unsigned int i=0; i<masses.size(); i++)
    {
        e += v[i]*masses[i]*v[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }
    return e/2;
}

template <class DataTypes, class MassType>
double DiagonalMass<DataTypes, MassType>::getPotentialEnergy( const VecCoord& x )
{
    const MassVector &masses= f_mass.getValue().getArray();
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<masses.size(); i++)
    {
        e -= theGravity*masses[i]*x[i];
    }
    return e;
}
template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::handleEvent( Event *event )
{
    component::topology::TopologyChangedEvent *tce=dynamic_cast<component::topology::TopologyChangedEvent *>(event);
    /// test that the event is a change of topology and that it
    if ((tce) && (tce->getTopology()== getContext()->getMainTopology()))
    {
        core::componentmodel::topology::BaseTopology *topology = static_cast<core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

        std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=topology->firstChange();
        std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=topology->lastChange();

        VecMass& masses = *f_mass.beginEdit();
        masses.handleTopologyEvents(itBegin,itEnd);
        f_mass.endEdit();


    }

}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::init()
{
    ForceField<DataTypes>::init();
    // add the functions to handle topology changes.

    VecMass& masses = *f_mass.beginEdit();
    masses.setCreateFunction(MassPointCreationFunction<MassType>);
    masses.setCreateEdgeFunction(MassEdgeCreationFunction<DataTypes,MassType>);
    masses.setDestroyEdgeFunction(MassEdgeDestroyFunction<DataTypes,MassType>);
    masses.setCreateParameter( (void *) this );
    masses.setDestroyParameter( (void *) this );
    f_mass.endEdit();
    /// handle events
    f_listening.setValue(true);

    if ((f_mass.getValue().size()==0) && (getContext()->getMainTopology()!=0))
    {
        /// check that the topology is of type EdgeSet
        /// \todo handle other types of topology
        component::topology::EdgeSetTopology<DataTypes> *est = dynamic_cast<component::topology::EdgeSetTopology<DataTypes>*>(getContext()->getMainTopology());
        assert(est!=0);
        VecMass& masses = *f_mass.beginEdit();
        topologyType=TOPOLOGY_EDGESET;

        component::topology::EdgeSetTopologyContainer *container=est->getEdgeSetTopologyContainer();
        component::topology::EdgeSetGeometryAlgorithms<DataTypes> *ga=est->getEdgeSetGeometryAlgorithms();

        const std::vector<component::topology::Edge> &ea=container->getEdgeArray();
        // resize array
        clear();
        masses.resize(est->getDOFNumber());
        unsigned int i;
        for(i=0; i<masses.size(); ++i)
            masses[i]=(Real)0;

        Real md=m_massDensity.getValue();
        Real mass;

        for (i=0; i<ea.size(); ++i)
        {
            const component::topology::Edge &e=ea[i];
            mass=(md*ga->computeEdgeLength(i))/2;
            masses[e.first]+=mass;
            masses[e.second]+=mass;
        }
        f_mass.endEdit();

    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    const MassVector &masses= f_mass.getValue().getArray();

    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = getContext()->getPositionInWorld() / vframe;
    aframe = getContext()->getPositionInWorld().backProjectVector( aframe );

    // add weight and inertia force
    for (unsigned int i=0; i<masses.size(); i++)
    {
        f[i] += theGravity*masses[i] + core::componentmodel::behavior::inertiaForce(vframe,aframe,masses[i],x[i],v[i]);
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
    const VecCoord& x = *this->mstate->getX();
    glDisable (GL_LIGHTING);
    glPointSize(2);
    glColor4f (1,1,1,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::glVertexT(x[i]);
    }
    glEnd();
}

template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *this->mstate->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        //const Coord& p = x[i];
        double p[3] = {0.0, 0.0, 0.0};
        DataTypes::get(p[0],p[1],p[2],x[i]);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}

template <class DataTypes, class MassType>
class DiagonalMass<DataTypes, MassType>::Loader : public MassSpringLoader
{
public:
    DiagonalMass<DataTypes, MassType>* dest;
    Loader(DiagonalMass<DataTypes, MassType>* dest) : dest(dest) {}
    virtual void addMass(double /*px*/, double /*py*/, double /*pz*/, double /*vx*/, double /*vy*/, double /*vz*/, double mass, double /*elastic*/, bool /*fixed*/, bool /*surface*/)
    {
        dest->addMass(MassType(mass));
    }
};

template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::load(const char *filename)
{
    clear();
    if (filename!=NULL && filename[0]!='\0')
    {
        Loader loader(this);
        return loader.load(filename);
    }
    else return false;
}

// Specialization for rigids
template <>
double DiagonalMass<RigidTypes, RigidMass>::getPotentialEnergy( const VecCoord& x );

template <>
void DiagonalMass<RigidTypes, RigidMass>::draw();

template <>
void DiagonalMass<RigidTypes, RigidMass>::init();

template <>
void DiagonalMass<RigidTypes, RigidMass>::handleEvent( Event *event );


} // namespace mass

} // namespace component

} // namespace sofa

#endif
