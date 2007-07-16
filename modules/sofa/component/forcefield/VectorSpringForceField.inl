#ifndef SOFA_COMPONENT_FORCEFIELD_VECTORSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_VECTORSPRINGFORCEFIELD_INL

#include "VectorSpringForceField.h"
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/component/topology/EdgeData.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <iostream>
#include <GL/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void VectorSpringForceField<DataTypes>::springCreationFunction(int /*index*/,
        void* param, Spring& t,
        const topology::Edge& e,
        const std::vector< unsigned int > &ancestors,
        const std::vector< double >& coefs)
{
    VectorSpringForceField<DataTypes> *ff= static_cast<VectorSpringForceField<DataTypes> *>(param);
    if (ff)
    {
        topology::EdgeSetTopology<DataTypes>* topology = dynamic_cast<topology::EdgeSetTopology<DataTypes>*>(ff->getContext()->getMainTopology());
        if (topology)
        {
            //EdgeSetGeometryAlgorithms<DataTypes> *ga=topology->getEdgeSetGeometryAlgorithms();
            //t.restLength=ga->computeRestEdgeLength(index);
            const typename DataTypes::VecCoord& x0 = *ff->getObject1()->getX0();
            t.restVector = x0[e.second] - x0[e.first];
            if (ancestors.size()>0)
            {
                t.kd=t.ks=0;
                const topology::EdgeData<Spring> &sa=ff->getSpringArray();
                unsigned int i;
                for (i=0; i<ancestors.size(); ++i)
                {
                    t.kd+=(typename DataTypes::Real)(sa[i].kd*coefs[i]);
                    t.ks+=(typename DataTypes::Real)(sa[i].ks*coefs[i]);
                }
            }
            else
            {
                t.kd=ff->getStiffness();
                t.ks=ff->getViscosity();
            }
        }
    }
}

template <class DataTypes>
class VectorSpringForceField<DataTypes>::Loader : public sofa::helper::io::MassSpringLoader
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    VectorSpringForceField<DataTypes>* dest;
    Loader(VectorSpringForceField<DataTypes>* dest) : dest(dest) {}
    virtual void addVectorSpring(int m1, int m2, double ks, double kd, double /*initpos*/, double restx, double resty, double restz)
    {
        dest->addSpring(m1,m2,ks,kd,Coord((Real)restx,(Real)resty,(Real)restz));
    }
    virtual void setNumSprings(int /*n*/)
    {
        //dest->resizeArray((unsigned int )n);
    }

};

template <class DataTypes>
bool VectorSpringForceField<DataTypes>::load(const char *filename)
{
    if (filename && filename[0])
    {
        Loader loader(this);
        return loader.load(filename);
    }
    else return false;
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::resizeArray(unsigned int n)
{
    springArray.resize(n);
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::addSpring(int m1, int m2, double ks, double kd, Coord restVector)
{
    if (useTopology && topology)
    {
        topology::EdgeSetTopologyContainer *container=topology->getEdgeSetTopologyContainer();

        int e=container->getEdgeIndex((unsigned int)m1,(unsigned int)m2);
        if (e>=0)
            springArray[e]=Spring((Real)ks,(Real)kd,restVector);
    }
    else
    {
        springArray.push_back(Spring((Real)ks, (Real)kd, restVector));
        edgeArray.push_back(topology::Edge(m1,m2));
    }
}

template <class DataTypes>
VectorSpringForceField<DataTypes>::VectorSpringForceField(MechanicalState* _object)
    : object1(_object), object2(_object)
    , m_potentialEnergy( 0.0 ), useTopology( false ), topology ( NULL )
    , m_filename( dataField(&m_filename,std::string(""),"filename","File name from which the spring informations are loaded") )
    , m_stiffness( dataField(&m_stiffness,1.0,"stiffness","Default edge stiffness used in absence of file information") )
    , m_viscosity( dataField(&m_viscosity,1.0,"viscosity","Default edge viscosity used in absence of file information") )
{
    springArray.setCreateFunction(springCreationFunction);
    springArray.setCreateParameter( (void *) this );
}

template <class DataTypes>
VectorSpringForceField<DataTypes>::VectorSpringForceField(MechanicalState* _object1, MechanicalState* _object2)
    : object1(_object1), object2(_object2)
    , m_potentialEnergy( 0.0 ), useTopology( false ), topology ( NULL )
    , m_filename( dataField(&m_filename,std::string(""),"filename","File name from which the spring informations are loaded") )
    , m_stiffness( dataField(&m_stiffness,1.0,"stiffness","Default edge stiffness used in absence of file information") )
    , m_viscosity( dataField(&m_viscosity,1.0,"viscosity","Default edge viscosity used in absence of file information") )
{
    springArray.setCreateFunction(springCreationFunction);
    springArray.setCreateParameter( (void *) this );
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::init()
{
    this->InteractionForceField::init();
    if( object1==NULL )
    {
        sofa::core::objectmodel::BaseObject* mstate = getContext()->getMechanicalState();
        assert(mstate!=NULL);
        MechanicalState* state = dynamic_cast<MechanicalState*>(mstate );
        assert( state!= NULL );
        object1 = object2 = state;
        topology = dynamic_cast<topology::EdgeSetTopology<DataTypes>*>(getContext()->getMainTopology());
    }

    if (!m_filename.getValue().empty())
    {
        // load the springs from a file
        load(( const char *)(m_filename.getValue().c_str()));
    }
    else if (topology)
    {
        // create springs based on the mesh topology
        useTopology = true;
        createDefaultSprings();
        f_listening.setValue(true);
    }
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::createDefaultSprings()
{
    topology::EdgeSetTopologyContainer *container=topology->getEdgeSetTopologyContainer();
    const std::vector<topology::Edge> &ea=container->getEdgeArray();
    std::cout << "Creating "<<ea.size()<<" Vector Springs from EdgeSetTopology"<<std::endl;
    springArray.resize(ea.size());
    //EdgeSetGeometryAlgorithms<DataTypes> *ga=topology->getEdgeSetGeometryAlgorithms();
    //EdgeLengthArrayInterface<Real,DataTypes> elai(springArray);
    //ga->computeEdgeLength(elai);
    const VecCoord& x0 = *this->object1->getX0();
    unsigned int i;
    for (i=0; i<ea.size(); ++i)
    {
        springArray[i].ks=(Real)m_stiffness.getValue();
        springArray[i].kd=(Real)m_viscosity.getValue();
        springArray[i].restVector = x0[ea[i].second]-x0[ea[i].first];
    }

}
template<class DataTypes>
void VectorSpringForceField<DataTypes>::handleEvent( Event* e )
{
    if (useTopology)
    {
        if( sofa::core::objectmodel::KeypressedEvent* ke = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>( e ) )
        {
            /// handle ctrl+d key
            if (ke->getKey()=='D')
            {
                if (topology->getEdgeSetTopologyContainer()->getNumberOfEdges()>12)
                {
                    topology::EdgeSetTopologyAlgorithms<DataTypes> *esta=topology->getEdgeSetTopologyAlgorithms();
                    std::vector<unsigned int> edgeArray;
                    edgeArray.push_back(12);
                    esta->removeEdges(edgeArray);
                }
                //            esta->splitEdges(edgeArray);
            }
        }
        else
        {
            sofa::component::topology::TopologyChangedEvent *tce=dynamic_cast<sofa::component::topology::TopologyChangedEvent *>(e);
            /// test that the event is a change of topology and that it
            if ((tce) && (tce->getTopology()== topology))
            {
                std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=topology->firstChange();
                std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=topology->lastChange();
                /// Topological events are handled by the EdgeData structure
                springArray.handleTopologyEvents(itBegin,itEnd);
            }
        }
    }
}

template<class DataTypes>
//void VectorSpringForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& p, const VecDeriv& v)
void VectorSpringForceField<DataTypes>::addForce()
{
    //assert(this->mstate);
    m_potentialEnergy = 0;
    const std::vector<topology::Edge> &ea=(useTopology)?topology->getEdgeSetTopologyContainer()->getEdgeArray() : edgeArray;
    Coord u;

    VecDeriv& f1 = *object1->getF();
    const VecCoord& p1 = *object1->getX();
    const VecDeriv& v1 = *object1->getV();

    VecDeriv& f2 = *object2->getF();
    const VecCoord& p2 = *object2->getX();
    const VecDeriv& v2 = *object2->getV();

    f1.resize(p1.size());
    f2.resize(p2.size());

    Deriv relativeVelocity,force;
    for (unsigned int i=0; i<ea.size(); i++)
    {
        const topology::Edge &e=ea[i];
        const Spring &s=springArray[i];
        // paul---------------------------------------------------------------
        Deriv current_direction = p2[e.second]-p1[e.first];
        Deriv squash_vector = current_direction - s.restVector;
        Deriv relativeVelocity = v2[e.second]-v1[e.first];
        force = (squash_vector * s.ks) + (relativeVelocity * s.kd);

        f1[e.first]+=force;
        f2[e.second]-=force;
    }
}

template<class DataTypes>
//void VectorSpringForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
void VectorSpringForceField<DataTypes>::addDForce()
{
    const std::vector<topology::Edge> &ea=(useTopology)?topology->getEdgeSetTopologyContainer()->getEdgeArray() : edgeArray;
    Deriv dforce,d;
    unsigned int i;

    VecDeriv& df1 = *object1->getF();
    const VecCoord& dx1 = *object1->getDx();

    VecDeriv& df2 = *object2->getF();
    const VecCoord& dx2 = *object2->getDx();

    df1.resize(dx1.size());
    df2.resize(dx2.size());

    for ( i=0; i<ea.size(); i++)
    {
        const topology::Edge &e=ea[i];
        const Spring &s=springArray[i];
        d = dx2[e.second]-dx1[e.first];
        dforce = d*s.ks;
        df1[e.first]+=dforce;
        df2[e.second]-=dforce;
    }

}

template<class DataTypes>
void VectorSpringForceField<DataTypes>::draw()
{
    if (!((this->object1 == this->object2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields()))
        return;
    //const VecCoord& p = *this->mstate->getX();
    const VecCoord& p1 = *this->object1->getX();
    const VecCoord& p2 = *this->object2->getX();
    const std::vector<topology::Edge> &ea=(useTopology)?topology->getEdgeSetTopologyContainer()->getEdgeArray() : edgeArray;

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<springArray.size(); i++)
    {
        const topology::Edge &e=ea[i];
        //const Spring &s=springArray[i];

        glColor4f(0,1,1,0.5f);

        glVertex3d(p1[e.first][0],p1[e.first][1],p1[e.first][2]);
        glVertex3d(p2[e.second][0],p2[e.second][1],p2[e.second][2]);
    }
    glEnd();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
