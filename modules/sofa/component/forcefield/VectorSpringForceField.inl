#ifndef SOFA_COMPONENT_FORCEFIELD_VECTORSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_VECTORSPRINGFORCEFIELD_INL

#include "VectorSpringForceField.h"
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/component/topology/EdgeData.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/gl.h>
#include <assert.h>
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
        const sofa::helper::vector< unsigned int > &ancestors,
        const sofa::helper::vector< double >& coefs)
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
            t.restVector = x0[e[1]] - x0[e[0]];
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
    virtual void addVectorSpring(int m1, int m2, SReal ks, SReal kd, SReal /*initpos*/, SReal restx, SReal resty, SReal restz)
    {
        dest->addSpring(m1,m2,ks,kd,Coord(restx,resty,restz));
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
void VectorSpringForceField<DataTypes>::addSpring(int m1, int m2, SReal ks, SReal kd, Coord restVector)
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
    : Inherit(_object, _object)
    , m_potentialEnergy( 0.0 ), useTopology( false ), topology ( NULL )
    , m_filename( initData(&m_filename,std::string(""),"filename","File name from which the spring informations are loaded") )
    , m_stiffness( initData(&m_stiffness,1.0,"stiffness","Default edge stiffness used in absence of file information") )
    , m_viscosity( initData(&m_viscosity,1.0,"viscosity","Default edge viscosity used in absence of file information") )
{
    springArray.setCreateFunction(springCreationFunction);
    springArray.setCreateParameter( (void *) this );
}

template <class DataTypes>
VectorSpringForceField<DataTypes>::VectorSpringForceField(MechanicalState* _object1, MechanicalState* _object2)
    : Inherit(_object1, _object2)
    , m_potentialEnergy( 0.0 ), useTopology( false ), topology ( NULL )
    , m_filename( initData(&m_filename,std::string(""),"filename","File name from which the spring informations are loaded") )
    , m_stiffness( initData(&m_stiffness,1.0,"stiffness","Default edge stiffness used in absence of file information") )
    , m_viscosity( initData(&m_viscosity,1.0,"viscosity","Default edge viscosity used in absence of file information") )
{
    springArray.setCreateFunction(springCreationFunction);
    springArray.setCreateParameter( (void *) this );
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::init()
{
    this->Inherit::init();
    topology = dynamic_cast<topology::EdgeSetTopology<DataTypes>*>(getContext()->getMainTopology());

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
    const sofa::helper::vector<topology::Edge> &ea=container->getEdgeArray();
    std::cout << "Creating "<<ea.size()<<" Vector Springs from EdgeSetTopology"<<std::endl;
    springArray.resize(ea.size());
    //EdgeSetGeometryAlgorithms<DataTypes> *ga=topology->getEdgeSetGeometryAlgorithms();
    //EdgeLengthArrayInterface<Real,DataTypes> elai(springArray);
    //ga->computeEdgeLength(elai);
    const VecCoord& x0 = *this->mstate1->getX0();
    unsigned int i;
    for (i=0; i<ea.size(); ++i)
    {
        springArray[i].ks=(Real)m_stiffness.getValue();
        springArray[i].kd=(Real)m_viscosity.getValue();
        springArray[i].restVector = x0[ea[i][1]]-x0[ea[i][0]];
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
                    sofa::helper::vector<unsigned int> edgeArray;
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
void VectorSpringForceField<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    //assert(this->mstate);
    m_potentialEnergy = 0;
    const sofa::helper::vector<topology::Edge> &ea=(useTopology)?topology->getEdgeSetTopologyContainer()->getEdgeArray() : edgeArray;
    Coord u;

    f1.resize(x1.size());
    f2.resize(x2.size());

    Deriv relativeVelocity,force;
    for (unsigned int i=0; i<ea.size(); i++)
    {
        const topology::Edge &e=ea[i];
        const Spring &s=springArray[i];
        // paul---------------------------------------------------------------
        Deriv current_direction = x2[e[1]]-x1[e[0]];
        Deriv squash_vector = current_direction - s.restVector;
        Deriv relativeVelocity = v2[e[1]]-v1[e[0]];
        force = (squash_vector * s.ks) + (relativeVelocity * s.kd);

        f1[e[0]]+=force;
        f2[e[1]]-=force;
    }
}

template<class DataTypes>
//void VectorSpringForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
void VectorSpringForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2)
{
    const sofa::helper::vector<topology::Edge> &ea=(useTopology)?topology->getEdgeSetTopologyContainer()->getEdgeArray() : edgeArray;
    Deriv dforce,d;
    unsigned int i;

    df1.resize(dx1.size());
    df2.resize(dx2.size());

    for ( i=0; i<ea.size(); i++)
    {
        const topology::Edge &e=ea[i];
        const Spring &s=springArray[i];
        d = dx2[e[1]]-dx1[e[0]];
        dforce = d*s.ks;
        df1[e[0]]+=dforce;
        df2[e[1]]-=dforce;
    }

}

template<class DataTypes>
void VectorSpringForceField<DataTypes>::draw()
{
    if (!((this->mstate1 == this->mstate2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields()))
        return;
    //const VecCoord& p = *this->mstate->getX();
    const VecCoord& x1 = *this->mstate1->getX();
    const VecCoord& x2 = *this->mstate2->getX();
    const sofa::helper::vector<topology::Edge> &ea=(useTopology)?topology->getEdgeSetTopologyContainer()->getEdgeArray() : edgeArray;

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<springArray.size(); i++)
    {
        const topology::Edge &e=ea[i];
        //const Spring &s=springArray[i];

        glColor4f(0,1,1,0.5f);

        glVertex3d(x1[e[0]][0],x1[e[0]][1],x1[e[0]][2]);
        glVertex3d(x2[e[1]][0],x2[e[1]][1],x2[e[1]][2]);
    }
    glEnd();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
