#include <sofa/component/forcefield/EdgePressureForceField.h>
#include <sofa/component/topology/EdgeSubsetData.inl>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>

#ifdef _WIN32
#include <windows.h>
#endif

// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::topology;

using std::cerr;
using std::cout;
using std::endl;

template <class DataTypes> EdgePressureForceField<DataTypes>::~EdgePressureForceField()
{
}
// Handle topological changes
template <class DataTypes> void  EdgePressureForceField<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());


    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();


    edgePressureMap.handleTopologyEvents(itBegin,itEnd,est->getEdgeSetTopologyContainer()->getNumberOfEdges());

}
template <class DataTypes> void EdgePressureForceField<DataTypes>::init()
{
    //std::cerr << "initializing EdgePressureForceField" << std::endl;
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();

    est= static_cast<sofa::component::topology::EdgeSetTopology<DataTypes> *>(getContext()->getMainTopology());
    assert(est!=0);

    if (est==NULL)
    {
        std::cerr << "ERROR(EdgePressureForceField): object must have an EdgeSetTopology.\n";
        return;
    }

    if (dmin.getValue()!=dmax.getValue())
    {
        selectEdgesAlongPlane();
    }
    if (edgeList.getValue().length()>0)
    {
        selectEdgesFromString();
    }

    initEdgeInformation();

}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& /*v*/)
{
    Deriv force;

    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;
    const std::vector<Edge> &ea=est->getEdgeSetTopologyContainer()->getEdgeArray();

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        force=(*it).second.force/2;
        f[ea[(*it).first].first]+=force;
        f[ea[(*it).first].second]+=force;

    }
}

template <class DataTypes>
double EdgePressureForceField<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    cerr<<"EdgePressureForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::initEdgeInformation()
{
    topology::EdgeSetGeometryAlgorithms<DataTypes> *esga=est->getEdgeSetGeometryAlgorithms();

    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        (*it).second.length=esga->computeRestEdgeLength((*it).first);
        (*it).second.force=pressure.getValue()*(*it).second.length;
    }
}


template<class DataTypes>
void EdgePressureForceField<DataTypes>::updateEdgeInformation()
{
    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        (*it).second.force=((*it).second.length)*pressure.getValue();
    }
}


template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesAlongPlane()
{
    const VecCoord& x = *this->mstate->getX0();
    std::vector<bool> vArray;
    unsigned int i,n;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    const std::vector<Edge> &ea=est->getEdgeSetTopologyContainer()->getEdgeArray();

    for (n=0; n<ea.size(); ++n)
    {
        if ((vArray[ea[n].first]) && (vArray[ea[n].second]))
        {
            // insert a dummy element : computation of pressure done later
            EdgePressureInformation t;
            edgePressureMap[n]=t;
        }
    }
}

template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesFromString()
{
    std::string inputString=edgeList.getValue();
    unsigned int i;
    do
    {
        const char *str=inputString.c_str();
        for(i=0; (i<inputString.length())&&(str[i]!=','); ++i);
        EdgePressureInformation t;

        if (i==inputString.length())
        {
            edgePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i;
        }
        else
        {
            inputString[i]='\0';
            edgePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i+1;
        }
    }
    while (inputString.length()>0);

}
template<class DataTypes>
void EdgePressureForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    glColor4f(0,1,0,1);

    typename topology::EdgeSubsetData<EdgePressureInformation>::iterator it;
    const std::vector<Edge> &ea=est->getEdgeSetTopologyContainer()->getEdgeArray();

    for(it=edgePressureMap.begin(); it!=edgePressureMap.end(); it++ )
    {
        helper::gl::glVertexT(x[ea[(*it).first].first]);
        helper::gl::glVertexT(x[ea[(*it).first].second]);
    }
    glEnd();


    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa
