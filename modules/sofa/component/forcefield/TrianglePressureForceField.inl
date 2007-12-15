#include <sofa/component/forcefield/TrianglePressureForceField.h>
#include <sofa/component/topology/TriangleSubsetData.inl>
#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>

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

template <class DataTypes> TrianglePressureForceField<DataTypes>::~TrianglePressureForceField()
{
}
// Handle topological changes
template <class DataTypes> void  TrianglePressureForceField<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());


    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();


    trianglePressureMap.handleTopologyEvents(itBegin,itEnd,tst->getTriangleSetTopologyContainer()->getNumberOfTriangles());

}
template <class DataTypes> void TrianglePressureForceField<DataTypes>::init()
{
    //std::cerr << "initializing TrianglePressureForceField" << std::endl;
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();

    tst= static_cast<sofa::component::topology::TriangleSetTopology<DataTypes> *>(getContext()->getMainTopology());
    assert(tst!=0);

    if (tst==NULL)
    {
        std::cerr << "ERROR(TrianglePressureForceField): object must have an TriangleSetTopology.\n";
        return;
    }

    if (dmin.getValue()!=dmax.getValue())
    {
        selectTrianglesAlongPlane();
    }
    if (triangleList.getValue().length()>0)
    {
        selectTrianglesFromString();
    }

    initTriangleInformation();

}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& /*v*/)
{
    Deriv force;

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;
    const std::vector<Triangle> &ta=tst->getTriangleSetTopologyContainer()->getTriangleArray();

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        force=(*it).second.force/3;
        f[ta[(*it).first][0]]+=force;
        f[ta[(*it).first][1]]+=force;
        f[ta[(*it).first][2]]+=force;

    }
}

template <class DataTypes>
double TrianglePressureForceField<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    cerr<<"TrianglePressureForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template<class DataTypes>
void TrianglePressureForceField<DataTypes>::initTriangleInformation()
{
    topology::TriangleSetGeometryAlgorithms<DataTypes> *esga=tst->getTriangleSetGeometryAlgorithms();

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        (*it).second.area=esga->computeRestTriangleArea((*it).first);
        (*it).second.force=pressure.getValue()*(*it).second.area;
    }
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::updateTriangleInformation()
{
    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        (*it).second.force=((*it).second.area)*pressure.getValue();
    }
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = *this->mstate->getX0();
    std::vector<bool> vArray;
    unsigned int i,n;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    const std::vector<Triangle> &ta=tst->getTriangleSetTopologyContainer()->getTriangleArray();

    for (n=0; n<ta.size(); ++n)
    {
        if ((vArray[ta[n][0]]) && (vArray[ta[n][1]])&& (vArray[ta[n][2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            trianglePressureMap[n]=t;
        }
    }
}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesFromString()
{
    std::string inputString=triangleList.getValue();
    unsigned int i;
    do
    {
        const char *str=inputString.c_str();
        for(i=0; (i<inputString.length())&&(str[i]!=','); ++i);
        TrianglePressureInformation t;

        if (i==inputString.length())
        {
            trianglePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i;
        }
        else
        {
            inputString[i]='\0';
            trianglePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i+1;
        }
    }
    while (inputString.length()>0);

}
template<class DataTypes>
void TrianglePressureForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    glColor4f(0,1,0,1);

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;
    const std::vector<Triangle> &ta=tst->getTriangleSetTopologyContainer()->getTriangleArray();

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        helper::gl::glVertexT(x[ta[(*it).first][0]]);
        helper::gl::glVertexT(x[ta[(*it).first][1]]);
        helper::gl::glVertexT(x[ta[(*it).first][2]]);
    }
    glEnd();


    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa
