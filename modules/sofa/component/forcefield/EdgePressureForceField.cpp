#include <sofa/component/forcefield/EdgePressureForceField.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/helper/gl/template.h>
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <set>
#include <sofa/defaulttype/Vec3Types.h>

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

using std::cerr;
using std::cout;
using std::endl;

template <class DataTypes> EdgePressureForceField<DataTypes>::~EdgePressureForceField()
{
}

template <class DataTypes> void EdgePressureForceField<DataTypes>::init()
{
    //std::cerr << "initializing EdgePressureForceField" << std::endl;
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();

    _mesh = dynamic_cast<sofa::component::topology::MeshTopology*>(this->getContext()->getTopology());

    // get restPosition
    if (_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX();
        _initialPoints.setValue(p);
    }
    if (usePlaneSelection)
        selectEdgesAlongPlane();

    initEdgeInformation();

}

template <class DataTypes>
void EdgePressureForceField<DataTypes>::addEdgePressure(Index ind1, Index ind2)
{
//	Real length=0;
    Deriv force;
    EdgePressureInformation edge;


//	force=pressure*length;

    edge.index[0]=ind1;
    edge.index[1]=ind2;
//	edge.length=length;
//	edge.force=force;

    edgeInfo.resize(nbEdges+1);
    edgeInfo[nbEdges]=edge;

    nbEdges++;
}
template <class DataTypes>
void EdgePressureForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/)
{
    unsigned int i;
    EdgePressureInformation *edge;

    f.resize(x.size());


    for(i=0; i<nbEdges; i++ )
    {

        edge=&edgeInfo[i];
        f[edge->index[0]]+=edge->force/2;
        f[edge->index[1]]+=edge->force/2;

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
    unsigned int i;

    for(i=0; i<nbEdges; i++ )
    {
        edgeInfo[i].length=(_initialPoints.getValue()[edgeInfo[i].index[0]] -
                _initialPoints.getValue()[edgeInfo[i].index[1]]).norm();

        edgeInfo[i].force=pressure*edgeInfo[i].length;
        //	std::cerr<< "force=" << edgeInfo[i].force<< std::endl;

    }
}


template<class DataTypes>
void EdgePressureForceField<DataTypes>::updateEdgeInformation()
{
    unsigned int i;
//	std::cerr<< "pressure=" << pressure << std::endl;
    for(i=0; i<nbEdges; i++ )
    {
        edgeInfo[i].force=pressure*edgeInfo[i].length;
        //	std::cerr<< "force=" << edgeInfo[i].force<< std::endl;

    }
}
template <class DataTypes>
void EdgePressureForceField<DataTypes>::setNormal(Coord dir)
{
    if (dir.norm2()>0)
    {
        normal=dir;
    }
}

template <class DataTypes>
void EdgePressureForceField<DataTypes>::selectEdgesAlongPlane()
{
    VecCoord& x = *this->mstate->getX();
    std::vector<unsigned int> vArray;
    Index i,j,k,l,m;
    int n;
    std::set<std::pair<unsigned int,unsigned int> > edgeSet;

    vArray.resize(x.size());
    i = 0;

    for(unsigned int cpt=0; cpt<x.size(); ++cpt)
    {
        vArray[i]=isPointInPlane(x[i]);
        i++;
    }

    sofa::component::topology::MeshTopology* _mesh = dynamic_cast<sofa::component::topology::MeshTopology*>(getContext()->getTopology());

    if (_mesh==NULL || (_mesh->getTriangles().empty()))
    {
        std::cerr << "ERROR(EdgePressureForceField): object must have a triangular MeshTopology.\n";
        return;
    }

    const topology::MeshTopology::SeqTriangles *_indexedElements = & (_mesh->getTriangles());

    for (n=0; n<_mesh->getNbTriangles(); ++n)
    {
        for (j=0; j<3; ++j)
        {
            k=(*_indexedElements)[n][(j+1)%3];
            l=(*_indexedElements)[n][(j+2)%3];
            if ((vArray[k]) && (vArray[l]))
            {
                if (k>l)
                {
                    m=k;
                    k=l;
                    l=m;
                }
                if (edgeSet.find(std::pair<unsigned int,unsigned int>(k,l))==edgeSet.end())
                {
                    edgeSet.insert(std::pair<unsigned int,unsigned int>(k,l));
//					std::cerr<<"adding vertices : "<< k << "and " << l << std::endl;
                    addEdgePressure((Index) k,(Index) l);
                }
            }
        }
    }
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    const VecCoord& x = *this->mstate->getX();

    Real dx=(x[1] -x[0]).norm();
    Real dy=(x[2] -x[1]).norm();
    Real ex=fabs(10-dx)/10;
    Real ey=fabs(10-dy)/10;

    std::cerr << "dx="<<dx<< " dy="<<dy<<" ex="<<ex<<" ey="<<ey << std::endl;

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    unsigned int i;
    glColor4f(0,1,0,1);

    for(i=0; i<edgeInfo.size(); i++ )
    {
        helper::gl::glVertexT(x[edgeInfo[i].index[0]]);
        helper::gl::glVertexT(x[edgeInfo[i].index[1]]);
    }
    glEnd();


    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

template<class DataTypes>
void EdgePressureForceField<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::parse(arg);
    EdgePressureForceField<DataTypes>* obj = this;
    if (arg->getAttribute("px"))
    {
        obj->setPressure(Vec3d(atof(arg->getAttribute("px","0")), atof(arg->getAttribute("py","0")), atof(arg->getAttribute("pz","0"))));
    }
    if (arg->getAttribute("edges"))
    {
        const char* str = arg->getAttribute("edges");
        const char* str2 = NULL;
        unsigned int ind1,ind2;
        for(;;)
        {
            ind1 = (unsigned int)strtod(str,(char**)&str2);
            if (str2==str) break;
            str = str2;
            ind2 = (unsigned int)strtod(str,(char**)&str2);
            if (str2==str) break;
            str = str2;
            obj->addEdgePressure(ind1,ind2);
        }
    }
    if (arg->getAttribute("normal"))
    {
        const char* str = arg->getAttribute("normal");
        const char* str2 = NULL;
        Real val[3];
        unsigned int i;
        for(i=0; i<3; i++)
        {
            val[i] = (Real)strtod(str,(char**)&str2);
            if (str2==str) break;
            str = str2;
        }
        Coord dir(val);
        obj->setNormal(dir);
    }
    if (arg->getAttribute("distance"))
    {
        const char* str = arg->getAttribute("distance");
        const char* str2 = NULL;
        Real val[2];
        unsigned int i;
        for(i=0; i<2; i++)
        {
            val[i] = (Real)strtod(str,(char**)&str2);
            if (str2==str) break;
            str = str2;
        }
        obj->setDminAndDmax(val[0],val[1]);
    }
}

using namespace sofa::defaulttype;

template class EdgePressureForceField<Vec3dTypes>;
template class EdgePressureForceField<Vec3fTypes>;

SOFA_DECL_CLASS(EdgePressureForceField)

int EdgePressureForceFieldClass = core::RegisterObject("TODO-EdgePressure")
        .add< EdgePressureForceField<Vec3dTypes> >()
        .add< EdgePressureForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
