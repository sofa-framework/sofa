/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MASS_FRAMEDIAGONALMASS_INL
#define SOFA_COMPONENT_MASS_FRAMEDIAGONALMASS_INL

#include <sofa/component/mass/FrameDiagonalMass.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/component/topology/PointData.inl>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/defaulttype/FrameMass.h>
#include <sofa/helper/gl/Axis.h>
//#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.inl>
#include <sofa/component/mapping/SkinningMapping.inl>


#include <sofa/simulation/common/Visitor.h>



namespace sofa
{

namespace component
{

namespace mass
{


using namespace	sofa::component::topology;
using namespace core::componentmodel::topology;

template<class MassType>
void MassPointCreationFunction ( int ,
        void* , MassType & t,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >& )
{
    t=0;
}

template< class DataTypes, class MassType>
inline void MassEdgeCreationFunction ( const sofa::helper::vector<unsigned int> &edgeAdded,
        void* param, vector<MassType> &masses )
{
    FrameDiagonalMass<DataTypes, MassType> *dm= ( FrameDiagonalMass<DataTypes, MassType> * ) param;
    if ( dm->getMassTopologyType() ==FrameDiagonalMass<DataTypes, MassType>::TOPOLOGY_EDGESET )
    {

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass= ( typename DataTypes::Real ) 0;
        unsigned int i;

        for ( i=0; i<edgeAdded.size(); ++i )
        {
            /// get the edge to be added
            const Edge &e=dm->_topology->getEdge ( edgeAdded[i] );
            // compute its mass based on the mass density and the edge length
            if ( dm->edgeGeo )
            {
                mass= ( md*dm->edgeGeo->computeRestEdgeLength ( edgeAdded[i] ) ) / ( typename DataTypes::Real ) 2.0;
            }
            // added mass on its two vertices
            masses[e[0]]+=mass;
            masses[e[1]]+=mass;
        }

    }
}

template< class DataTypes, class MassType>
inline void MassEdgeDestroyFunction ( const sofa::helper::vector<unsigned int> &edgeRemoved,
        void* param, vector<MassType> &masses )
{
    FrameDiagonalMass<DataTypes, MassType> *dm= ( FrameDiagonalMass<DataTypes, MassType> * ) param;
    if ( dm->getMassTopologyType() ==FrameDiagonalMass<DataTypes, MassType>::TOPOLOGY_EDGESET )
    {

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass= ( typename DataTypes::Real ) 0;
        unsigned int i;

        for ( i=0; i<edgeRemoved.size(); ++i )
        {
            /// get the edge to be added
            const Edge &e=dm->_topology->getEdge ( edgeRemoved[i] );
            // compute its mass based on the mass density and the edge length
            if ( dm->edgeGeo )
            {
                mass= ( md*dm->edgeGeo->computeRestEdgeLength ( edgeRemoved[i] ) ) / ( typename DataTypes::Real ) 2.0;
            }
            // added mass on its two vertices
            masses[e[0]]-=mass;
            masses[e[1]]-=mass;
        }

    }
}

template< class DataTypes, class MassType>
inline void MassTriangleCreationFunction ( const sofa::helper::vector<unsigned int> &triangleAdded,
        void* param, vector<MassType> &masses )
{
    FrameDiagonalMass<DataTypes, MassType> *dm= ( FrameDiagonalMass<DataTypes, MassType> * ) param;
    if ( dm->getMassTopologyType() ==FrameDiagonalMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET )
    {

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass= ( typename DataTypes::Real ) 0;
        unsigned int i;

        for ( i=0; i<triangleAdded.size(); ++i )
        {
            /// get the triangle to be added
            const Triangle &t=dm->_topology->getTriangle ( triangleAdded[i] );
            // compute its mass based on the mass density and the triangle area
            if ( dm->triangleGeo )
            {
                mass= ( md*dm->triangleGeo->computeRestTriangleArea ( triangleAdded[i] ) ) / ( typename DataTypes::Real ) 3.0;
            }
            // removed  mass on its three vertices
            masses[t[0]]+=mass;
            masses[t[1]]+=mass;
            masses[t[2]]+=mass;
        }

    }
}

template< class DataTypes, class MassType>
inline void MassTriangleDestroyFunction ( const sofa::helper::vector<unsigned int> &triangleRemoved,
        void* param, vector<MassType> &masses )
{
    FrameDiagonalMass<DataTypes, MassType> *dm= ( FrameDiagonalMass<DataTypes, MassType> * ) param;
    if ( dm->getMassTopologyType() ==FrameDiagonalMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET )
    {

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass= ( typename DataTypes::Real ) 0;
        unsigned int i;

        for ( i=0; i<triangleRemoved.size(); ++i )
        {
            /// get the triangle to be added
            const Triangle &t=dm->_topology->getTriangle ( triangleRemoved[i] );
            // compute its mass based on the mass density and the triangle area
            if ( dm->triangleGeo )
            {
                mass= ( md*dm->triangleGeo->computeRestTriangleArea ( triangleRemoved[i] ) ) / ( typename DataTypes::Real ) 3.0;
            }
            // removed  mass on its three vertices
            masses[t[0]]-=mass;
            masses[t[1]]-=mass;
            masses[t[2]]-=mass;
            // Commented to prevent from printing in case of triangle removal
            //serr<< "mass vertex " << t[0]<< " = " << masses[t[0]]<<sendl;
            //serr<< "mass vertex " << t[1]<< " = " << masses[t[1]]<<sendl;
            //serr<< "mass vertex " << t[2]<< " = " << masses[t[2]]<<sendl;
        }

    }
}

template< class DataTypes, class MassType>
inline void MassTetrahedronCreationFunction ( const sofa::helper::vector<unsigned int> &tetrahedronAdded,
        void* param, vector<MassType> &masses )
{
    FrameDiagonalMass<DataTypes, MassType> *dm= ( FrameDiagonalMass<DataTypes, MassType> * ) param;
    if ( dm->getMassTopologyType() ==FrameDiagonalMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET )
    {

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass= ( typename DataTypes::Real ) 0;
        unsigned int i;

        for ( i=0; i<tetrahedronAdded.size(); ++i )
        {
            /// get the tetrahedron to be added
            const Tetrahedron &t=dm->_topology->getTetrahedron ( tetrahedronAdded[i] );
            // compute its mass based on the mass density and the tetrahedron volume
            if ( dm->tetraGeo )
            {
                mass= ( md*dm->tetraGeo->computeRestTetrahedronVolume ( tetrahedronAdded[i] ) ) / ( typename DataTypes::Real ) 4.0;
            }
            // removed  mass on its four vertices
            masses[t[0]]+=mass;
            masses[t[1]]+=mass;
            masses[t[2]]+=mass;
            masses[t[3]]+=mass;

        }

    }
}

template< class DataTypes, class MassType>
inline void MassTetrahedronDestroyFunction ( const sofa::helper::vector<unsigned int> &tetrahedronRemoved,
        void* param, vector<MassType> &masses )
{
    FrameDiagonalMass<DataTypes, MassType> *dm= ( FrameDiagonalMass<DataTypes, MassType> * ) param;
    if ( dm->getMassTopologyType() ==FrameDiagonalMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET )
    {

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass= ( typename DataTypes::Real ) 0;
        unsigned int i;

        for ( i=0; i<tetrahedronRemoved.size(); ++i )
        {
            /// get the tetrahedron to be added
            const Tetrahedron &t=dm->_topology->getTetrahedron ( tetrahedronRemoved[i] );
            if ( dm->tetraGeo )
            {
                // compute its mass based on the mass density and the tetrahedron volume
                mass= ( md*dm->tetraGeo->computeRestTetrahedronVolume ( tetrahedronRemoved[i] ) ) / ( typename DataTypes::Real ) 4.0;
            }
            // removed  mass on its four vertices
            masses[t[0]]-=mass;
            masses[t[1]]-=mass;
            masses[t[2]]-=mass;
            masses[t[3]]-=mass;
        }

    }
}


using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;


template <class DataTypes, class MassType>
FrameDiagonalMass<DataTypes, MassType>::FrameDiagonalMass()
    : f_mass ( initData ( &f_mass, "mass", "values of the particles masses" ) )
    , f_mass0 ( initData ( &f_mass0, "mass0", "values of the particles masses" ) )
    , m_massDensity ( initData ( &m_massDensity, ( Real ) 1.0,"massDensity", "mass density that allows to compute the  particles masses from a mesh topology and geometry.\nOnly used if > 0" ) )
    , showCenterOfGravity ( initData ( &showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , showAxisSize ( initData ( &showAxisSize, 1.0f, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , fileMass( initData(&fileMass,  "fileMass", "File to specify the mass" ) )
    , damping ( initData ( &damping, 0.0f, "damping", "add a force which is \"- damping * speed\"" ) )
    , rotateMass ( initData ( &rotateMass, false, "rotateMass", "Rotate mass matrices instead of computing it at each step." ) )
    , topologyType ( TOPOLOGY_UNKNOWN )
{
    this->addAlias(&fileMass,"filename");
}




template <class DataTypes, class MassType>
FrameDiagonalMass<DataTypes, MassType>::~FrameDiagonalMass()
{
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::clear()
{
    MassVector& masses = *f_mass.beginEdit();
    masses.clear();
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addMass ( const MassType& m )
{
    MassVector& masses = *f_mass.beginEdit();
    masses.push_back ( m );
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::resize ( int vsize )
{
    MassVector& masses = *f_mass.beginEdit();
    masses.resize ( vsize );
    f_mass.endEdit();
    if( rotateMass.getValue())
    {
        MassVector& masses0 = *f_mass0.beginEdit();
        masses0.resize ( vsize );
        f_mass0.endEdit();
    }
}

// -- Mass interface
template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addMDx ( VecDeriv& res, const VecDeriv& dx, double factor )
{
    const MassVector &masses= f_mass.getValue();
    if ( factor == 1.0 )
    {
        for ( unsigned int i=0; i<dx.size(); i++ )
        {
            res[i] += dx[i] * masses[i];
        }
    }
    else
    {
        for ( unsigned int i=0; i<dx.size(); i++ )
        {
            res[i] += ( dx[i]* masses[i] ) * ( Real ) factor; // damping.getValue() * invSqrDT;
        }
    }
}



template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::accFromF ( VecDeriv& a, const VecDeriv& f )
{

    const MassVector &masses= f_mass.getValue();
    for ( unsigned int i=0; i<f.size(); i++ )
    {
        a[i] = f[i] / masses[i];
    }
}

template <class DataTypes, class MassType>
double FrameDiagonalMass<DataTypes, MassType>::getKineticEnergy ( const VecDeriv& v )
{

    const MassVector &masses= f_mass.getValue();
    double e = 0;
    for ( unsigned int i=0; i<masses.size(); i++ )
    {
        e += v[i]*masses[i]*v[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }
    return e/2;
}

template <class DataTypes, class MassType>
double FrameDiagonalMass<DataTypes, MassType>::getPotentialEnergy ( const VecCoord& x )
{
    double e = 0;
    const MassVector &masses= f_mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2] );
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        e -= theGravity.getVCenter() *masses[i].mass*x[i].getCenter();
    }
    return e;
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addMToMatrix ( defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset )
{
    const MassVector &masses= f_mass.getValue();
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    for ( unsigned int i=0; i<masses.size(); i++ )
        calc ( mat, masses[i], offset + N*i, mFact );
}


template <class DataTypes, class MassType>
double FrameDiagonalMass<DataTypes, MassType>::getElementMass ( unsigned int index ) const
{
    return ( SReal ) ( f_mass.getValue() [index] );
}


//TODO: special case for Rigid Mass
template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::getElementMass ( unsigned int index, defaulttype::BaseMatrix *m ) const
{
    const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
    if ( m->rowSize() != dimension || m->colSize() != dimension ) m->resize ( dimension,dimension );

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>() ( m, f_mass.getValue() [index], 0, 1 );
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->lastChange();

//	VecMass& masses = *f_mass.beginEdit();
    f_mass.handleTopologyEvents ( itBegin,itEnd );
//	f_mass.endEdit();
}


template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::init()
{
    if (!fileMass.getValue().empty()) load(fileMass.getFullPath().c_str());
    Inherited::init();
}


template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::reinit()
{
    unsigned int nbPt = this->volMass->size();
    for( unsigned int i = 0; i < nbPt; i++) (*this->volMass)[i] = m_massDensity.getValue();

    this->resize ( (*J0).size() );
    if( rotateMass.getValue())
    {
        MassVector& vecMass0 = * ( f_mass0.beginEdit() );
        for ( unsigned int i=0; i<(*J0).size(); i++ )
            updateMass ( vecMass0[i], (*J0)[i], *vol, *volMass );
        f_mass.endEdit();
    }
    else
    {
        MassVector& vecMass = * ( f_mass.beginEdit() );
        for ( unsigned int i=0; i<(*J).size(); i++ )
            updateMass ( vecMass[i], (*J)[i], *vol, *volMass );
        f_mass.endEdit();
    }

    Inherited::reinit();
}


template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::bwdInit()
{
    dqStorage = NULL;
    vector<SMapping*> vSMapping;
    sofa::core::objectmodel::BaseContext* context=  this->getContext();
    context->get<SMapping>( &vSMapping, core::objectmodel::BaseContext::SearchDown);
    SMapping* sMapping = NULL;
    for( typename vector<SMapping *>::iterator it = vSMapping.begin(); it != vSMapping.end(); it++)
    {
        sMapping = (*it);
        if( sMapping && sMapping->computeAllMatrices.getValue() )
        {
            dqStorage = sMapping;
            break;
        }
    }
    if ( ! dqStorage )
    {
        serr << "Can't find dqStorage component." << sendl;
        return;
    }
    else
    {
        this->J = & dqStorage->J;
        this->J0 = & dqStorage->J0;
        this->vol = & dqStorage->vol;
        this->volMass = & dqStorage->volMass;
    }

    unsigned int nbPt = this->volMass->size();
    for( unsigned int i = 0; i < nbPt; i++) (*this->volMass)[i] = m_massDensity.getValue();

    this->resize ( (*J0).size() );
    if( rotateMass.getValue())
    {
        MassVector& vecMass0 = * ( f_mass0.beginEdit() );
        for ( unsigned int i=0; i<(*J0).size(); i++ )
            updateMass ( vecMass0[i], (*J0)[i], *vol, *volMass );
        f_mass.endEdit();
    }
    else
    {
        MassVector& vecMass = * ( f_mass.beginEdit() );
        for ( unsigned int i=0; i<(*J).size(); i++ )
            updateMass ( vecMass[i], (*J)[i], *vol, *volMass );
        f_mass.endEdit();
    }
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addGravityToV ( double dt )
{
    if ( this->mstate )
    {
        VecDeriv& v = *this->mstate->getV();

        // gravity
        Vec3d g ( this->getContext()->getLocalGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2] );
        Deriv hg = theGravity * ( typename DataTypes::Real ) dt;

        for ( unsigned int i=0; i<v.size(); i++ )
        {
            v[i] += hg;
        }
    }
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addForce ( VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& v )
{
    if( rotateMass.getValue())
    {
        // Rotate the mass
        const VecCoord& xfrom0 = *this->mstate->getX0();
        const VecCoord& xfrom = *this->mstate->getX();
        const MassVector& vecMass0 = f_mass0.getValue();
        MassVector& vecMass = (*f_mass.beginEdit());
        if( vecMass0.size() != xfrom.size())
        {
            // Update the mass
            int size = vecMass0.size();
            this->resize ( (*J0).size() );
            MassVector& vecMasses0 = * ( f_mass0.beginEdit() );
            for ( unsigned int i=size; i<(*J0).size(); i++ )
                updateMass ( vecMasses0[i], (*J0)[i], *vol, *volMass );
            f_mass0.endEdit();
        }
        for( unsigned int i = 0; i < xfrom.size(); ++i)
        {
            rotateM( vecMass[i].inertiaMatrix, vecMass0[i].inertiaMatrix, xfrom[i].getOrientation(), xfrom0[i].getOrientation());
            vecMass[i].recalc();
        }
        f_mass.endEdit();
    }
    else
    {
        // Update the mass
        this->resize ( (*J).size() );
        MassVector& vecMass = * ( f_mass.beginEdit() );
        for ( unsigned int i=0; i<(*J).size(); i++ )
            updateMass ( vecMass[i], (*J)[i], *vol, *volMass );
        f_mass.endEdit();
    }

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;

    const MassVector &masses= f_mass.getValue();

    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2] );

    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector ( aframe );

    // add weight and inertia force
    const double& invDt = 1./this->getContext()->getDt();
    for ( unsigned int i=0; i<masses.size(); i++ )
    {
        Deriv fDamping = - (masses[i] * v[i] * damping.getValue() * invDt);
        f[i] += theGravity*masses[i] + fDamping; //  + core::componentmodel::behavior::inertiaForce ( vframe,aframe,masses[i],x[i],v[i] );
    }
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::draw()
{
    const MassVector &masses= f_mass.getValue();
    if ( !this->getContext()->getShowBehaviorModels() ) return;
    VecCoord& x = *this->mstate->getX();
    if( x.size() != masses.size()) return;
    Real totalMass=0;
    RigidTypes::Vec3 gravityCenter;
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        const Quat& orient = x[i].getOrientation();
        const RigidTypes::Vec3& center = x[i].getCenter();

        simulation::getSimulation()->DrawUtility.drawFrame(center, orient, Vec3d(1,1,1)*showAxisSize.getValue() );

        gravityCenter += ( center * masses[i].mass );
        totalMass += masses[i].mass;
    }

    if ( showCenterOfGravity.getValue() )
    {
        glColor3f ( 1,1,0 );
        glBegin ( GL_LINES );
        gravityCenter /= totalMass;
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( showAxisSize.getValue(),0,0 ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( showAxisSize.getValue(),0,0 ) );
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( 0,showAxisSize.getValue(),0 ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( 0,showAxisSize.getValue(),0 ) );
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( 0,0,showAxisSize.getValue() ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( 0,0,showAxisSize.getValue() ) );
        glEnd();
    }
}

template <class DataTypes, class MassType>
bool FrameDiagonalMass<DataTypes, MassType>::addBBox ( double* minBBox, double* maxBBox )
{
    const VecCoord& x = *this->mstate->getX();
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        //const Coord& p = x[i];
        Real p[3] = {0.0, 0.0, 0.0};
        DataTypes::get ( p[0],p[1],p[2],x[i] );
        for ( int c=0; c<3; c++ )
        {
            if ( p[c] > maxBBox[c] ) maxBBox[c] = p[c];
            if ( p[c] < minBBox[c] ) minBBox[c] = p[c];
        }
    }
    return true;
}

template <class DataTypes, class MassType>
class FrameDiagonalMass<DataTypes, MassType>::Loader : public helper::io::MassSpringLoader
{
public:
    FrameDiagonalMass<DataTypes, MassType>* dest;
    Loader ( FrameDiagonalMass<DataTypes, MassType>* dest ) : dest ( dest ) {}
    virtual void addMass ( SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal mass, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/ )
    {
        dest->addMass ( MassType ( ( Real ) mass ) );
    }
};

template <class DataTypes, class MassType>
bool FrameDiagonalMass<DataTypes, MassType>::load ( const char *filename )
{
    clear();
    if ( filename!=NULL && filename[0]!='\0' )
    {
        Loader loader ( this );
        return loader.load ( filename );
    }
    else return false;
}


template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::updateMass ( MassType& mass, const VMat36& J, const VD& vol, const VD& volmass )
{
    // Mass_ij=sum(d.p.Ji^TTJj)
    int j,nbP=vol.size();
    Mat63 JT;
    Mat66 JJT;

    mass.mass = 1.0;//volmass[i] * vol[i]; (in skinning method, each point mass is distributed on frames depending on weights and so, are directly stored in the inertia matrix)
    Mat66& frameMass = mass.inertiaMatrix;
    // Init the diagonal block 'i'
    for ( unsigned int l = 0; l < 6; l++ )
        for ( unsigned int m = 0; m < 6; m++ )
            frameMass[l][m] = 0.0;

    for ( j=0; j<nbP; j++ )
    {
        JT.transpose ( J[j] );
        JT*=vol[j] * volmass[j];
        //*/ Without lumping
        JJT=JT*J[j];
        frameMass += JJT;
        /*/
        //					serr << "J[i][j]: " << J[i][j] << sendl;
        for ( unsigned int k=0;k<nbDOF;k++ )
        	{
        		JJT=JT*J[k][j];
        		frameMass += JJT;
        	}
        	//*/
    }
    mass.recalc();
    //serr << "Mass["<<i<<"]: " << vecMass[i] << sendl;
}

template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::rotateM( Mat66& M, const Mat66& M0, const Quat& q, const Quat& q0)
{
    int i,j,k;

    Quat q0_inv; q0_inv[0]=-q0[0]; q0_inv[1]=-q0[1]; q0_inv[2]=-q0[2]; q0_inv[3]=q0[3];
    Quat qrel;
    qrel = q * q0_inv;
    Mat33 R; QtoR( R, qrel);
    Mat66 M1; M1.fill(0);
    M.fill(0);
    for(i=0; i<3; i++) for(j=0; j<3; j++) for(k=0; k<3; k++)
            {
                M1[i][j]+=R[i][k]*M0[k][j];
                M1[i][j+3]+=R[i][k]*M0[k][j+3];
                M1[i+3][j]+=R[i][k]*M0[k+3][j];
                M1[i+3][j+3]+=R[i][k]*M0[k+3][j+3];
            }
    for(i=0; i<3; i++) for(j=0; j<3; j++) for(k=0; k<3; k++)
            {
                M[i][j]+=M1[i][k]*R[j][k];
                M[i][j+3]+=M1[i][k+3]*R[j][k];
                M[i+3][j]+=M1[i+3][k]*R[j][k];
                M[i+3][j+3]+=M1[i+3][k+3]*R[j][k];
            }
}

template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::QtoR( Mat33& M, const Quat& q)
{
// q to M
    double xs = q[0]*2., ys = q[1]*2., zs = q[2]*2.;
    double wx = q[3]*xs, wy = q[3]*ys, wz = q[3]*zs;
    double xx = q[0]*xs, xy = q[0]*ys, xz = q[0]*zs;
    double yy = q[1]*ys, yz = q[1]*zs, zz = q[2]*zs;
    M[0][0] = 1.0 - (yy + zz); M[0][1]= xy - wz; M[0][2] = xz + wy;
    M[1][0] = xy + wz; M[1][1] = 1.0 - (xx + zz); M[1][2] = yz - wx;
    M[2][0] = xz - wy; M[2][1] = yz + wx; M[2][2] = 1.0 - (xx + yy);
}


} // namespace mass

} // namespace component

} // namespace sofa

#endif
