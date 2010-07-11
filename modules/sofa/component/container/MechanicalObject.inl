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
#ifndef SOFA_COMPONENT_MECHANICALOBJECT_INL
#define SOFA_COMPONENT_MECHANICALOBJECT_INL

#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#ifdef SOFA_SMP
#include <sofa/component/container/MechanicalObjectTasks.inl>
#endif
#include <sofa/component/topology/PointSetTopologyChange.h>

#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/helper/io/MassSpringLoader.h>

#include <sofa/helper/accessor.h>

#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/helper/system/glut.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/simulation/common/Visitor.h>
#endif

#include <sofa/component/linearsolver/SparseMatrix.h>

#include <assert.h>
#include <iostream>

#include <sofa/component/container/MeshLoader.h>

namespace
{

template<class V>
void renumber(V* v, V* tmp, const sofa::helper::vector< unsigned int > &index )
{
    if (v == NULL)
        return;

    if (v->empty())
        return;

    *tmp = *v;
    for (unsigned int i = 0; i < v->size(); ++i)
        (*v)[i] = (*tmp)[index[i]];
}

} // anonymous namespace

namespace sofa
{

namespace component
{

namespace container
{

using namespace topology;
using namespace sofa::core::topology;
using namespace sofa::defaulttype;

template <class DataTypes>
MechanicalObject<DataTypes>::MechanicalObject()
    : x(initData(&x, "position", "position coordinates of the degrees of freedom"))
    , v(initData(&v, "velocity", "velocity coordinates of the degrees of freedom"))
    , f(initData(&f, "force", "force vector of the degrees of freedom"))
    , internalForces(initData(&internalForces, "internalForce", "internalForces vector of the degrees of freedom"))
    , externalForces(initData(&externalForces, "externalForce", "externalForces vector of the degrees of freedom"))
    , dx(initData(&dx, "derivX", "dx vector of the degrees of freedom"))
    , xfree(initData(&xfree, "free_position", "free position coordinates of the degrees of freedom"))
    , vfree(initData(&vfree, "free_velocity", "free velocity coordinates of the degrees of freedom"))
    , x0(initData(&x0, "rest_position", "rest position coordinates of the degrees of freedom"))
    , c(initData(&c, "constraint", "constraints applied to the degrees of freedom"))
    , restScale(initData(&restScale, (SReal)1.0, "restScale", "optional scaling of rest position coordinates (to simulated pre-existing internal tension)"))
    , debugViewIndices(initData(&debugViewIndices, (bool) false, "debugViewIndices", "Debug : view indices"))
    , debugViewIndicesScale(initData(&debugViewIndicesScale, (float) 0.0001, "debugViewIndicesScale", "Debug : scale for view indices"))
    , reset_position(NULL)
    , v0(NULL)
    , translation(initData(&translation, Vector3(), "translation", "Translation of the DOFs"))
    , rotation(initData(&rotation, Vector3(), "rotation", "Rotation of the DOFs"))
    , scale(initData(&scale, Vector3(1.0,1.0,1.0), "scale3d", "Scale of the DOFs in 3 dimensions"))
    , translation2(initData(&translation2, Vector3(), "translation2", "Translation of the DOFs, applied after the rest position has been computed"))
    , rotation2(initData(&rotation2, Vector3(), "rotation2", "Rotation of the DOFs, applied the after the rest position has been computed"))
    , filename(initData(&filename, std::string(""), "filename", "File corresponding to the Mechanical Object", false))
    , ignoreLoader(initData(&ignoreLoader, (bool) false, "ignoreLoader", "Is the Mechanical Object do not use a loader"))
    , f_reserve(initData(&f_reserve, 0, "reserve", "Size to reserve when creating vectors"))
    , vsize(0)
    , m_gnuplotFileX(NULL)
    , m_gnuplotFileV(NULL)
{
    // HACK
    if (!restScale.isSet())
    {
        restScale.setValue(1);
    }

    m_initialized = false;

    x				.setGroup("Vector");
    v				.setGroup("Vector");
    f				.setGroup("Vector");
    internalForces	.setGroup("Vector");
    externalForces	.setGroup("Vector");
    dx				.setGroup("Vector");
    xfree			.setGroup("Vector");
    vfree			.setGroup("Vector");
    x0				.setGroup("Vector");

    translation		.setGroup("Transformation");
    translation2	.setGroup("Transformation");
    rotation		.setGroup("Transformation");
    rotation2		.setGroup("Transformation");
    scale			.setGroup("Transformation");

    // Deactivate the Filter.
    // MechanicalObjects created during the collision response must not use the filter as it will be empty
    this->forceMask.activate(false);

    // default size is 1
    resize(1);

    m_posId				= VecId::position();
    m_velId				= VecId::velocity();
    m_forceId			= VecId::force();
    m_internalForcesId	= VecId::internalForce();
    m_externalForcesId	= VecId::externalForce();
    m_dxId				= VecId::dx();
    m_freePosId			= VecId::freePosition();
    m_freeVelId			= VecId::freeVelocity();
    m_x0Id				= VecId::restPosition();
    m_constraintId		= VecId::holonomicC();

    setVecCoord(m_posId.index, &x);
    setVecDeriv(m_velId.index, &v);
    setVecDeriv(m_forceId.index, &f);
    setVecDeriv(m_internalForcesId.index, &internalForces);
    setVecDeriv(m_externalForcesId.index, &externalForces);
    setVecDeriv(m_dxId.index, &dx);
    setVecCoord(m_freePosId.index, &xfree);
    setVecDeriv(m_freeVelId.index, &vfree);
    setVecCoord(m_x0Id.index, &x0);
    setVecConst(m_constraintId.index, &c);

    // Set f to internalForces.
    m_forceId = m_internalForcesId;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::initGnuplot(const std::string path)
{
    if( !this->getName().empty() )
    {
        if (m_gnuplotFileX != NULL)
            delete m_gnuplotFileX;

        if (m_gnuplotFileV != NULL)
            delete m_gnuplotFileV;

        m_gnuplotFileX = new std::ofstream( (path + this->getName()+"_x.txt").c_str() );
        m_gnuplotFileV = new std::ofstream( (path + this->getName()+"_v.txt").c_str() );
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::exportGnuplot(Real time)
{
    if( m_gnuplotFileX!=NULL )
    {
        (*m_gnuplotFileX) << time <<"\t"<< *getX() << std::endl;
    }

    if( m_gnuplotFileV!=NULL )
    {
        (*m_gnuplotFileV) << time <<"\t"<< *getV() << std::endl;
    }
}

template <class DataTypes>
MechanicalObject<DataTypes> &MechanicalObject<DataTypes>::operator = (const MechanicalObject& obj)
{
    resize(obj.getSize());
    /*    *getX() = *obj.getX();
    if( obj.x0 != NULL ){
    x0 = new VecCoord;
    *x0 = *obj.x0;
    }
    *getV() = *obj.getV();
    if( obj.v0 != NULL ){
    v0 = new VecDeriv;
    *v0 = *obj.v0;
    }*/
    return *this;
}

template<class DataTypes>
class MechanicalObject<DataTypes>::Loader : public helper::io::MassSpringLoader
{
public:
    MechanicalObject<DataTypes>* dest;
    int index;
    Loader(MechanicalObject<DataTypes>* dest) : dest(dest), index(0) {}

    virtual void addMass(SReal px, SReal py, SReal pz, SReal vx, SReal vy, SReal vz, SReal /*mass*/, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/)
    {
        dest->resize(index+1);
        DataTypes::set((*dest->getX())[index], px, py, pz);
        DataTypes::set((*dest->getV())[index], vx, vy, vz);
        ++index;
    }
};

template<class DataTypes>
bool MechanicalObject<DataTypes>::load(const char* filename)
{
    typename MechanicalObject<DataTypes>::Loader loader(this);
    return loader.load(filename);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::parse ( BaseObjectDescription* arg )
{
    if (arg->getAttribute("filename"))
    {
        filename.setValue(arg->getAttribute("filename"));
    }

    if (!filename.getValue().empty())
    {
        load(filename.getFullPath().c_str());
        filename.setValue(std::string("")); //clear the field filename: When we save the scene, we don't need anymore the filename
    }

    Inherited::parse(arg);

    // DEPRECATED: Warning, you should not use these parameters, but a TransformEngine instead
    if (arg->getAttribute("scale") != NULL)
    {
        SReal s = (SReal)atof(arg->getAttribute("scale", "1.0"));
        scale.setValue(Vector3(s, s, s));
    }

    if (arg->getAttribute("sx") != NULL || arg->getAttribute("sy") != NULL || arg->getAttribute("sz") != NULL)
    {
        scale.setValue(Vector3((SReal)(atof(arg->getAttribute("sx","1.0"))),(SReal)(atof(arg->getAttribute("sy","1.0"))),(SReal)(atof(arg->getAttribute("sz","1.0")))));
    }

    if (arg->getAttribute("rx") != NULL || arg->getAttribute("ry") != NULL || arg->getAttribute("rz") != NULL)
    {
        rotation.setValue(Vector3((SReal)(atof(arg->getAttribute("rx","0.0"))),(SReal)(atof(arg->getAttribute("ry","0.0"))),(SReal)(atof(arg->getAttribute("rz","0.0")))));
    }

    if (arg->getAttribute("dx") != NULL || arg->getAttribute("dy") != NULL || arg->getAttribute("dz") != NULL)
    {
        translation.setValue(Vector3((Real)atof(arg->getAttribute("dx","0.0")), (Real)atof(arg->getAttribute("dy","0.0")), (Real)atof(arg->getAttribute("dz","0.0"))));
    }

    if (arg->getAttribute("rx2") != NULL || arg->getAttribute("ry2") != NULL || arg->getAttribute("rz2") != NULL)
    {
        rotation2.setValue(Vector3((SReal)(atof(arg->getAttribute("rx2","0.0"))),(SReal)(atof(arg->getAttribute("ry2","0.0"))),(SReal)(atof(arg->getAttribute("rz2","0.0")))));
    }

    if (arg->getAttribute("dx2") != NULL || arg->getAttribute("dy2") != NULL || arg->getAttribute("dz2") != NULL)
    {
        translation2.setValue(Vector3((Real)atof(arg->getAttribute("dx2","0.0")), (Real)atof(arg->getAttribute("dy2","0.0")), (Real)atof(arg->getAttribute("dz2","0.0"))));
    }
}

template <class DataTypes>
MechanicalObject<DataTypes>::~MechanicalObject()
{
    if (reset_position != NULL)
        delete reset_position;

    if (v0 != NULL)
        delete v0;

    if (m_gnuplotFileV != NULL)
        delete m_gnuplotFileV;

    if (m_gnuplotFileX != NULL)
        delete m_gnuplotFileX;
}


template <class DataTypes>
void MechanicalObject<DataTypes>::handleStateChange()
{
    using sofa::core::topology::TopologyChange;

    std::list< const TopologyChange * >::const_iterator itBegin = m_topology->firstStateChange();
    std::list< const TopologyChange * >::const_iterator itEnd = m_topology->lastStateChange();

    while( itBegin != itEnd )
    {
        TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch( changeType )
        {
        case core::topology::POINTSADDED:
        {
            using sofa::helper::vector;

            unsigned int nbPoints = ( static_cast< const PointsAdded * >( *itBegin ) )->getNbAddedVertices();
            vector< vector< unsigned int > > ancestors = ( static_cast< const PointsAdded * >( *itBegin ) )->ancestorsList;
            vector< vector< double       > > coefs     = ( static_cast< const PointsAdded * >( *itBegin ) )->coefs;

            if (!ancestors.empty() )
            {
                unsigned int prevSizeMechObj = getSize();
                resize(prevSizeMechObj + nbPoints);

                vector< vector< double > > coefs2;
                coefs2.resize(ancestors.size());

                for (unsigned int i = 0; i < ancestors.size(); ++i)
                {
                    coefs2[i].resize(ancestors[i].size());

                    for (unsigned int j = 0; j < ancestors[i].size(); ++j)
                    {
                        // constructng default coefs if none were defined
                        if (coefs == (const vector< vector< double > >)0 || coefs[i].size() == 0)
                            coefs2[i][j] = 1.0f / ancestors[i].size();
                        else
                            coefs2[i][j] = coefs[i][j];
                    }
                }

                for (unsigned int i = 0; i < ancestors.size(); ++i)
                {
                    computeWeightedValue( prevSizeMechObj + i, ancestors[i], coefs2[i] );
                }
            }
            else
            {
                // No ancestors specified, resize DOFs vectors and set new values to the reset default value.
                resize(getSize() + nbPoints);
            }
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const PointsRemoved * >( *itBegin ) )->getArray();

            unsigned int prevSizeMechObj   = getSize();
            unsigned int lastIndexMech = prevSizeMechObj - 1;
            for (unsigned int i = 0; i < tab.size(); ++i)
            {
                replaceValue(lastIndexMech, tab[i] );

                --lastIndexMech;
            }
            resize( prevSizeMechObj - tab.size() );
            break;
        }
        case core::topology::POINTSMOVED:
        {
            using sofa::helper::vector;

            const vector< unsigned int > indicesList = ( static_cast <const PointsMoved *> (*itBegin))->indicesList;
            const vector< vector< unsigned int > > ancestors = ( static_cast< const PointsMoved * >( *itBegin ) )->ancestorsList;
            const vector< vector< double > > coefs = ( static_cast< const PointsMoved * >( *itBegin ) )->baryCoefsList;

            if (ancestors.size() != indicesList.size() || ancestors.empty())
            {
                this->serr << "Error ! MechanicalObject::POINTSMOVED topological event, bad inputs (inputs don't share the same size or are empty)."<<this->sendl;
                break;
            }

            vector< vector < double > > coefs2;
            coefs2.resize (coefs.size());

            for (unsigned int i = 0; i<ancestors.size(); ++i)
            {
                coefs2[i].resize(ancestors[i].size());

                for (unsigned int j = 0; j < ancestors[i].size(); ++j)
                {
                    // constructng default coefs if none were defined
                    if (coefs == (const vector< vector< double > >)0 || coefs[i].size() == 0)
                        coefs2[i][j] = 1.0f / ancestors[i].size();
                    else
                        coefs2[i][j] = coefs[i][j];
                }
            }

            for (unsigned int i = 0; i < indicesList.size(); ++i)
            {
                computeWeightedValue( indicesList[i], ancestors[i], coefs2[i] );
            }

            break;
        }
        case core::topology::POINTSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();

            renumberValues( tab );
            break;
        }
        default:
            // Ignore events that are not Quad  related.
            break;
        };

        ++itBegin;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::replaceValue (const int inputIndex, const int outputIndex)
{
    const unsigned int maxIndex = std::max(inputIndex, outputIndex);

    // standard state vectors
    // Note that the x,v,x0,f,dx,xfree,vfree and internalForces vectors (but
    // not v0, reset_position, and externalForces) are present in the
    // array of all vectors, so then don't need to be processed separatly.
    //(*x) [outputIndex] = (*x) [inputIndex];
    //if((*x0).size() > maxIndex)
    //    (*x0)[outputIndex] = (*x0)[inputIndex];
    //(*v) [outputIndex] = (*v) [inputIndex];
    if (v0 != NULL && (*v0).size() > maxIndex)
        (*v0)[outputIndex] = (*v0)[inputIndex];
    //if ((*f).size()>0)
    //    (*f) [outputIndex] = (*f) [inputIndex];
    //if ((*dx).size()>0)
    //    (*dx)[outputIndex] = (*dx)[inputIndex];
    // forces
    //if ((*internalForces).size()>0)
    //    (*internalForces)[outputIndex] = (*internalForces)[inputIndex];

    // PJB Modif : externalForces is in vectorsDeriv.
    /*if (externalForces.getValue().size() > maxIndex)
    {
    	(*(externalForces.beginEdit()))[outputIndex] = externalForces.getValue()[inputIndex];
    	externalForces.endEdit();
    }*/

    // Note: the following assumes that topological changes won't be reset
    if (reset_position != NULL && (*reset_position).size() > maxIndex)
        (*reset_position)[outputIndex] = (*reset_position)[inputIndex];

    // temporary state vectors
    const unsigned int vecCoordSize = vectorsCoord.size();
    for (unsigned int i = 0; i < vecCoordSize; i++)
    {
        if (vectorsCoord[i] != NULL)
        {
            VecCoord& vector = *(vectorsCoord[i]->beginEdit());

            if (vector.size() > maxIndex)
                vector[outputIndex] = vector[inputIndex];

            vectorsCoord[i]->endEdit();
        }
    }

    const unsigned int vecDerivSize = vectorsDeriv.size();
    for (unsigned int i = 0; i < vecDerivSize; i++)
    {
        if (vectorsDeriv[i] != NULL)
        {
            VecDeriv& vector = *(vectorsDeriv[i]->beginEdit());

            if (vector.size() > maxIndex)
                vector[outputIndex] = vector[inputIndex];

            vectorsDeriv[i]->endEdit();
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::swapValues (const int idx1, const int idx2)
{
    const unsigned int maxIndex = std::max(idx1, idx2);

    // standard state vectors
    // Note that the x,v,x0,f,dx,xfree,vfree and internalForces vectors (but
    // not v0, reset_position, and externalForces) are present in the
    // array of all vectors, so then don't need to be processed separatly.
    Coord tmp;
    Deriv tmp2;
    //tmp = (*x)[idx1];
    //(*x) [idx1] = (*x) [idx2];
    //(*x) [idx2] = tmp;

    //if((*x0).size() > maxIndex)
    //{
    //	tmp = (*x0)[idx1];
    //	(*x0)[idx1] = (*x0)[idx2];
    //	(*x0)[idx2] = tmp;
    //}
    //tmp2 = (*v)[idx1];
    //(*v) [idx1] = (*v) [idx2];
    //(*v) [idx2] = tmp2;

    if(v0 != NULL && (*v0).size() > maxIndex)
    {
        tmp2 = (*v0) [idx1];
        (*v0)[idx1] = (*v0)[idx2];
        (*v0)[idx2] = tmp2;
    }
    //tmp2 = (*f) [idx1];
    //(*f) [idx1] = (*f)[idx2];
    //(*f) [idx2] = tmp2;

    //tmp2 = (*dx) [idx1];
    //(*dx)[idx1] = (*dx)[idx2];
    //(*dx)[idx2] = tmp2;

    // forces
    //tmp2 = (*internalForces)[idx1];
    //(*internalForces)[idx1] = (*internalForces)[idx2];
    //(*internalForces)[idx2] = tmp2;

    // PJB Modif : externalForces is in vectorsDeriv.
    /*if (externalForces.getValue().size() > maxIndex)
    {
    	VecDeriv *extForcesEdit = externalForces.beginEdit();

    	tmp2 = (*extForcesEdit)[idx1];
    	(*extForcesEdit)[idx1] = (*extForcesEdit)[idx2];
    	(*extForcesEdit)[idx2] = tmp2;

    	externalForces.endEdit();
    }*/

    // Note: the following assumes that topological changes won't be reset
    if (reset_position != NULL && (*reset_position).size() > maxIndex)
    {
        tmp = (*reset_position)[idx1];
        (*reset_position)[idx1] = (*reset_position)[idx2];
        (*reset_position)[idx2] = tmp;
    }

    // temporary state vectors
    unsigned int i;
    for (i=0; i<vectorsCoord.size(); i++)
    {
        if(vectorsCoord[i] != NULL)
        {
            VecCoord& vector = *vectorsCoord[i]->beginEdit();
            if(vector.size() > maxIndex)
            {
                tmp = vector[idx1];
                vector[idx1] = vector[idx2];
                vector[idx2] = tmp;
            }
            vectorsCoord[i]->endEdit();
        }
    }
    for (i=0; i<vectorsDeriv.size(); i++)
    {
        if(vectorsDeriv[i] != NULL)
        {
            VecDeriv& vector = *vectorsDeriv[i]->beginEdit();
            if(vector.size() > maxIndex)
            {
                tmp2 = vector[idx1];
                vector[idx1] = vector[idx2];
                vector[idx2] = tmp2;
            }
            vectorsDeriv[i]->endEdit();
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::renumberValues( const sofa::helper::vector< unsigned int > &index )
{
    VecDeriv dtmp;
    VecCoord ctmp;
    // standard state vectors
    // Note that the x,v,x0,f,dx,xfree,vfree and internalForces vectors (but
    // not v0, reset_position, and externalForces) are present in the
    // array of all vectors, so then don't need to be processed separatly.
    //renumber(x, &ctmp, index);
    //renumber(x0, &ctmp, index);
    //renumber(v, &dtmp, index);
    renumber(v0, &dtmp, index);
    //renumber(f, &dtmp, index);
    //renumber(dx, &dtmp, index);
    //renumber(internalForces, &dtmp, index);

    // PJB Modif : externalForces is now in vectorsDeriv.
    //renumber(externalForces.beginEdit(), &dtmp, index);
    //externalForces.endEdit();

    // Note: the following assumes that topological changes won't be reset
    renumber(reset_position, &ctmp, index);

    for (unsigned int i = 0; i < vectorsCoord.size(); ++i)
    {
        if (vectorsCoord[i] != NULL)
        {
            renumber(vectorsCoord[i]->beginEdit(), &ctmp, index);
            vectorsCoord[i]->endEdit();
        }
    }

    for (unsigned int i = 0; i < vectorsDeriv.size(); ++i)
    {
        if (vectorsDeriv[i] != NULL)
        {
            renumber(vectorsDeriv[i]->beginEdit(), &dtmp, index);
            vectorsDeriv[i]->endEdit();
        }
    }
}



template <class DataTypes>
void MechanicalObject<DataTypes>::resize(const int size)
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif

    x.beginEdit()->resize(size);
    x.endEdit();

    if (m_initialized && !(x0.getValue().empty()))
    {
        x0.beginEdit()->resize(size);
        x0.endEdit();
    }

    v.beginEdit()->resize(size);
    v.endEdit();

    if (v0 != NULL)
        (*v0).resize(size);

    f.beginEdit()->resize(size);
    f.endEdit();

    dx.beginEdit()->resize(size);
    dx.endEdit();

    xfree.beginEdit()->resize(size);
    xfree.endEdit();

    vfree.beginEdit()->resize(size);
    vfree.endEdit();

    if (externalForces.getValue().size() > 0)
    {
        externalForces.beginEdit()->resize(size);
        externalForces.endEdit();
    }

    internalForces.beginEdit()->resize(size);
    internalForces.endEdit();

    // Note: the following assumes that topological changes won't be reset
    if (reset_position != NULL)
        (*reset_position).resize(size);

    //if (size!=vsize)
    {
        vsize = size;
        for (unsigned int i = 0; i < vectorsCoord.size(); i++)
        {
            if (vectorsCoord[i] != NULL && vectorsCoord[i]->getValue().size() != 0)
            {
                vectorsCoord[i]->beginEdit()->resize(size);
                vectorsCoord[i]->endEdit();

#ifdef SOFA_SMP
                vectorsCoordSharedAllocated[i] = true;
#endif
            }
        }

        for (unsigned int i = 0; i < vectorsDeriv.size(); i++)
        {
            if (vectorsDeriv[i] != NULL && vectorsDeriv[i]->getValue().size() != 0)
            {
                vectorsDeriv[i]->beginEdit()->resize(size);
                vectorsDeriv[i]->endEdit();
#ifdef SOFA_SMP
                vectorsDerivSharedAllocated[i] = true;
#endif
            }
        }
    }
}


template <class DataTypes>
void MechanicalObject<DataTypes>::reserve(const int size)
{
    if (size == 0) return;

    x.beginEdit()->reserve(size);
    x.endEdit();

    if (m_initialized && !x0.getValue().empty())
    {
        x0.beginEdit()->reserve(size);
        x0.endEdit();
    }

    v.beginEdit()->reserve(size);
    v.endEdit();

    if (v0 != NULL)
        (*v0).reserve(size);

    f.beginEdit()->reserve(size);
    f.endEdit();

    dx.beginEdit()->reserve(size);
    dx.endEdit();

    xfree.beginEdit()->reserve(size);
    xfree.endEdit();

    vfree.beginEdit()->reserve(size);
    vfree.endEdit();

    externalForces.beginEdit()->reserve(size);
    externalForces.endEdit();

    internalForces.beginEdit()->reserve(size);
    internalForces.endEdit();

    if (reset_position != NULL)
        (*reset_position).reserve(size);

    for (unsigned int i = 0; i < vectorsCoord.size(); i++)
    {
        if (vectorsCoord[i] != NULL)
        {
            vectorsCoord[i]->beginEdit()->reserve(size);
            vectorsCoord[i]->endEdit();
        }
    }

    for (unsigned int i = 0; i < vectorsDeriv.size(); i++)
    {
        if (vectorsDeriv[i] != NULL)
        {
            vectorsDeriv[i]->beginEdit()->reserve(size);
            vectorsDeriv[i]->endEdit();
        }
    }
}


template <class DataTypes>
void MechanicalObject<DataTypes>::applyTranslation (const double dx,const double dy,const double dz)
{
    VecCoord& x = *this->getX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        DataTypes::add(x[i], dx, dy, dz);
    }

}


//Apply Rotation from Euler angles (in degree!)
template <class DataTypes>
void MechanicalObject<DataTypes>::applyRotation (const double rx, const double ry, const double rz)
{
    Quaternion q = helper::Quater< SReal >::createQuaterFromEuler(Vec< 3, SReal >(rx, ry, rz) * M_PI / 180.0);
    applyRotation(q);
}


template <class DataTypes>
void MechanicalObject<DataTypes>::applyRotation (const defaulttype::Quat q)
{
    VecCoord& x = *this->getX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        Vec<3,Real> pos;
        DataTypes::get(pos[0],pos[1],pos[2],x[i]);
        Vec<3,Real> newposition = q.rotate(pos);
        DataTypes::set(x[i],newposition[0],newposition[1],newposition[2]);
    }
}

#ifndef SOFA_FLOAT
template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::applyRotation (const defaulttype::Quat q);
/*    template <>
bool MechanicalObject<Vec1dTypes>::addBBox(double* minBBox, double* maxBBox)*/;
#endif
#ifndef SOFA_DOUBLE
template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::applyRotation (const defaulttype::Quat q);
//     template <>
// 	bool MechanicalObject<Vec1fTypes>::addBBox(double* minBBox, double* maxBBox);
#endif

template <class DataTypes>
void MechanicalObject<DataTypes>::applyScale(const double sx,const double sy,const double sz)
{
    //       std::cout << "MechanicalObject : applyScale " << this->getName() << " s=" << s << "\n";
    VecCoord& x = *this->getX();
    const Vector3 s(sx,sy,sz);
    for (unsigned int i=0; i<x.size(); i++)
    {
        x[i][0] = x[i][0]*sx;
        x[i][1] = x[i][1]*sy;
        x[i][2] = x[i][2]*sz;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::getIndicesInSpace(sofa::helper::vector<unsigned>& indices, Real xmin, Real xmax, Real ymin, Real ymax, Real zmin, Real zmax) const
{
    //const VecCoord& X = *getX();
    helper::ReadAccessor<VecCoord> X = *getX();
    for( unsigned i=0; i<X.size(); ++i )
    {
        Real x=0.0,y=0.0,z=0.0;
        DataTypes::get(x,y,z,X[i]);
        if( x >= xmin && x <= xmax && y >= ymin && y <= ymax && z >= zmin && z <= zmax )
        {
            indices.push_back(i);
        }
    }
}

/*
template <class DataTypes>
void MechanicalObject<DataTypes>::computeWeightedValue( const unsigned int i, const sofa::helper::vector< unsigned int >& ancestors, const sofa::helper::vector< double >& coefs)
{
// HD interpolate position, speed,force,...
// assume all coef sum to 1.0
unsigned int j;

// Note that the x,v,x0,f,dx,xfree,vfree and internalForces vectors (but
// not v0, reset_position, and externalForces) are present in the
// array of all vectors, so then don't need to be processed separatly.
if (v0 != NULL)
{
(*v0)[i] = Deriv();
for (j = 0; j < ancestors.size(); ++j)
{
(*v0)[i] += (*v0)[ancestors[j]] * (Real)coefs[j];
}
}

// Note: the following assumes that topological changes won't be reset
if (reset_position != NULL)
{
(*reset_position)[i] = Coord();
for (j = 0; j < ancestors.size(); ++j)
{
(*reset_position)[i] += (*reset_position)[ancestors[j]] * (Real)coefs[j];
}
}

if (externalForces->size() > 0)
{
(*externalForces)[i] = Deriv();
for (j = 0; j < ancestors.size(); ++j)
{
(*externalForces)[i] += (*externalForces)[ancestors[j]] * (Real)coefs[j];
}
}


for (unsigned int k = 0; k < vectorsCoord.size(); k++)
{
if (vectorsCoord[k]!=NULL && vectorsCoord[k]->size()!=0)
{
(*vectorsCoord[k])[i] = Coord();
for (j = 0; j < ancestors.size(); ++j)
{
(*vectorsCoord[k])[i] += (*vectorsCoord[k])[ancestors[j]] * (Real)coefs[j];
}
}
}

for (unsigned int k = 0; k < vectorsDeriv.size(); k++)
{
if (vectorsDeriv[k]!=NULL && vectorsDeriv[k]->size()!=0)
{
(*vectorsDeriv[k])[i] = Deriv();
for (j = 0; j < ancestors.size(); ++j)
{
(*vectorsDeriv[k])[i] += (*vectorsDeriv[k])[ancestors[j]] * (Real)coefs[j];
}
}
}
}
*/


template <class DataTypes>
void MechanicalObject<DataTypes>::computeWeightedValue( const unsigned int i, const sofa::helper::vector< unsigned int >& ancestors, const sofa::helper::vector< double >& coefs)
{
    // HD interpolate position, speed,force,...
    // assume all coef sum to 1.0
    unsigned int j;

    const unsigned int ancestorsSize = ancestors.size();

    helper::vector< Coord > ancestorsCoord(ancestorsSize);
    helper::vector< Deriv > ancestorsDeriv(ancestorsSize);
    helper::vector< Real > ancestorsCoefs(ancestorsSize);


    // Note that the x,v,x0,f,dx,xfree,vfree and internalForces vectors (but
    // not v0, reset_position, and externalForces) are present in the
    // array of all vectors, so then don't need to be processed separatly.
    if (v0 != NULL)
    {
        for (j = 0; j < ancestorsSize; ++j)
        {
            ancestorsDeriv[j] = (*v0)[ancestors[j]];
            ancestorsCoefs[j] = coefs[j];
        }

        (*v0)[i] = DataTypes::interpolate(ancestorsDeriv, ancestorsCoefs);
    }

    // Note: the following assumes that topological changes won't be reset
    if (reset_position != NULL)
    {
        for (j = 0; j < ancestorsSize; ++j)
        {
            ancestorsCoord[j] = (*reset_position)[ancestors[j]];
            ancestorsCoefs[j] = coefs[j];
        }

        (*reset_position)[i] = DataTypes::interpolate(ancestorsCoord, ancestorsCoefs);
    }

    // PJB Modif : externalForces is now in vectorsDeriv.
    /*if (externalForces.getValue().size() > 0)
    {
    	VecDeriv *extForcesEdit = externalForces.beginEdit();

    	for (j = 0; j < ancestorsSize; ++j)
    	{
    		ancestorsDeriv[j] = (*extForcesEdit)[ancestors[j]];
    		ancestorsCoefs[j] = coefs[j];
    	}

    	(*extForcesEdit)[i] = DataTypes::interpolate(ancestorsDeriv, ancestorsCoefs);

    	externalForces.endEdit();
    }*/


    for (unsigned int k = 0; k < vectorsCoord.size(); k++)
    {
        if (vectorsCoord[k] != NULL)
        {
            VecCoord &vecCoord = *(vectorsCoord[k]->beginEdit());

            if (vecCoord.size() != 0)
            {
                for (j = 0; j < ancestorsSize; ++j)
                {
                    ancestorsCoord[j] = vecCoord[ancestors[j]];
                    ancestorsCoefs[j] = coefs[j];
                }

                vecCoord[i] = DataTypes::interpolate(ancestorsCoord, ancestorsCoefs);
            }

            vectorsCoord[k]->endEdit();
        }
    }

    for (unsigned int k = 0; k < vectorsDeriv.size(); k++)
    {
        if (vectorsDeriv[k] != NULL)
        {
            VecDeriv &vecDeriv = *(vectorsDeriv[k]->beginEdit());

            if (vecDeriv.size() != 0)
            {
                for (j = 0; j < ancestorsSize; ++j)
                {
                    ancestorsDeriv[j] = vecDeriv[ancestors[j]];
                    ancestorsCoefs[j] = coefs[j];
                }

                vecDeriv[i] = DataTypes::interpolate(ancestorsDeriv, ancestorsCoefs);
            }

            vectorsDeriv[k]->endEdit();
        }
    }
}


//	template <class DataTypes>
//    void MechanicalObject<DataTypes>::computeNewPoint( const unsigned int i, const sofa::helper::vector< double >& m_x)
//	{
//                  this->resize(i+1);
//		  Vec<3,Real> pos(m_x[0], m_x[1], m_x[2]);
//		  Vec<3,Real> restpos(pos);
//
//		  Quaternion q=helper::Quater<SReal>::createQuaterFromEuler( Vec<3,SReal>(rotation.getValue()[0],rotation.getValue()[1],rotation.getValue()[2]));
//		  pos = q.rotate(pos*scale.getValue());
//		  pos += translation.getValue();
//
//		  restpos = q.rotate(restpos*restScale.getValue());
//		  restpos += translation.getValue();
//
//		  DataTypes::set((*getX())[i], pos[0], pos[1], pos[2]);
//		  DataTypes::set((*getXfree())[i], pos[0], pos[1], pos[2]);
//		  DataTypes::set((*getX0())[i], restpos[0],restpos[1],restpos[2]);
//
//		  if (reset_position != NULL)
//		    DataTypes::set((*reset_position)[i], pos[0], pos[1], pos[2]);
//	}

// Force the position of a point (and force its velocity to zero value)
template <class DataTypes>
void MechanicalObject<DataTypes>::forcePointPosition(const unsigned int i, const sofa::helper::vector< double >& m_x)
{
    DataTypes::set((*getX())[i], m_x[0], m_x[1], m_x[2]);
    DataTypes::set((*getV())[i], (Real) 0.0, (Real) 0.0, (Real) 0.0);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::loadInVector(defaulttype::BaseVector * dest, VecId src, unsigned int offset)
{
    if (src.type == VecId::V_COORD)
    {
        helper::ReadAccessor<VecCoord> vSrc = *getVecCoord(src.index);

        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = dest->size()/coordDim;

        for (unsigned int i = 0; i < nbEntries; ++i)
            for (unsigned int j = 0; j < coordDim; ++j)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vSrc[offset + i], j, tmp);
                dest->set(i * coordDim + j, tmp);
            }
        // offset += vSrc.size() * coordDim;
    }
    else
    {
        helper::ReadAccessor<VecDeriv> vSrc = *getVecDeriv(src.index);

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = dest->size()/derivDim;

        for (unsigned int i = 0; i < nbEntries; i++)
            for (unsigned int j = 0; j < derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vSrc[i + offset], j, tmp);
                dest->set(i * derivDim + j, tmp);
            }
        // offset += vSrc.size() * derivDim;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::loadInBaseVector(defaulttype::BaseVector * dest, VecId src, unsigned int &offset)
{
    if (src.type == VecId::V_COORD)
    {
        //const VecCoord* vSrc = getVecCoord(src.index);
        helper::ReadAccessor<VecCoord> vSrc = *getVecCoord(src.index);
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i = 0; i < vSrc.size(); i++)
        {
            for (unsigned int j = 0; j < coordDim; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vSrc[i], j, tmp);
                dest->set(offset + i * coordDim + j, tmp);
            }
        }

        offset += vSrc.size() * coordDim;
    }
    else
    {
        //const VecDeriv* vSrc = getVecDeriv(src.index);
        helper::ReadAccessor<VecDeriv> vSrc = *getVecDeriv(src.index);
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();

        for (unsigned int i = 0; i < vSrc.size(); i++)
        {
            for (unsigned int j = 0; j < derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vSrc[i], j, tmp);
                dest->set(offset + i * derivDim + j, tmp);
            }
        }

        offset += vSrc.size() * derivDim;
    }
}

#ifndef SOFA_FLOAT
template <>
void MechanicalObject<defaulttype::Rigid3dTypes>::addBaseVectorToState(VecId dest, defaulttype::BaseVector *src, unsigned int &offset);
#endif
#ifndef SOFA_DOUBLE
template <>
void MechanicalObject<defaulttype::Rigid3fTypes>::addBaseVectorToState(VecId dest, defaulttype::BaseVector *src, unsigned int &offset);
#endif

template <class DataTypes>
void MechanicalObject<DataTypes>::addBaseVectorToState(VecId dest, defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == VecId::V_COORD)
    {
        //VecCoord* vDest = getVecCoord(dest.index);
        helper::WriteAccessor<VecCoord> vDest = *getVecCoord(dest.index);
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i = 0; i < vDest.size(); i++)
        {
            for (unsigned int j = 0; j < coordDim; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i], j, tmp);
                DataTypeInfo<Coord>::setValue(vDest[i], j, tmp + src->element(offset + i * coordDim + j));
            }
        }

        offset += vDest.size() * coordDim;
    }
    else
    {
        //VecDeriv* vDest = getVecDeriv(dest.index);
        helper::WriteAccessor<VecDeriv> vDest = *getVecDeriv(dest.index);
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();

        for (unsigned int i = 0; i < vDest.size(); i++)
        {
            for (unsigned int j = 0; j < derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i], j, tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i], j, tmp + src->element(offset + i * derivDim + j));
            }
        }

        offset += vDest.size() * derivDim;
    }
}


#ifndef SOFA_FLOAT
template <>
void MechanicalObject<defaulttype::Rigid3dTypes>::addVectorToState(VecId dest, defaulttype::BaseVector *src, unsigned int &offset);
#endif
#ifndef SOFA_DOUBLE
template <>
void MechanicalObject<defaulttype::Rigid3fTypes>::addVectorToState(VecId dest, defaulttype::BaseVector *src, unsigned int &offset);
#endif

template <class DataTypes>
void MechanicalObject<DataTypes>::addVectorToState(VecId dest, defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == VecId::V_COORD)
    {
        helper::WriteAccessor<VecCoord> vDest = *getVecCoord(dest.index);
        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = src->size()/coordDim;
        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<coordDim; ++j)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i+offset],j, tmp + src->element(i*coordDim+j));
            }
        }
        offset += nbEntries;
    }
    else
    {
        helper::WriteAccessor<VecDeriv> vDest = *getVecDeriv(dest.index);

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = src->size()/derivDim;
        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<derivDim; ++j)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i+offset],j, tmp + src->element(i*derivDim+j));
            }
        }
        offset += nbEntries;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addDxToCollisionModel()
{
    helper::WriteAccessor< VecCoord > x_wa = *(x.beginEdit());
    helper::ReadAccessor< VecCoord > xfree_ra = xfree.getValue();
    helper::ReadAccessor< VecDeriv > dx_ra = dx.getValue();

    for (unsigned int i = 0; i < xfree_ra.size(); i++)
    {
        x_wa[i] = xfree_ra[i] + dx_ra[i];
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::init()
{
    //std::cout << "MechanicalObject::init " << this->getName() << std::endl;
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
    m_topology = this->getContext()->getMeshTopology();

    /*f_X->beginEdit();
    f_V->beginEdit();
    f_F->beginEdit();
    f_Dx->beginEdit();
    f_Xfree->beginEdit();
    f_Vfree->beginEdit();
    f_X0->beginEdit();*/

    //case if X0 has been set but not X
    if (getX0()->size() > getX()->size())
    {
        //*x = *x0;
        x.setValue(x0.getValue());
    }

    if (getX()->size() != (std::size_t)vsize || getV()->size() != (std::size_t)vsize)
    {
        // X and/or V where user-specified
        // copy the last specified velocity to all points
        if (getV()->size() >= 1 && getV()->size() < getX()->size())
        {
            unsigned int i = getV()->size();
            Deriv v1 = (*getV())[i-1];
            getV()->resize(getX()->size());
            while (i < getV()->size())
                (*getV())[i++] = v1;
        }
        resize(getX()->size()>getV()->size()?getX()->size():getV()->size());
    }
    else if (getX()->size() <= 1)
    {

        if (!ignoreLoader.getValue())
        {
            MeshLoader* m_loader;
            this->getContext()->get(m_loader);

            if(m_loader && m_loader->getFillMState())
            {

                int nbp = m_loader->getNbPoints();

                //std::cout<<"Setting "<<nbp<<" points from MeshLoader. " <<std::endl;

                // copy the last specified velocity to all points
                if (getV()->size() >= 1 && getV()->size() < (unsigned)nbp)
                {
                    unsigned int i = getV()->size();
                    Deriv v1 = (*getV())[i-1];
                    getV()->resize(nbp);
                    while (i < getV()->size())
                        (*getV())[i++] = v1;
                }
                this->resize(nbp);
                for (int i=0; i<nbp; i++)
                {
                    (*getX())[i] = Coord();
                    DataTypes::set((*getX())[i], m_loader->getPX(i), m_loader->getPY(i), m_loader->getPZ(i));
                }

            }
            else
            {

                if (m_topology != NULL && m_topology->hasPos() && m_topology->getContext() == this->getContext())
                {
                    int nbp = m_topology->getNbPoints();
                    //std::cout<<"Setting "<<nbp<<" points from topology. " << this->getName() << " topo : " << m_topology->getName() <<std::endl;
                    // copy the last specified velocity to all points
                    if (getV()->size() >= 1 && getV()->size() < (unsigned)nbp)
                    {
                        unsigned int i = getV()->size();
                        Deriv v1 = (*getV())[i-1];
                        getV()->resize(nbp);
                        while (i < getV()->size())
                            (*getV())[i++] = v1;
                    }
                    this->resize(nbp);
                    for (int i=0; i<nbp; i++)
                    {
                        (*getX())[i] = Coord();
                        DataTypes::set((*getX())[i], m_topology->getPX(i), m_topology->getPY(i), m_topology->getPZ(i));
                    }

                }
            }
        }
    }

    reinit();

    if (v0 == NULL)
    {
        this->v0 = new VecDeriv;
    }

//	*this->v0 = *v;
    *this->v0 = v.getValue();

    // free position = position
//	*this->xfree = *x;
    this->xfree.setValue(x.getValue());

    VecCoord *x0_edit = x0.beginEdit();

    //Rest position
    if (x0_edit->size() == 0)
    {
        //	*x0 = *x;
        x0.setValue(x.getValue());
        if (restScale.getValue() != (Real)1)
        {
            Real s = (Real)restScale.getValue();
            for (unsigned int i=0; i<x0_edit->size(); i++)
                (*x0_edit)[i] *= s;
        }
    }

    x0.endEdit();

    if (rotation2.getValue()[0]!=0.0 || rotation2.getValue()[1]!=0.0 || rotation2.getValue()[2]!=0.0)
    {
        this->applyRotation(rotation2.getValue()[0],rotation2.getValue()[1],rotation2.getValue()[2]);
    }

    if (translation2.getValue()[0]!=0.0 || translation2.getValue()[1]!=0.0 || translation2.getValue()[2]!=0.0)
    {
        this->applyTranslation( translation2.getValue()[0],translation2.getValue()[1],translation2.getValue()[2]);
    }

    m_initialized = true;

    if (f_reserve.getValue() > 0)
        reserve(f_reserve.getValue());

    /*f_X->endEdit();
    f_V->endEdit();
    f_F->endEdit();
    f_Dx->endEdit();
    f_Xfree->endEdit();
    f_Vfree->endEdit();
    f_X0->endEdit();*/
}


template <class DataTypes>
void MechanicalObject<DataTypes>::reinit()
{
    Vector3 p0;
    sofa::component::topology::RegularGridTopology *grid;
    this->getContext()->get(grid, BaseContext::Local);
    if (grid) p0 = grid->getP0();

    if (scale.getValue() != Vector3(1.0,1.0,1.0))
    {
        this->applyScale(scale.getValue()[0],scale.getValue()[1],scale.getValue()[2]);
        p0 = p0.linearProduct(scale.getValue());
    }

    if (rotation.getValue()[0]!=0.0 || rotation.getValue()[1]!=0.0 || rotation.getValue()[2]!=0.0)
    {
        this->applyRotation(rotation.getValue()[0],rotation.getValue()[1],rotation.getValue()[2]);

        if (grid)
        {
            this->serr << "Warning ! MechanicalObject initial rotation is not applied to its grid topology"<<this->sendl;
            this->serr << "Regular grid topologies rotations are unsupported."<<this->sendl;
            //  p0 = q.rotate(p0);
        }
    }

    if (translation.getValue()[0]!=0.0 || translation.getValue()[1]!=0.0 || translation.getValue()[2]!=0.0)
    {
        this->applyTranslation( translation.getValue()[0],translation.getValue()[1],translation.getValue()[2]);
        p0 += translation.getValue();
    }


    if (grid) grid->setP0(p0);

    /*
    translation.setValue(Vector3());
    rotation.setValue(Vector3());
    scale.setValue(Vector3(1.0,1.0,1.0));
    */
}
template <class DataTypes>
void MechanicalObject<DataTypes>::storeResetState()
{
    // Save initial state for reset button
    if (reset_position == NULL) this->reset_position = new VecCoord;
//	*this->reset_position = *x;
    *this->reset_position = x.getValue();
}

//
// Integration related methods
//

template <class DataTypes>
void MechanicalObject<DataTypes>::reset()
{
    if (reset_position == NULL)
        return;

    // Back to initial state
    this->resize(reset_position->size());
    //std::cout << this->getName() << ": reset X"<<std::endl;
    //*this->x = *reset_position;
    *this->getVecCoord(VecId::position().index) = *this->reset_position;
    //std::cout << this->getName() << ": reset V"<<std::endl;
    //*this->v = *v0;

    if (v0 == NULL)
    {
        VecDeriv *vEdit = this->v.beginEdit();

        for( unsigned int i = 0; i < vEdit->size(); ++i )
            (*vEdit)[i] = Deriv();

        v.endEdit();

        return;
    }

    *this->getVecDeriv(VecId::velocity().index) = *this->v0;

    //std::cout << this->getName() << ": reset Xfree"<<std::endl;
    //*this->xfree = *x;

    *this->getVecCoord(VecId::freePosition().index) = *this->getVecCoord(VecId::position().index);
    //std::cout << this->getName() << ": reset Vfree"<<std::endl;
    //*this->vfree = *v;
    *this->getVecDeriv(VecId::freeVelocity().index) = *this->getVecDeriv(VecId::velocity().index);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::writeX(std::ostream &out)
{
    out << *getX();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::readX(std::istream &in)
{
    //in >> *getX(); //Modified to handle a modification of the number of Dofs. Problem to resolve: how to modify the containers to handle this...
    Coord pos;
    int i = 0;

    VecCoord *xEdit = x.beginEdit();

    while (in >> pos)
    {
        if (i >= getSize())
            resize(i+1);

        (*xEdit)[i++] = pos;
    }

    x.endEdit();

    if (i < getSize())
        resize(i);
}

template <class DataTypes>
double MechanicalObject<DataTypes>::compareX(std::istream &in)
{
    std::string ref,cur;
    getline(in, ref);

    std::ostringstream out;
    out << *getX();
    cur = out.str();

    double error=0;
    std::istringstream compareX_ref(ref);
    std::istringstream compareX_cur(cur);

    Real value_ref, value_cur;
    unsigned int count=0;
    while (compareX_ref >> value_ref && compareX_cur >> value_cur )
    {
        // /* if ( fabs(value_ref-value_cur) != 0) */std::cout << " Eroor ! " << fabs(value_ref-value_cur) << " for " << this->getName() << "at time: " << this->getContext()->getTime() << " between " << value_ref << " && " << value_cur << "\n";
        error += fabs(value_ref-value_cur);
        count ++;
    }
    return error/count;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::writeV(std::ostream &out)
{
    out << *getV();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::readV(std::istream &in)
{
    Deriv vel;
    int i = 0;

    VecDeriv *vEdit = v.beginEdit();

    while (in >> vel)
    {
        if (i >= getSize())
            resize(i+1);

        (*vEdit)[i++] = vel;
    }

    v.endEdit();

    if (i < getSize())
        resize(i);
}

template <class DataTypes>
double MechanicalObject<DataTypes>::compareV(std::istream &in)
{
    std::string ref,cur;
    getline(in, ref);

    std::ostringstream out;
    out << *getV();
    cur = out.str();

    double error=0;
    std::istringstream compareV_ref(ref);
    std::istringstream compareV_cur(cur);

    Real value_ref, value_cur;
    unsigned int count=0;
    while (compareV_ref >> value_ref && compareV_cur >> value_cur )
    {
        error += fabs(value_ref-value_cur);
        count ++;
    }
    return error/count;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::writeF(std::ostream &out)
{
    out << *getF();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::writeDx(std::ostream &out)
{
    out << *getDx();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::writeState(std::ostream& out)
{
    writeX(out); out << " "; writeV(out);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::beginIntegration(Real /*dt*/)
{
    m_forceId = VecId::internalForce();
    this->forceMask.activate(false);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::endIntegration(Real /*dt*/)
{
    m_forceId = VecId::externalForce();
    getVecDeriv(VecId::externalForce().index )->clear();

    this->forceMask.clear();
    //By default the mask is disabled, the user has to enable it to benefit from the speedup
    this->forceMask.setInUse(this->useMask.getValue());
#ifdef SOFA_SMP
    BaseObject::Task<vClear<VecDeriv,Deriv> >  (this,**this->externalForces.beginEdit(),0);
    this->externalForces.endEdit();
#else
    this->externalForces.beginEdit()->clear();
    this->externalForces.endEdit();
#endif
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accumulateForce()
{
#ifdef SOFA_SMP
    BaseObject::Task < vPEq2 <  VecDeriv,
               VecDeriv >
               >(this,**this->f.beginEdit(),
                       **getVecDeriv(VecId::externalForce().index));
    this->f.endEdit();
#else
    if (!getVecDeriv(VecId::externalForce().index)->empty())
    {
        helper::WriteAccessor< VecDeriv > f = *getF();
        helper::ReadAccessor< VecDeriv > externalForces = *getVecDeriv(VecId::externalForce().index);

        if (!this->forceMask.isInUse())
        {
            for (unsigned int i=0; i < externalForces.size(); i++)
                f[i] += externalForces[i];
        }
        else
        {
            typedef helper::ParticleMask ParticleMask;
            const ParticleMask::InternalStorage &indices = this->forceMask.getEntries();
            ParticleMask::InternalStorage::const_iterator it;
            for (it = indices.begin(); it != indices.end(); it++)
            {
                const int i = (*it);
                f[i] += externalForces[i];
            }
        }
    }
#endif
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecCoord(unsigned int index, Data< VecCoord > *v)
{
    if (index >= vectorsCoord.size())
    {
        vectorsCoord.resize(index + 1, 0);
#ifdef SOFA_SMP
        vectorsCoordSharedAllocated.resize(index + 1);
#endif

    }

    vectorsCoord[index] = v;
#ifdef SOFA_SMP
    vectorsCoordSharedAllocated[index] = true;
#endif
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecDeriv(unsigned int index, Data< VecDeriv > *v)
{
    if (index >= vectorsDeriv.size())
    {
        vectorsDeriv.resize(index + 1, 0);
#ifdef SOFA_SMP
        vectorsDerivSharedAllocated.resize(index + 1);
#endif

    }

    vectorsDeriv[index] = v;
#ifdef SOFA_SMP
    vectorsDerivSharedAllocated[index] = true;
#endif

}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecConst(unsigned int index, Data < VecConst > *v)
{
    if (index >= vectorsConst.size())
    {
        vectorsConst.resize(index + 1, 0);
    }

    vectorsConst[index] = v;
}


template<class DataTypes>
typename MechanicalObject<DataTypes>::VecCoord* MechanicalObject<DataTypes>::getVecCoord(unsigned int index)
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
    if (index >= vectorsCoord.size())
    {
        vectorsCoord.resize(index + 1, 0);
#ifdef SOFA_SMP
        vectorsCoordSharedAllocated.resize(index + 1);
#endif
    }

    if (vectorsCoord[index] == NULL)
    {
        vectorsCoord[index] = new Data< VecCoord >;
#ifdef SOFA_SMP
        vectorsCoordSharedAllocated[index]=true;
#endif
        if (f_reserve.getValue() > 0)
        {
            vectorsCoord[index]->beginEdit()->reserve(f_reserve.getValue());
            vectorsCoord[index]->endEdit();
        }
    }

    return vectorsCoord[index]->beginEdit();
    // @TODO endEdit has to be called.
}

template<class DataTypes>
const typename MechanicalObject<DataTypes>::VecCoord* MechanicalObject<DataTypes>::getVecCoord(unsigned int index) const
{
    if (index >= vectorsCoord.size())
        return NULL;

    if (vectorsCoord[index] == NULL)
        return NULL;

    return &(vectorsCoord[index]->getValue());
}

template<class DataTypes>
typename MechanicalObject<DataTypes>::VecDeriv* MechanicalObject<DataTypes>::getVecDeriv(unsigned int index)
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
    if (index >= vectorsDeriv.size())
    {
        vectorsDeriv.resize(index + 1, 0);
#ifdef SOFA_SMP
        vectorsDerivSharedAllocated.resize(index + 1);
#endif
    }

    if (vectorsDeriv[index] == NULL)
    {
        vectorsDeriv[index] = new Data< VecDeriv >;
#ifdef SOFA_SMP
        vectorsDerivSharedAllocated[index] = true;
#endif
        if (f_reserve.getValue() > 0)
        {
            vectorsDeriv[index]->beginEdit()->reserve(f_reserve.getValue());
            vectorsDeriv[index]->endEdit();
        }
    }

    return vectorsDeriv[index]->beginEdit();
    // @TODO endEdit has to be called.
}

template<class DataTypes>
const typename MechanicalObject<DataTypes>::VecDeriv* MechanicalObject<DataTypes>::getVecDeriv(unsigned int index) const
{
    if (index >= vectorsDeriv.size())
        return NULL;

    if (vectorsDeriv[index] == NULL)
        return NULL;

    return &(vectorsDeriv[index]->getValue());
}

template<class DataTypes>
typename MechanicalObject<DataTypes>::VecConst* MechanicalObject<DataTypes>::getVecConst(unsigned int index)
{
    if (index >= vectorsConst.size())
        vectorsConst.resize(index + 1, 0);

    if (vectorsConst[index] == NULL)
        vectorsConst[index] = new Data< VecConst >;

    return vectorsConst[index]->beginEdit();
}

template<class DataTypes>
const typename MechanicalObject<DataTypes>::VecConst* MechanicalObject<DataTypes>::getVecConst(unsigned int index) const
{
    if (index >= vectorsConst.size())
        return NULL;

    if (vectorsConst[index] == NULL)
        return NULL;

    return &(vectorsConst[index]->getValue());
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAvail(VecId& v)
{
    if (v.type == VecId::V_COORD)
    {
        for (unsigned int i=v.index; i < vectorsCoord.size(); ++i)
#ifdef SOFA_SMP
            if (vectorsCoord[i] &&   vectorsCoordSharedAllocated[i])
#else
            if ((vectorsCoord[i]) && !vectorsCoord[i]->getValue().empty())
#endif
                v.index = i+1;
    }
    else if (v.type == VecId::V_DERIV)
    {
        for (unsigned int i=v.index; i < vectorsDeriv.size(); ++i)
#ifdef SOFA_SMP
            if (vectorsDeriv[i] != NULL &&vectorsDerivSharedAllocated[i])
#else
            if ((vectorsDeriv[i] != NULL) && !vectorsDeriv[i]->getValue().empty())
#endif
                v.index = i+1;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAlloc(VecId v)
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
    if (v.type == VecId::V_COORD && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = getVecCoord(v.index);
        vec->resize(vsize);
#ifdef SOFA_SMP
        vectorsCoordSharedAllocated[v.index]=true;
        BaseObject::Task< VecInitResize < VecCoord > >(this,**vec, this->vsize);
#endif
    }
    else if (v.type == VecId::V_DERIV && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecDeriv* vec = getVecDeriv(v.index);
        vec->resize(vsize);
#ifdef SOFA_SMP
        vectorsDerivSharedAllocated[v.index]=true;
        BaseObject::Task < VecInitResize < VecDeriv > >(this,**vec, this->vsize);
#endif
    }
    else
    {
        std::cerr << "Invalid alloc operation ("<<v<<")\n";
        return;
    }
    //vOp(v); // clear vector
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vFree(VecId vId)
{
    if (vId.type == VecId::V_COORD && vId.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = getVecCoord(vId.index);
        vec->resize(0);

        // Check X is not pointing on the deleted Dynamic Vector
        if (vec == &x.getValue())
        {
            if (this->f_printLog.getValue())
            {
                serr << "Warning! MechanicalObject " << this->getName() << " x vector is pointing on a deleted dynamic vector." << sendl;
                serr << "Restoring X" << sendl;
            }

            setX(VecId::position());
        }

        // Check XFree is not pointing on the deleted Dynamic Vector
        if (vec == &xfree.getValue())
        {
            if (this->f_printLog.getValue())
            {
                serr << "Warning! MechanicalObject " << this->getName() << " xfree vector is pointing on a deleted dynamic vector." << sendl;
                serr << "Restoring XFree" << sendl;
            }

            setXfree(VecId::freePosition());
        }

#ifdef SOFA_SMP
        vectorsCoordSharedAllocated[vId.index]=false;
#endif
    }
    else if (vId.type == VecId::V_DERIV && vId.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecDeriv* vec = getVecDeriv(vId.index);
        vec->resize(0);

        // Check V is not pointing on the deleted Dynamic Vector
        if (vec == &v.getValue())
        {
            if (this->f_printLog.getValue())
            {
                serr << "Warning! MechanicalObject " << this->getName() << " v vector is pointing on a deleted dynamic vector." << sendl;
                serr << "Restoring V" << sendl;
            }

            setV(VecId::velocity());
        }

        // Check VFree is not pointing on the deleted Dynamic Vector
        if (vec == &vfree.getValue())
        {
            if (this->f_printLog.getValue())
            {
                serr << "Warning! MechanicalObject " << this->getName() << " vfree vector is pointing on a deleted dynamic vector." << sendl;
                serr << "Restoring VFree" << sendl;
            }

            setVfree(VecId::freeVelocity());
        }

        // Check F is not pointing on the deleted Dynamic Vector
        if (vec == getF())
        {
            if (this->f_printLog.getValue())
            {
                serr << "Warning! MechanicalObject " << this->getName() << " f vector is pointing on a deleted dynamic vector with vecId = " << vId.index << "." << sendl;
                serr << "Restoring F" << sendl;
            }

            setF(VecId::force());
        }

        // Check InternalForces is not pointing on the deleted Dynamic Vector
        if (vec == &internalForces.getValue())
        {
            if (this->f_printLog.getValue())
            {
                serr << "Warning! MechanicalObject " << this->getName() << " internalForces vector is pointing on a deleted dynamic vector." << sendl;
                serr << "Restoring InternalForces" << sendl;
            }

            setF(VecId::internalForce());
        }

        // Check ExternalForces is not pointing on the deleted Dynamic Vector
        if (vec == &externalForces.getValue())
        {
            if (this->f_printLog.getValue())
            {
                serr << "Warning! MechanicalObject " << this->getName() << " externalForces vector is pointing on a deleted dynamic vector." << sendl;
                serr << "Restoring ExternalForces" << sendl;
            }

            setF(VecId::externalForce());
        }


#ifdef SOFA_SMP
        vectorsDerivSharedAllocated[vId.index]=false;
#endif
    }
    else
    {
        std::cerr << "Invalid free operation (" << vId << ")\n";
        return;
    }
}


#ifndef SOFA_SMP
template <class DataTypes>
void MechanicalObject<DataTypes>::vOp(VecId v, VecId a, VecId b, double f)
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
    if(v.isNull())
    {
        // ERROR
        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
        return;
    }
    if (a.isNull())
    {
        if (b.isNull())
        {
            // v = 0
            if (v.type == VecId::V_COORD)
            {
                VecCoord* vv = getVecCoord(v.index);
                vv->resize(this->vsize);
                for (unsigned int i=0; i<vv->size(); i++)
                    (*vv)[i] = Coord();
            }
            else
            {
                VecDeriv* vv = getVecDeriv(v.index);
                vv->resize(this->vsize);
                for (unsigned int i=0; i<vv->size(); i++)
                    (*vv)[i] = Deriv();
            }
        }
        else
        {
            if (b.type != v.type)
            {
                // ERROR
                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                return;
            }
            if (v == b)
            {
                // v *= f
                if (v.type == VecId::V_COORD)
                {
                    VecCoord* vv = getVecCoord(v.index);
                    for (unsigned int i=0; i<vv->size(); i++)
                        (*vv)[i] *= (Real)f;
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    for (unsigned int i=0; i<vv->size(); i++)
                        (*vv)[i] *= (Real)f;
                }
            }
            else
            {
                // v = b*f
                if (v.type == VecId::V_COORD)
                {
                    VecCoord* vv = getVecCoord(v.index);
                    VecCoord* vb = getVecCoord(b.index);
                    vv->resize(vb->size());
                    for (unsigned int i=0; i<vv->size(); i++)
                        (*vv)[i] = (*vb)[i] * (Real)f;
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    VecDeriv* vb = getVecDeriv(b.index);
                    vv->resize(vb->size());
                    for (unsigned int i=0; i<vv->size(); i++)
                        (*vv)[i] = (*vb)[i] * (Real)f;
                }
            }
        }
    }
    else
    {
        if (a.type != v.type)
        {
            // ERROR
            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
            return;
        }
        if (b.isNull())
        {
            // v = a
            if (v.type == VecId::V_COORD)
            {
                VecCoord* vv = getVecCoord(v.index);
                VecCoord* va = getVecCoord(a.index);
                vv->resize(va->size());
                for (unsigned int i=0; i<vv->size(); i++)
                    (*vv)[i] = (*va)[i];
            }
            else
            {
                VecDeriv* vv = getVecDeriv(v.index);
                VecDeriv* va = getVecDeriv(a.index);
                vv->resize(va->size());
                for (unsigned int i=0; i<vv->size(); i++)
                    (*vv)[i] = (*va)[i];
            }
        }
        else
        {
            if (v == a)
            {
                if (f==1.0)
                {
                    // v += b
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            vv->resize(vb->size());
                            for (unsigned int i=0; i<vv->size(); i++)
                                (*vv)[i] += (*vb)[i];
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            vv->resize(vb->size());
                            for (unsigned int i=0; i<vv->size(); i++)
                                (*vv)[i] += (*vb)[i];
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(vb->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                            (*vv)[i] += (*vb)[i];
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v += b*f
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            vv->resize(vb->size());
                            for (unsigned int i=0; i<vv->size(); i++)
                                (*vv)[i] += (*vb)[i]*(Real)f;
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            vv->resize(vb->size());
                            for (unsigned int i=0; i<vv->size(); i++)
                                (*vv)[i] += (*vb)[i]*(Real)f;
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(vb->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                            (*vv)[i] += (*vb)[i]*(Real)f;
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
            else if (v == b)
            {
                if (f==1.0)
                {
                    // v += a
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        if (a.type == VecId::V_COORD)
                        {
                            VecCoord* va = getVecCoord(a.index);
                            vv->resize(va->size());
                            for (unsigned int i=0; i<vv->size(); i++)
                                (*vv)[i] += (*va)[i];
                        }
                        else
                        {
                            VecDeriv* va = getVecDeriv(a.index);
                            vv->resize(va->size());
                            for (unsigned int i=0; i<vv->size(); i++)
                                (*vv)[i] += (*va)[i];
                        }
                    }
                    else if (a.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        vv->resize(va->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                            (*vv)[i] += (*va)[i];
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v = a+v*f
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        VecCoord* va = getVecCoord(a.index);
                        vv->resize(va->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                        {
                            (*vv)[i] *= (Real)f;
                            (*vv)[i] += (*va)[i];
                        }
                    }
                    else
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        vv->resize(va->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                        {
                            (*vv)[i] *= (Real)f;
                            (*vv)[i] += (*va)[i];
                        }
                    }
                }
            }
            else
            {
                if (f==1.0)
                {
                    // v = a+b
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        VecCoord* va = getVecCoord(a.index);
                        vv->resize(va->size());
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            for (unsigned int i=0; i<vv->size(); i++)
                            {
                                (*vv)[i] = (*va)[i];
                                (*vv)[i] += (*vb)[i];
                            }
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            for (unsigned int i=0; i<vv->size(); i++)
                            {
                                (*vv)[i] = (*va)[i];
                                (*vv)[i] += (*vb)[i];
                            }
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(va->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                        {
                            (*vv)[i] = (*va)[i];
                            (*vv)[i] += (*vb)[i];
                        }
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
                else
                {
                    // v = a+b*f
                    if (v.type == VecId::V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        VecCoord* va = getVecCoord(a.index);
                        vv->resize(va->size());
                        if (b.type == VecId::V_COORD)
                        {
                            VecCoord* vb = getVecCoord(b.index);
                            for (unsigned int i=0; i<vv->size(); i++)
                            {
                                (*vv)[i] = (*va)[i];
                                (*vv)[i] += (*vb)[i]*(Real)f;
                            }
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            for (unsigned int i=0; i<vv->size(); i++)
                            {
                                (*vv)[i] = (*va)[i];
                                (*vv)[i] += (*vb)[i]*(Real)f;
                            }
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* va = getVecDeriv(a.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(va->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                        {
                            (*vv)[i] = (*va)[i];
                            (*vv)[i] += (*vb)[i]*(Real)f;
                        }
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                        return;
                    }
                }
            }
        }
    }
}
#endif
template <class DataTypes>
void MechanicalObject<DataTypes>::vMultiOp(const VMultiOp& ops)
{
    // optimize common integration case: v += a*dt, x += v*dt
    if (ops.size() == 2 && ops[0].second.size() == 2 && ops[0].first == ops[0].second[0].first && ops[0].first.type == VecId::V_DERIV && ops[0].second[1].first.type == VecId::V_DERIV
        && ops[1].second.size() == 2 && ops[1].first == ops[1].second[0].first && ops[0].first == ops[1].second[1].first && ops[1].first.type == VecId::V_COORD)
    {
        const VecDeriv& va = *getVecDeriv(ops[0].second[1].first.index);
        VecDeriv& vv = *getVecDeriv(ops[0].first.index);
        VecCoord& vx = *getVecCoord(ops[1].first.index);
        const unsigned int n = vx.size();
        const Real f_v_v = (Real)(ops[0].second[0].second);
        const Real f_v_a = (Real)(ops[0].second[1].second);
        const Real f_x_x = (Real)(ops[1].second[0].second);
        const Real f_x_v = (Real)(ops[1].second[1].second);
        if (f_v_v == 1.0 && f_x_x == 1.0) // very common case
        {
            if (f_v_a == 1.0) // used by euler implicit and other integrators that directly computes a*dt
            {
                for (unsigned int i=0; i<n; ++i)
                {
                    vv[i] += va[i];
                    vx[i] += vv[i]*f_x_v;
                }
            }
            else
            {
                for (unsigned int i=0; i<n; ++i)
                {
                    vv[i] += va[i]*f_v_a;
                    vx[i] += vv[i]*f_x_v;
                }
            }
        }
        else if (f_x_x == 1.0) // some damping is applied to v
        {
            for (unsigned int i=0; i<n; ++i)
            {
                vv[i] *= f_v_v;
                vv[i] += va[i];
                vx[i] += vv[i]*f_x_v;
            }
        }
        else // general case
        {
            for (unsigned int i=0; i<n; ++i)
            {
                vv[i] *= f_v_v;
                vv[i] += va[i]*f_v_a;
                vx[i] *= f_x_x;
                vx[i] += vv[i]*f_x_v;
            }
        }
    }
    else // no optimization for now for other cases
        Inherited::vMultiOp(ops);
}

template <class T> inline void clear( T& t )
{
    t.clear();
}
template<> inline void clear( float& t )
{
    t=0;
}
template<> inline void clear( double& t )
{
    t=0;
}


template <class DataTypes>
void MechanicalObject<DataTypes>::vThreshold(VecId v, double t)
{
    if( v.type==VecId::V_DERIV)
    {
        helper::WriteAccessor<VecDeriv> vv = *getVecDeriv(v.index);
        Real t2 = (Real)(t*t);
        for (unsigned int i=0; i<vv.size(); i++)
        {
            if( vv[i]*vv[i] < t2 )
                clear(vv[i]);
        }
    }
    else
    {
        std::cerr<<"MechanicalObject<DataTypes>::vThreshold does not apply to coordinate vectors"<<std::endl;
    }
}

template <class DataTypes>
double MechanicalObject<DataTypes>::vDot(VecId a, VecId b)
{
    Real r = 0.0;
    if (a.type == VecId::V_COORD && b.type == VecId::V_COORD)
    {
        VecCoord* va = getVecCoord(a.index);
        VecCoord* vb = getVecCoord(b.index);
        for (unsigned int i=0; i<va->size(); i++)
            r += (*va)[i] * (*vb)[i];
    }
    else if (a.type == VecId::V_DERIV && b.type == VecId::V_DERIV)
    {
        VecDeriv* va = getVecDeriv(a.index);
        VecDeriv* vb = getVecDeriv(b.index);
        for (unsigned int i=0; i<va->size(); i++)
            r += (*va)[i] * (*vb)[i];
    }
    else
    {
        std::cerr << "Invalid dot operation ("<<a<<','<<b<<")\n";
    }
    return r;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setX(VecId vId)
{
    if (vId.type == VecId::V_COORD)
    {
        if (this->f_printLog.getValue() == true)
        {
            std::cout << "setX (" << vId.index << ") is called" << std::endl;
        }

        this->m_posId = vId;
    }
    else
    {
        std::cerr << "Invalid setX operation (" << vId << ")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setXfree(VecId vId)
{
    if (vId.type == VecId::V_COORD)
    {
        if (this->f_printLog.getValue() == true)
        {
            std::cout << "setXfree (" << vId.index << ") is called" << std::endl;
        }

        this->m_freePosId = vId;
    }
    else
    {
        std::cerr << "Invalid setXfree operation (" << vId << ")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVfree(VecId vId)
{
    if (vId.type == VecId::V_DERIV)
    {
        this->m_freeVelId = vId;
    }
    else
    {
        std::cerr << "Invalid setVfree operation (" << vId << ")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setV(VecId vId)
{
    if (vId.type == VecId::V_DERIV)
    {
        this->m_velId = vId;
    }
    else
    {
        std::cerr << "Invalid setV operation (" << vId << ")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setF(VecId vId)
{
    if (vId.type == VecId::V_DERIV)
    {
        this->m_forceId = vId;
    }
    else
    {
        std::cerr << "Invalid setF operation (" << vId << ")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setDx(VecId vId)
{
    if (vId.type == VecId::V_DERIV)
    {
        this->m_dxId = vId;
    }
    else
    {
        std::cerr << "Invalid setDx operation (" << vId << ")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setC(VecId vId)
{
    if (vId.type == VecId::V_CONST)
    {
        this->m_constraintId = vId;
    }
    else
    {
        std::cerr << "Invalid setC operation (" << vId << ")\n";
    }
}

#ifndef SOFA_SMP
template <class DataTypes>
void MechanicalObject<DataTypes>::printDOF( VecId v, std::ostream& out, int firstIndex, int range) const
{
    const unsigned int size=this->getSize();
    if ((unsigned int) (abs(firstIndex)) >= size) return;
    const unsigned int first=((firstIndex>=0)?firstIndex:size+firstIndex);
    const unsigned int max=( ( (range >= 0) && ( (range+first)<size) ) ? (range+first):size);
    if( v.type==VecId::V_COORD)
    {
        if (getVecCoord(v.index))
        {
            const VecCoord& x= *getVecCoord(v.index);
            if (x.empty()) return;
            for( unsigned i=first; i<max; ++i )
            {
                out<<x[i];
                if (i != max-1) out <<" ";
            }
        }
    }
    else if( v.type==VecId::V_DERIV)
    {
        if (getVecDeriv(v.index))
        {
            const VecDeriv& x= *getVecDeriv(v.index);
            if (x.empty()) return;
            for( unsigned i=first; i<max; ++i )
            {
                out<<x[i];
                if (i != max-1) out <<" ";
            }
        }
    }
    else
        out<<"MechanicalObject<DataTypes>::printDOF, unknown v.type = "<<v.type<<std::endl;
}
#endif

template <class DataTypes>
unsigned MechanicalObject<DataTypes>::printDOFWithElapsedTime(VecId v, unsigned count, unsigned time, std::ostream& out)
{
    if (v.type == VecId::V_COORD)
    {
        VecCoord& x = *getVecCoord(v.index);

        for (unsigned i = 0; i < x.size(); ++i)
        {
            out << count + i << "\t" << time << "\t" << x[i] << std::endl;
        }
        out << std::endl << std::endl;
        return x.size();
    }
    else if (v.type == VecId::V_DERIV)
    {
        VecDeriv& x = *getVecDeriv(v.index);

        for (unsigned i = 0; i < x.size(); ++i)
        {
            out << count + i << "\t" << time << "\t" << x[i] << std::endl;
        }
        out << std::endl << std::endl;

        return x.size();
    }
    else
        out << "MechanicalObject<DataTypes>::printDOFWithElapsedTime, unknown v.type = " << v.type << std::endl;

    return 0;
}


template <class DataTypes>
void MechanicalObject<DataTypes>::resetForce()
{
#ifdef SOFA_SMP
    BaseObject::Task< vClear<VecDeriv, Deriv> >(this, **getF());
#else
    helper::WriteAccessor<VecDeriv> f = *getF();

    if (!this->forceMask.isInUse())
    {
        for (unsigned i = 0; i < f.size(); ++i)
        {
            f[i] = Deriv();
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;

        const ParticleMask::InternalStorage &indices = this->forceMask.getEntries();
        ParticleMask::InternalStorage::const_iterator it;

        for (it = indices.begin(); it != indices.end(); it++)
        {
            f[(*it)] = Deriv();
        }
    }
#endif
}

template <class DataTypes>
void MechanicalObject<DataTypes>::resetAcc()
{
#ifdef SOFA_SMP
    BaseObject::Task< vClear<VecDeriv, Deriv> >(this, **getDx());
#else
    helper::WriteAccessor<VecDeriv> a= *getDx();

    for (unsigned i = 0; i < a.size(); ++i)
    {
        a[i] = Deriv();
    }
#endif
}



template <class DataTypes>
void MechanicalObject<DataTypes>::resetConstraint()
{
    //	std::cout << "resetConstraint()\n";
    VecConst& c= *getC();
    c.clear();

    constraintId.clear();
}


template <class DataTypes>
void MechanicalObject<DataTypes>::setConstraintId(unsigned int i)
{
    constraintId.push_back(i);

    //for (int j=0; j<constraintId.size(); j++)
    //{
    //	std::cout << "constraintId[j] = " << constraintId[j] << std::endl;
    //}
}

template <class DataTypes>
sofa::helper::vector<unsigned int>& MechanicalObject<DataTypes>::getConstraintId()
{
    return constraintId;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::renumberConstraintId(const sofa::helper::vector<unsigned>& renumbering)
{
    for (unsigned int i = 0; i < constraintId.size(); ++i)
        constraintId[i] = renumbering[constraintId[i]];
}

template <class DataTypes>
std::list< core::behavior::BaseMechanicalState::ConstraintBlock > MechanicalObject<DataTypes>::constraintBlocks( const std::list<unsigned int> &indices) const
{
    const unsigned int dimensionDeriv = defaulttype::DataTypeInfo< Deriv >::size();
    assert( indices.size() > 0 );
    assert( dimensionDeriv > 0 );

    // simple column/block map

    typedef sofa::component::linearsolver::SparseMatrix<SReal> matrix_t;
    // typedef sofa::component::linearsolver::FullMatrix<SReal> matrix_t;

    typedef std::map<unsigned int, matrix_t* > blocks_t;
    blocks_t blocks;

    // for all row indices
    typedef std::list<unsigned int> indices_t;

    unsigned int block_row = 0;
    for(indices_t::const_iterator rowIt = indices.begin(); rowIt != indices.end(); ++rowIt, ++block_row)
    {
        unsigned int row = getIdxConstraintFromId(*rowIt);

        // for all sparse data in the row
        assert( row < c.getValue().size() );
        std::pair< ConstraintIterator, ConstraintIterator > range = c.getValue()[row].data();
        ConstraintIterator chunk = range.first, last = range.second;
        for( ; chunk != last; ++chunk)
        {
            const unsigned int column = chunk->first;

            // do we already have a block for this column ?
            if( blocks.find( column ) == blocks.end() )
            {
                // nope: let's create it
                matrix_t* mat = new matrix_t(indices.size(), dimensionDeriv);
                blocks[column] = mat;

                // for(unsigned int i = 0; i < mat->rowSize(); ++i) {
                //   for(unsigned int j = 0; j < mat->colSize(); ++j) {
                //     mat->set(i, j, 0);
                //   }
                // }

            }

            // now it's created no matter what \o/
            matrix_t& block = *blocks[column];

            // fill the right line of the block
            for (unsigned int i = 0; i < dimensionDeriv; ++i)
            {
                SReal value;
                defaulttype::DataTypeInfo< Deriv >::getValue(chunk->second, i, value); // somebody should pay for this
                block.set(block_row, i, value);
            }
        }
    }

    // put all blocks in a list and we're done
    std::list<ConstraintBlock> res;
    for(blocks_t::const_iterator b = blocks.begin(); b != blocks.end(); ++b)
    {
        res.push_back( ConstraintBlock( b->first, b->second ) );
    }

    return res;
}

template <class DataTypes>
SReal MechanicalObject<DataTypes>::getConstraintJacobianTimesVecDeriv( unsigned int line, VecId id)
{
    SReal result = 0;
    if (std::find(constraintId.begin(), constraintId.end(), line) == constraintId.end()) return 0;

    SparseVecDeriv &value = (*c.beginEdit())[getIdxConstraintFromId(line)];

    VecDeriv *data = 0;

    //Maybe we should extend this to restvelocity
    if (id == VecId::velocity())
    {
        data = v.beginEdit();
    }
    else if (id == VecId::dx())
    {
        data = dx.beginEdit();
    }
    else
    {
        this->serr << "getConstraintJacobianTimesVecDeriv " << "NOT IMPLEMENTED for " << id.getName() << this->sendl;
        return 0;
    }

    std::pair< SparseVecDerivIterator, SparseVecDerivIterator > range = value.data();

    for (SparseVecDerivIterator it = range.first; it != range.second; ++it)
    {
        result += it->second * (*data)[it->first];
    }

    c.endEdit();

    return result;
}


template <class DataTypes>
bool MechanicalObject<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *getX();
    const unsigned int xSize = x.size();

    if (xSize <= 0)
        return false;

    Real p[3] = {0,0,0};

    for (unsigned int i = 0; i < xSize; i++)
    {
        DataTypes::get(p[0], p[1], p[2], x[i]);
        for (int c = 0; c < 3; c++)
        {
            if (p[c] > maxBBox[c])
                maxBBox[c] = p[c];

            if (p[c] < minBBox[c])
                minBBox[c] = p[c];
        }
    }

    return true;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::draw()
{
    Mat<4,4, GLfloat> modelviewM;
    Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context;
    if (debugViewIndices.getValue())
    {
        context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
        glColor3f(1.0,1.0,1.0);
        glDisable(GL_LIGHTING);
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (sceneMaxBBox - sceneMinBBox).norm() * debugViewIndicesScale.getValue();

        for (int i=0 ; i< vsize ; i++)
        {
            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            //glVertex3f(getPX(i),getPY(i),getPZ(i) );
            glPushMatrix();

            glTranslatef(getPX(i), getPY(i), getPZ(i));
            glScalef(scale,scale,scale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            Vec3d temp(getPX(i), getPY(i), getPZ(i));
            temp = modelviewM.transform(temp);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(scale,scale,scale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();

        }
    }

}
#ifdef SOFA_SMP
template < class DataTypes >
void MechanicalObject < DataTypes >::vOpMEq (VecId v, VecId b,
        a1::Shared <
        double >*f)
{

    if (v.type == VecId::V_COORD)
    {

        if (b.type == VecId::V_COORD)
        {


            BaseObject::Task < vOpMinusEqualMult < DataTypes, VecCoord,
                       VecCoord > >(this,**getVecCoord (v.index),
                               **getVecCoord(b.index), *f);

        }
        else
        {
            BaseObject::Task < vOpMinusEqualMult < DataTypes, VecCoord,
                       VecDeriv > >(this,**getVecCoord (v.index),
                               **getVecDeriv (b.index), *f);

        }
    }
    else if (b.type == VecId::V_DERIV)
    {
        BaseObject::Task < vOpMinusEqualMult < DataTypes, VecDeriv,
                   VecDeriv > >(this,**getVecDeriv (v.index),
                           **getVecDeriv (b.index), *f);

    }
    else
    {
        // ERROR
        //  std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
        return;
    }


}
template < class DataTypes >
void MechanicalObject < DataTypes >::vOp(VecId v, VecId a, VecId b , double f) { vOp(v,a,b,f,NULL);}

template < class DataTypes >
void MechanicalObject < DataTypes >::vOp (VecId v, VecId a,
        VecId b, double f,
        a1::Shared  < double >*fSh)
{
    if (v.isNull ())
    {
        // ERROR
        std::cerr << "Invalid vOp operation (" << v << ',' << a << ','
                << b << ',' << f << ")\n";
        return;
    }
    if (a.isNull ())
    {
        if (b.isNull ())
        {
            // v = 0
            if (v.type == VecId::V_COORD)
            {
                //VecCoord* vv = getVecCoord(v.index);
                BaseObject::Task < vClear < VecDeriv,Deriv >
                >(this,**getVecCoord (v.index), (unsigned) (this->vsize));

                /*unsigned int vt=ExecutionGraph::add_operation("v=0");
                ExecutionGraph::read_var(vt,vv);
                ExecutionGraph::write_var(vt,vv); */




            }
            else
            {
                //  VecDeriv* vv = getVecDeriv(v.index);
                BaseObject::Task < vClear < VecCoord,Coord >
                >(this,**getVecDeriv (v.index), (unsigned) (this->vsize));
                /*unsigned int vt=ExecutionGraph::add_operation("v=0");
                ExecutionGraph::read_var(vt,vv);
                ExecutionGraph::write_var(vt,vv);
                vv->resize(this->this->vsize);
                for (unsigned int i=0; i<vv->size(); i++)
                (*vv)[i] = Deriv();
                */
            }

        }
        else
        {
            if (b.type != v.type)
            {
                // ERROR
                std::cerr << "Invalid vOp operation (" << v << ',' << a
                        << ',' << b << ',' << f << ")\n";
                return;
            }
            if (v == b)
            {
                // v *= f
                if (v.type == VecId::V_COORD)
                {

                    /*VecCoord* vv = getVecCoord(v.index);
                    unsigned int vt=ExecutionGraph::add_operation("v*=f");
                    ExecutionGraph::read_var(vt,vv);
                    ExecutionGraph::write_var(vt,vv); */
                    BaseObject::Task < vTEq < VecCoord, Real >
                    >(this,**getVecCoord (v.index), f);
                }
                else
                {
                    /*            VecDeriv* vv = getVecDeriv(v.index);
                    unsigned int vt=ExecutionGraph::add_operation("v*=f");
                    ExecutionGraph::read_var(vt,vv);
                    ExecutionGraph::write_var(vt,vv);
                    */




                    BaseObject::Task < vTEq < VecDeriv, Real >
                    >(this,**getVecDeriv (v.index), f);





                }
            }
            else
            {
                // v = b*f
                if (v.type == VecId::V_COORD)
                {
                    BaseObject::Task < vEqBF < VecCoord,Real >
                    >(this,**getVecCoord (b.index),
                      **getVecCoord (v.index), f);
                }
                else
                {
                    BaseObject::Task < vEqBF < VecDeriv,Real >
                    >(this,**getVecDeriv (b.index),
                      **getVecDeriv (v.index), f);
                }
            }
        }
    }
    else
    {
        if (a.type != v.type)
        {
            // ERROR
            std::cerr << "Invalid vOp operation (" << v << ',' << a <<
                    ',' << b << ',' << f << ")\n";
            return;
        }
        if (b.isNull ())
        {
            // v = a
            if (v.type == VecId::V_COORD)
            {
                BaseObject::Task < vAssign <
                VecCoord > >(this,**getVecCoord (v.index),
                        **getVecCoord (a.index));
            }
            else
            {
                BaseObject::Task < vAssign <
                VecDeriv > >(this,**getVecDeriv (v.index),
                        **getVecDeriv (a.index));
            }
        }
        else
        {
            if (v == a)
            {
                if (f == 1.0 && !fSh)
                {
                    // v += b
                    if (v.type == VecId::V_COORD)
                    {

                        if (b.type == VecId::V_COORD)
                        {

                            BaseObject::Task < vPEq < VecCoord,
                                       VecCoord >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (b.index));

                        }
                        else
                        {
                            BaseObject::Task < vPEq < VecCoord,
                                       VecDeriv >
                                       >(this,**getVecCoord (v.index),
                                               **getVecDeriv (b.index));

                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {

                        BaseObject::Task < vPEq <  VecDeriv,
                                   VecDeriv >
                                   >(this,**getVecDeriv (v.index),
                                           **getVecDeriv (b.index));

                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation (" << v <<
                                ',' << a << ',' << b << ',' << f << ")\n";
                        return;
                    }
                }
                else
                {
                    // v += b*f
                    if (v.type == VecId::V_COORD)
                    {
                        if (b.type == VecId::V_COORD)
                        {

                            if (fSh)
                            {
                                BaseObject::Task < vPEqBF < DataTypes,
                                           VecCoord,
                                           VecCoord >
                                           >(this,**getVecCoord (v.index),
                                                   **getVecCoord (b.index), *fSh, f);
                            }
                            else
                            {
                                BaseObject::Task < vPEqBF < DataTypes,
                                           VecCoord,
                                           VecCoord >
                                           >(this,**getVecCoord (v.index),
                                                   **getVecCoord (b.index), f);
                            }
                        }
                        else
                        {
                            if (fSh)
                            {
                                BaseObject::Task < vPEqBF < DataTypes, VecCoord, VecDeriv > >(this,**getVecCoord (v.index), **getVecDeriv (b.index), *fSh, f);
                            }
                            else
                            {
                                BaseObject::Task < vPEqBF < DataTypes,
                                           VecCoord,
                                           VecDeriv >
                                           >(this,**getVecCoord (v.index),
                                                   **getVecDeriv (b.index), f);
                            }
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        if (fSh)
                        {
                            BaseObject::Task < vPEqBF < DataTypes,
                                       VecDeriv,
                                       VecDeriv >
                                       >(this,**getVecDeriv (v.index),
                                               **getVecDeriv (b.index), *fSh, f);
                        }
                        else
                        {
                            BaseObject::Task < vPEqBF < DataTypes,
                                       VecDeriv,
                                       VecDeriv >
                                       >(this,**getVecDeriv (v.index),
                                               **getVecDeriv (b.index), f);
                        }
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation (" << v <<
                                ',' << a << ',' << b << ',' << f << ")\n";
                        return;
                    }
                }
            }
            else if (v == b)
            {
                if (f == 1.0 && !fSh)
                {
                    // v += a
                    if (v.type == VecId::V_COORD)
                    {

                        if (a.type == VecId::V_COORD)
                        {
                            BaseObject::Task < vPEq < VecCoord,
                                       VecCoord >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (a.index));

                        }
                        else
                        {
                            BaseObject::Task < vPEq <  VecCoord,
                                       VecDeriv >
                                       >(this,**getVecCoord (v.index),
                                               **getVecDeriv (a.index));

                        }
                    }
                    else if (a.type == VecId::V_DERIV)
                    {
                        BaseObject::Task<vPEq <VecCoord,VecDeriv> > (this,**getVecDeriv(v.index),**getVecDeriv(a.index));
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation (" << v <<
                                ',' << a << ',' << b << ',' << f << ")\n";
                        return;
                    }
                }
                else
                {
                    // v = a+v*f
                    if (v.type == VecId::V_COORD)
                    {
                        if (fSh)
                        {
                            BaseObject::Task < vOpSumMult < DataTypes, VecCoord,
                                       VecCoord >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (a.index), *fSh,
                                               (Real) f);
                        }
                        else
                        {
                            BaseObject::Task < vOpSumMult < DataTypes, VecCoord,
                                       VecCoord >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (a.index), (Real) f);
                        }
                    }
                    else
                    {
                        if (fSh)
                        {
                            BaseObject::Task < vOpSumMult < DataTypes, VecDeriv,
                                       VecDeriv >
                                       >(this,**getVecDeriv (v.index),
                                               **getVecDeriv (a.index), *fSh,
                                               (Real) f);
                        }
                        else
                        {
                            BaseObject::Task < vOpSumMult < DataTypes, VecDeriv,
                                       VecDeriv >
                                       >(this,**getVecDeriv (v.index),
                                               **getVecDeriv (a.index), (Real) f);
                        }
                    }
                }
            }
            else
            {
                if (f == 1.0 && !fSh)
                {
                    // v = a+b
                    if (v.type == VecId::V_COORD)
                    {
                        if (b.type == VecId::V_COORD)
                        {
                            BaseObject::Task < vAdd < VecCoord,
                                       VecCoord, VecCoord >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (a.index),
                                               **getVecCoord (b.index));
                        }
                        else
                        {
                            BaseObject::Task < vAdd <  VecCoord,
                                       VecCoord, VecDeriv >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (a.index),
                                               **getVecDeriv (b.index));
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        BaseObject::Task < vAdd < VecDeriv,
                                   VecDeriv, VecDeriv >
                                   >(this,**getVecDeriv (v.index),
                                           **getVecDeriv (a.index),
                                           **getVecDeriv (b.index));
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation (" << v <<
                                ',' << a << ',' << b << ',' << f << ")\n";
                        return;
                    }
                }
                else
                {
                    // v = a+b*f
                    if (v.type == VecId::V_COORD)
                    {
                        if (b.type == VecId::V_COORD)
                        {
                            BaseObject::Task < vOpVeqAplusBmultF < DataTypes,
                                       VecCoord,
                                       VecCoord >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (a.index),
                                               **getVecCoord (b.index), (Real) f);
                        }
                        else
                        {
                            BaseObject::Task < vOpVeqAplusBmultF < DataTypes,
                                       VecCoord,
                                       VecDeriv >
                                       >(this,**getVecCoord (v.index),
                                               **getVecCoord (a.index),
                                               **getVecDeriv (b.index), (Real) f);
                        }
                    }
                    else if (b.type == VecId::V_DERIV)
                    {
                        BaseObject::Task < vOpVeqAplusBmultF < DataTypes,
                                   VecDeriv,
                                   VecDeriv >
                                   >(this,**getVecDeriv (v.index),
                                           **getVecDeriv (a.index),
                                           **getVecDeriv (b.index), (Real) f);
                    }
                    else
                    {
                        // ERROR
                        std::cerr << "Invalid vOp operation (" << v <<
                                ',' << a << ',' << b << ',' << f << ")\n";
                        return;
                    }
                }
            }
        }
    }
}



template < class DataTypes >
void MechanicalObject < DataTypes >::vDot ( a1::Shared  < double >*res,VecId a, VecId b)
{
    //      double r = 0.0;
    if (a.type == VecId::V_COORD && b.type == VecId::V_COORD)
    {
        if (a.index == b.index)
        {
            BaseObject::Task < vDotOp < DataTypes,
                       VecCoord > >(this,**getVecCoord (a.index), *res);
        }
        else
        {
            BaseObject::Task < vDotOp < DataTypes,
                       VecCoord > >(this,**getVecCoord (a.index),
                               **getVecCoord (b.index), *res);
        }
    }
    else if (a.type == VecId::V_DERIV && b.type == VecId::V_DERIV)
    {
        if (a.index == b.index)
        {
            BaseObject::Task < vDotOp < DataTypes,
                       VecDeriv > >(this,**getVecDeriv (a.index), *res);
        }
        else
        {
            BaseObject::Task < vDotOp < DataTypes,
                       VecDeriv > >(this,**getVecDeriv (a.index),
                               **getVecDeriv (b.index), *res);
        }
    }
    else
    {
        std::cerr << "Invalid dot operation (" << a << ',' << b << ")\n";
    }
    // return r;
}

//     template < class DataTypes >
//       void MechanicalObject < DataTypes >::setX (VecId v)
//     {
//       if (v.type == VecId::V_COORD)
// 	{
//
// 	  this->xSh = *getVecCoord (v.index);
//
// 	  this->x = getVecCoord (v.index);
//
// 	}
//       else
// 	{
// 	  std::cerr << "Invalid setX operation (" << v << ")\n";
// 	}
//     }
//
//     template < class DataTypes >
//       void ParallelMechanicalObject < DataTypes >::setXfree (VecId v)
//     {
//       if (v.type == VecId::V_COORD)
// 	{
// 	  this->xfreeSh = *getVecCoord (v.index);
//
// 	  this->xfree = getVecCoord (v.index);
//
// 	}
//       else
// 	{
// 	  std::cerr << "Invalid setXfree operation (" << v << ")\n";
// 	}
//     }
//
//     template < class DataTypes >
//       void ParallelMechanicalObject < DataTypes >::setVfree (VecId v)
//     {
//       if (v.type == VecId::V_DERIV)
// 	{
//
// 	  this->vfreeSh = *getVecDeriv (v.index);
//
// 	  this->vfree = getVecDeriv (v.index);
//
// 	}
//       else
// 	{
// 	  std::cerr << "Invalid setVfree operation (" << v << ")\n";
// 	}
//     }
//
//     template < class DataTypes >
//       void ParallelMechanicalObject < DataTypes >::setV (VecId v)
//     {
//       if (v.type == VecId::V_DERIV)
// 	{
// 	  this->vSh = *getVecDeriv (v.index);
//
// 	  this->v = getVecDeriv (v.index);
//
// 	}
//       else
// 	{
// 	  std::cerr << "Invalid setV operation (" << v << ")\n";
// 	}
//     }
//
//     template < class DataTypes >
//       void ParallelMechanicalObject < DataTypes >::setF (VecId v)
//     {
//       if (v.type == VecId::V_DERIV)
// 	{
//
// 	  this->fSh = *getVecDeriv (v.index);
//
// 	  this->f = getVecDeriv (v.index);
//
// 	}
//       else
// 	{
// 	  std::cerr << "Invalid setF operation (" << v << ")\n";
// 	}
//     }
//
//     template < class DataTypes >
//       void ParallelMechanicalObject < DataTypes >::setDx (VecId v)
//     {
//       if (v.type == VecId::V_DERIV)
// 	{
// 	  this->dxSh = *getVecDeriv (v.index);
//
//
// 	  this->dx = getVecDeriv (v.index);
//
// 	}
//       else
// 	{
// 	  std::cerr << "Invalid setDx operation (" << v << ")\n";
// 	}
//     }
//
//
template < class DataTypes >
void MechanicalObject < DataTypes >::printDOF (VecId /*v*/,
        std::
        ostream & /*out*/, int /* firstIndex */, int /* range */) const
{
    // 	if (v.type == VecId::V_COORD)
    // 	{
    // 		VecCoord & x = *getVecCoord (v.index);
    // 		Task < printDOFSh < VecCoord > >(this,*x);
    // 	}
    // 	else if (v.type == VecId::V_DERIV)
    // 	{
    // 		VecDeriv & x = *getVecDeriv (v.index);
    // 		Task < printDOFSh < VecDeriv > >(this,*x);
    // 	}
    // 	else
    // 		out << "ParallelMechanicalObject<DataTypes>::printDOF, unknown v.type = " <<
    // 			v.type << std::endl;

}
#endif


/// Find mechanical particles hit by the given ray.
/// A mechanical particle is defined as a 2D or 3D, position or rigid DOF
/// Returns false if this object does not support picking
template <class DataTypes>
bool MechanicalObject<DataTypes>::pickParticles(double rayOx, double rayOy, double rayOz, double rayDx, double rayDy, double rayDz, double radius0, double dRadius,
        std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> >& particles)
{
    if (DataTypeInfo<Coord>::size() == 2 || DataTypeInfo<Coord>::size() == 3
        || (DataTypeInfo<Coord>::size() == 7 && DataTypeInfo<Deriv>::size() == 6))
    {
        // seems to be valid DOFs
        const VecCoord& x = *this->getX();
        Vec<3,Real> origin(rayOx, rayOy, rayOz);
        Vec<3,Real> direction(rayDx, rayDy, rayDz);
        for (int i=0; i< vsize; ++i)
        {
            Vec<3,Real> pos;
            DataTypes::get(pos[0],pos[1],pos[2],x[i]);

            if (pos == origin) continue;
            double dist = (pos-origin)*direction;
            if (dist < 0) continue;

            Vec<3,Real> vecPoint = (pos-origin) - direction*dist;
            double distToRay = vecPoint.norm2();
            double maxr = radius0 + dRadius*dist;
            double r2 = (pos-origin-direction*dist).norm2();
            if (r2 <= maxr*maxr)
                particles.insert(std::make_pair(distToRay,std::make_pair(this,i)));
        }
        return true;
    }
    else
        return false;
}


template <class DataTypes>
unsigned int  MechanicalObject<DataTypes>::getIdxConstraintFromId(unsigned int id) const
{
    const unsigned int constraintIdSize = constraintId.size();

    for (unsigned int i = 0; i < constraintIdSize; ++i)
    {
        if (constraintId[i] == id)
            return i;
    }

    serr << "Constraint Equation " << id << " Was not found!" << sendl;

    return 0;
}


} // namespace container

} // namespace component

} // namespace sofa

#endif

