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
#ifdef SOFA_SMP
#include <sofa/component/container/MechanicalObjectTasks.inl>
#endif
#include <sofa/component/container/MeshLoader.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/component/topology/RegularGridTopology.h>

#include <sofa/defaulttype/DataTypeInfo.h>

#include <sofa/helper/accessor.h>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/system/glut.h>

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/simulation/common/Visitor.h>
#endif

#include <assert.h>
#include <iostream>

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
using namespace sofa::core;
using namespace sofa::core::topology;
using namespace sofa::defaulttype;

template <class DataTypes>
MechanicalObject<DataTypes>::MechanicalObject()
    : x(initData(&x, "position", "position coordinates of the degrees of freedom"))
    , v(initData(&v, "velocity", "velocity coordinates of the degrees of freedom"))
    , f(initData(&f, "force", "force vector of the degrees of freedom"))
    , externalForces(initData(&externalForces, "externalForce", "externalForces vector of the degrees of freedom"))
    , dx(initData(&dx, "derivX", "dx vector of the degrees of freedom"))
    , xfree(initData(&xfree, "free_position", "free position coordinates of the degrees of freedom"))
    , vfree(initData(&vfree, "free_velocity", "free velocity coordinates of the degrees of freedom"))
    , x0(initData(&x0, "rest_position", "rest position coordinates of the degrees of freedom"))
    , c(initData(&c, "constraint", "constraints applied to the degrees of freedom"))
    , reset_position(initData(&reset_position, "reset_position", "reset position coordinates of the degrees of freedom"))
    , reset_velocity(initData(&reset_velocity, "reset_velocity", "reset velocity coordinates of the degrees of freedom"))
    , restScale(initData(&restScale, (SReal)1.0, "restScale", "optional scaling of rest position coordinates (to simulated pre-existing internal tension)"))
    , showObject(initData(&showObject, (bool) false, "showObject", "Show objects"))
    , showObjectScale(initData(&showObjectScale, (float) 0.1, "showObjectScale", "Scale for object display"))
    , showIndices(initData(&showIndices, (bool) false, "showIndices", "Show indices"))
    , showIndicesScale(initData(&showIndicesScale, (float) 0.0001, "showIndicesScale", "Scale for indices display"))
    , translation(initData(&translation, Vector3(), "translation", "Translation of the DOFs"))
    , rotation(initData(&rotation, Vector3(), "rotation", "Rotation of the DOFs"))
    , scale(initData(&scale, Vector3(1.0,1.0,1.0), "scale3d", "Scale of the DOFs in 3 dimensions"))
    , translation2(initData(&translation2, Vector3(), "translation2", "Translation of the DOFs, applied after the rest position has been computed"))
    , rotation2(initData(&rotation2, Vector3(), "rotation2", "Rotation of the DOFs, applied the after the rest position has been computed"))
    , filename(initData(&filename, std::string(""), "filename", "File corresponding to the Mechanical Object", false))
    , ignoreLoader(initData(&ignoreLoader, (bool) false, "ignoreLoader", "Is the Mechanical Object do not use a loader"))
    , f_reserve(initData(&f_reserve, 0, "reserve", "Size to reserve when creating vectors"))
    , vsize(0)
    , data(this)
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
    externalForces	.setGroup("Vector");
    dx				.setGroup("Vector");
    xfree			.setGroup("Vector");
    vfree			.setGroup("Vector");
    x0				.setGroup("Vector");
    c				.setGroup("Vector");
    reset_position	.setGroup("Vector");
    reset_velocity	.setGroup("Vector");

    translation		.setGroup("Transformation");
    translation2	.setGroup("Transformation");
    rotation		.setGroup("Transformation");
    rotation2		.setGroup("Transformation");
    scale			.setGroup("Transformation");

    // Deactivate the Filter.
    // MechanicalObjects created during the collision response must not use the filter as it will be empty
    this->forceMask.activate(false);

    setVecCoord(core::VecCoordId::position().index, &x);
    setVecCoord(core::VecCoordId::freePosition().index, &xfree);
    setVecCoord(core::VecCoordId::restPosition().index, &x0);
    setVecCoord(core::VecCoordId::resetPosition().index, &reset_position);
    setVecDeriv(core::VecDerivId::velocity().index, &v);
    setVecDeriv(core::VecDerivId::force().index, &f);
    setVecDeriv(core::VecDerivId::externalForce().index, &externalForces);
    setVecDeriv(core::VecDerivId::dx().index, &dx);
    setVecDeriv(core::VecDerivId::freeVelocity().index, &vfree);
    setVecDeriv(core::VecDerivId::resetVelocity().index, &reset_velocity);
    setVecMatrixDeriv(core::MatrixDerivId::holonomicC().index, &c);

    // These vectors are set as modified as they are mandatory in the MechanicalObject.
    x				.forceSet();
//	x0				.forceSet();
    v				.forceSet();
    dx				.forceSet();
    f				.forceSet();
    externalForces	.forceSet();

    write(VecCoordId::null())->forceSet();
    write(VecDerivId::null())->forceSet();

    // default size is 1
    resize(1);
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

        Data<typename DataTypes::VecCoord> *x_d = dest->write(VecCoordId::position());
        Data<typename DataTypes::VecDeriv> *v_d = dest->write(VecDerivId::velocity());

        DataTypes::set((*x_d->beginEdit())[index], px, py, pz);
        DataTypes::set((*v_d->beginEdit())[index], vx, vy, vz);
        ++index;

        x_d->endEdit();
        v_d->endEdit();
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
    if (m_gnuplotFileV != NULL)
        delete m_gnuplotFileV;

    if (m_gnuplotFileX != NULL)
        delete m_gnuplotFileX;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::handleStateChange()
{
    using sofa::core::topology::TopologyChange;

    std::list< const TopologyChange * >::const_iterator itBegin = m_topology->beginStateChange();
    std::list< const TopologyChange * >::const_iterator itEnd = m_topology->endStateChange();

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

    Coord tmp;
    Deriv tmp2;

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

    //if (size!=vsize)
    {
        vsize = size;
        for (unsigned int i = 0; i < vectorsCoord.size(); i++)
        {
            if (vectorsCoord[i] != NULL && vectorsCoord[i]->isSet())
            {
                vectorsCoord[i]->beginEdit()->resize(size);
                vectorsCoord[i]->endEdit();
            }
        }

        for (unsigned int i = 0; i < vectorsDeriv.size(); i++)
        {
            if (vectorsDeriv[i] != NULL && vectorsDeriv[i]->isSet())
            {
                vectorsDeriv[i]->beginEdit()->resize(size);
                vectorsDeriv[i]->endEdit();
            }
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::reserve(const int size)
{
    if (size == 0) return;

    for (unsigned int i = 0; i < vectorsCoord.size(); i++)
    {
        if (vectorsCoord[i] != NULL && vectorsCoord[i]->isSet())
        {
            vectorsCoord[i]->beginEdit()->reserve(size);
            vectorsCoord[i]->endEdit();
        }
    }

    for (unsigned int i = 0; i < vectorsDeriv.size(); i++)
    {
        if (vectorsDeriv[i] != NULL && vectorsDeriv[i]->isSet())
        {
            vectorsDeriv[i]->beginEdit()->reserve(size);
            vectorsDeriv[i]->endEdit();
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::applyTranslation (const double dx, const double dy, const double dz)
{
    helper::WriteAccessor< Data<VecCoord> > x_wA = *this->write(VecCoordId::position());

    for (unsigned int i = 0; i < x_wA.size(); i++)
    {
        DataTypes::add(x_wA[i], dx, dy, dz);
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
    helper::WriteAccessor< Data<VecCoord> > x_wA = *this->write(VecCoordId::position());

    for (unsigned int i = 0; i < x_wA.size(); i++)
    {
        Vec<3,Real> pos;
        DataTypes::get(pos[0], pos[1], pos[2], x_wA[i]);
        Vec<3,Real> newposition = q.rotate(pos);
        DataTypes::set(x_wA[i], newposition[0], newposition[1], newposition[2]);
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
    helper::WriteAccessor< Data<VecCoord> > x_wA = *this->write(VecCoordId::position());

    const Vector3 s(sx, sy, sz);
    for (unsigned int i=0; i<x_wA.size(); i++)
    {
        x_wA[i][0] = x_wA[i][0] * sx;
        x_wA[i][1] = x_wA[i][1] * sy;
        x_wA[i][2] = x_wA[i][2] * sz;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::getIndicesInSpace(sofa::helper::vector<unsigned>& indices, Real xmin, Real xmax, Real ymin, Real ymax, Real zmin, Real zmax) const
{
    helper::ReadAccessor< Data<VecCoord> > x_rA = *this->read(ConstVecCoordId::position());

    for( unsigned i=0; i<x_rA.size(); ++i )
    {
        Real x=0.0,y=0.0,z=0.0;
        DataTypes::get(x,y,z,x_rA[i]);
        if( x >= xmin && x <= xmax && y >= ymin && y <= ymax && z >= zmin && z <= zmax )
        {
            indices.push_back(i);
        }
    }
}

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

// Force the position of a point (and force its velocity to zero value)
template <class DataTypes>
void MechanicalObject<DataTypes>::forcePointPosition(const unsigned int i, const sofa::helper::vector< double >& m_x)
{
    helper::WriteAccessor< Data<VecCoord> > x_wA = *this->write(VecCoordId::position());
    helper::WriteAccessor< Data<VecDeriv> > v_wA = *this->write(VecDerivId::velocity());

    DataTypes::set(x_wA[i], m_x[0], m_x[1], m_x[2]);
    DataTypes::set(v_wA[i], (Real) 0.0, (Real) 0.0, (Real) 0.0);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::copyToBaseVector(defaulttype::BaseVector * dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        helper::ReadAccessor< Data<VecCoord> > vSrc = *this->read(ConstVecCoordId(src));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i = 0; i < vSrc.size(); i++)
        {
            for (unsigned int j = 0; j < coordDim; j++)
            {
                Real tmp = (Real)0.0;
                DataTypeInfo<Coord>::getValue(vSrc[i], j, tmp);
                dest->set(offset + i * coordDim + j, tmp);
            }
        }

        offset += vSrc.size() * coordDim;
    }
    else
    {
        helper::ReadAccessor< Data<VecDeriv> > vSrc = *this->read(ConstVecDerivId(src));
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

template <class DataTypes>
void MechanicalObject<DataTypes>::copyFromBaseVector(VecId dest, const defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i = 0; i < vDest.size(); i++)
        {
            for (unsigned int j = 0; j < coordDim; j++)
            {
                Real tmp;
                tmp = src->element(offset + i * coordDim + j);
                DataTypeInfo<Coord>::setValue(vDest[i], j, tmp);
            }
        }

        offset += vDest.size() * coordDim;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(VecDerivId(dest));
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();

        for (unsigned int i = 0; i < vDest.size(); i++)
        {
            for (unsigned int j = 0; j < derivDim; j++)
            {
                Real tmp;
                tmp = src->element(offset + i * derivDim + j);
                DataTypeInfo<Deriv>::setValue(vDest[i], j, tmp);
            }
        }

        offset += vDest.size() * derivDim;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addToBaseVector(defaulttype::BaseVector* dest, ConstVecId src, unsigned int &offset)
{
    if (src.type == sofa::core::V_COORD)
    {
        helper::ReadAccessor< Data<VecCoord> > vSrc = *this->read(ConstVecCoordId(src));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i = 0; i < vSrc.size(); i++)
        {
            for (unsigned int j = 0; j < coordDim; j++)
            {
                Real tmp = (Real)0.0;
                DataTypeInfo<Coord>::getValue(vSrc[i], j, tmp);
                dest->add(offset + i * coordDim + j, tmp);
            }
        }

        offset += vSrc.size() * coordDim;
    }
    else
    {
        helper::ReadAccessor< Data<VecDeriv> > vSrc = *this->read(ConstVecDerivId(src));
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();

        for (unsigned int i = 0; i < vSrc.size(); i++)
        {
            for (unsigned int j = 0; j < derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue(vSrc[i], j, tmp);
                dest->add(offset + i * derivDim + j, tmp);
            }
        }

        offset += vSrc.size() * derivDim;
    }
}


#ifndef SOFA_FLOAT
template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset );
#endif
#ifndef SOFA_DOUBLE
template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::addFromBaseVectorSameSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset );
#endif

template <class DataTypes>
void MechanicalObject<DataTypes>::addFromBaseVectorSameSize(VecId dest, const defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == sofa::core::V_COORD)
    {
        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i = 0; i < vDest.size(); i++)
        {
            for (unsigned int j = 0; j < coordDim; j++)
            {
                Real tmp = (Real)0.0;
                DataTypeInfo<Coord>::getValue(vDest[i], j, tmp);
                DataTypeInfo<Coord>::setValue(vDest[i], j, tmp + src->element(offset + i * coordDim + j));
            }
        }

        offset += vDest.size() * coordDim;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(VecDerivId(dest));
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();

        for (unsigned int i = 0; i < vDest.size(); i++)
        {
            for (unsigned int j = 0; j < derivDim; j++)
            {
                Real tmp = (Real)0.0;
                DataTypeInfo<Deriv>::getValue(vDest[i], j, tmp);
                DataTypeInfo<Deriv>::setValue(vDest[i], j, tmp + src->element(offset + i * derivDim + j));
            }
        }

        offset += vDest.size() * derivDim;
    }
}

#ifndef SOFA_FLOAT
template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::addFromBaseVectorDifferentSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset );
#endif
#ifndef SOFA_DOUBLE
template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::addFromBaseVectorDifferentSize(core::VecId dest, const defaulttype::BaseVector* src, unsigned int &offset );
#endif

template <class DataTypes>
void MechanicalObject<DataTypes>::addFromBaseVectorDifferentSize(VecId dest, const defaulttype::BaseVector* src, unsigned int &offset )
{
    if (dest.type == sofa::core::V_COORD)
    {
        helper::WriteAccessor< Data<VecCoord> > vDest = *this->write(VecCoordId(dest));
        const unsigned int coordDim = DataTypeInfo<Coord>::size();
        const unsigned int nbEntries = src->size()/coordDim;
        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<coordDim; ++j)
            {
                Real tmp = (Real)0.0;
                DataTypeInfo<Coord>::getValue(vDest[i+offset],j,tmp);
                DataTypeInfo<Coord>::setValue(vDest[i+offset],j, tmp + src->element(i*coordDim+j));
            }
        }
        offset += nbEntries;
    }
    else
    {
        helper::WriteAccessor< Data<VecDeriv> > vDest = *this->write(VecDerivId(dest));

        const unsigned int derivDim = DataTypeInfo<Deriv>::size();
        const unsigned int nbEntries = src->size()/derivDim;
        for (unsigned int i=0; i<nbEntries; i++)
        {
            for (unsigned int j=0; j<derivDim; ++j)
            {
                Real tmp = (Real)0.0;
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
    helper::WriteAccessor< Data<VecCoord> > x_wa = *this->write(VecCoordId::position());
    helper::ReadAccessor< Data<VecCoord> > xfree_ra = *this->read(ConstVecCoordId::freePosition());
    helper::ReadAccessor< Data<VecDeriv> > dx_ra =  *this->read(ConstVecDerivId::dx());

    for (unsigned int i = 0; i < xfree_ra.size(); i++)
    {
        x_wa[i] = xfree_ra[i] + dx_ra[i];
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::init()
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
    m_topology = this->getContext()->getMeshTopology();

    //helper::WriteAccessor< Data<VecCoord> > x_wA = *this->write(VecCoordId::position());
    //helper::WriteAccessor< Data<VecDeriv> > v_wA = *this->write(VecDerivId::velocity());
    Data<VecCoord>* x_wAData = this->write(VecCoordId::position());
    Data<VecDeriv>* v_wAData = this->write(VecDerivId::velocity());
    VecCoord& x_wA = *x_wAData->beginEdit();
    VecDeriv& v_wA = *v_wAData->beginEdit();

    //case if X0 has been set but not X
    if (getX0()->size() > x_wA.size())
    {
        vOp(VecId::position(), VecId::restPosition());
    }

    if (x_wA.size() != (std::size_t)vsize || v_wA.size() != (std::size_t)vsize)
    {
        // X and/or V where user-specified
        // copy the last specified velocity to all points

        const unsigned int xSize = x_wA.size();

        helper::WriteAccessor< Data<VecDeriv> > v_wA = *this->write(VecDerivId::velocity());

        if (v_wA.size() >= 1 && v_wA.size() < xSize)
        {
            unsigned int i = v_wA.size();
            Deriv v1 = v_wA[i-1];
            v_wA.resize(xSize);
            while (i < v_wA.size())
                v_wA[i++] = v1;
        }

        resize(xSize > v_wA.size() ? xSize : v_wA.size());
    }
    else if (x_wA.size() <= 1)
    {
        if (!ignoreLoader.getValue())
        {
            MeshLoader* m_loader;
            this->getContext()->get(m_loader);

            if (m_loader && m_loader->getFillMState())
            {
                int nbp = m_loader->getNbPoints();

                //std::cout<<"Setting "<<nbp<<" points from MeshLoader. " <<std::endl;

                // copy the last specified velocity to all points
                if (v_wA.size() >= 1 && v_wA.size() < (unsigned)nbp)
                {
                    unsigned int i = v_wA.size();
                    Deriv v1 = v_wA[i-1];
                    v_wA.resize(nbp);
                    while (i < v_wA.size())
                        v_wA[i++] = v1;
                }
                this->resize(nbp);

                for (int i=0; i<nbp; i++)
                {
                    x_wA[i] = Coord();
                    DataTypes::set(x_wA[i], m_loader->getPX(i), m_loader->getPY(i), m_loader->getPZ(i));
                }
            }
            else
            {
                if (m_topology != NULL && m_topology->hasPos() && m_topology->getContext() == this->getContext())
                {
                    int nbp = m_topology->getNbPoints();
                    //std::cout<<"Setting "<<nbp<<" points from topology. " << this->getName() << " topo : " << m_topology->getName() <<std::endl;
                    // copy the last specified velocity to all points
                    if (v_wA.size() >= 1 && v_wA.size() < (unsigned)nbp)
                    {
                        unsigned int i = v_wA.size();
                        Deriv v1 = v_wA[i-1];
                        v_wA.resize(nbp);
                        while (i < v_wA.size())
                            v_wA[i++] = v1;
                    }
                    this->resize(nbp);
                    for (int i=0; i<nbp; i++)
                    {
                        x_wA[i] = Coord();
                        DataTypes::set(x_wA[i], m_topology->getPX(i), m_topology->getPY(i), m_topology->getPZ(i));
                    }
                }
            }
        }
    }

    reinit();

//	*this->v0 = *v;
//	*this->v0 = v.getValue();

    // Free motion position = position
//	vOp(VecId::freePosition(), VecId::position());

    VecCoord *x0_edit = x0.beginEdit();

    // Rest position
    if (x0_edit->size() == 0)
    {
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


    if (grid)
        grid->setP0(p0);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::storeResetState()
{
    // Save initial state for reset button
    vOp(VecId::resetPosition(), VecId::position());

    //vOp(VecId::resetVelocity(), VecId::velocity());
    // we only store a resetVelocity if the velocity is not zero
    helper::ReadAccessor< Data<VecDeriv> > v = *this->read(VecDerivId::velocity());
    bool zero = true;
    for (unsigned int i=0; i<v.size(); ++i)
    {
        const Deriv& vi = v[i];
        for (unsigned int j=0; j<vi.size(); ++j)
            if (vi[j] != 0) zero = false;
        if (!zero) break;
    }
    if (!zero)
        vOp(VecId::resetVelocity(), VecId::velocity());
}

//
// Integration related methods
//

template <class DataTypes>
void MechanicalObject<DataTypes>::reset()
{
    if (!reset_position.isSet())
        return;

    vOp(VecId::position(), VecId::resetPosition());

    if (!reset_velocity.isSet())
    {
        vOp(VecId::velocity());
    }
    else
    {
        vOp(VecId::velocity(), VecId::resetVelocity());
    }

    vOp(VecId::freePosition(), VecId::position());
    vOp(VecId::freeVelocity(), VecId::velocity());
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
void MechanicalObject<DataTypes>::writeVec(ConstVecId v, std::ostream &out)
{
    switch (v.type)
    {
    case sofa::core::V_COORD:
        out << this->read(ConstVecCoordId(v))->getValue();
        break;
    case sofa::core::V_DERIV:
        out << this->read(ConstVecDerivId(v))->getValue();
        break;
    case sofa::core::V_MATDERIV:
        out << this->read(ConstMatrixDerivId(v))->getValue();
        break;
    default:
        break;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::readVec(VecId v, std::istream &in)
{
    int i = 0;

    switch (v.type)
    {
    case sofa::core::V_COORD:
    {
        Coord coord;
        helper::WriteAccessor< Data< VecCoord > > vec = *this->write(VecCoordId(v));

        while (in >> coord)
        {
            if (i >= getSize())
                resize(i+1);
            vec[i++] = coord;
        }

        break;
    }
    case sofa::core::V_DERIV:
    {
        Deriv deriv;
        helper::WriteAccessor< Data< VecDeriv > > vec = *this->write(VecDerivId(v));

        while (in >> deriv)
        {
            if (i >= getSize())
                resize(i+1);
            vec[i++] = deriv;
        }

        break;
    }
    case sofa::core::V_MATDERIV:
        //TODO
        break;
    default:
        break;
    }

    if (i < getSize())
        resize(i);
}

template <class DataTypes>
double MechanicalObject<DataTypes>::compareVec(ConstVecId v, std::istream &in)
{
    std::string ref,cur;
    getline(in, ref);

    std::ostringstream out;

    switch (v.type)
    {
    case sofa::core::V_COORD:
        out << this->read(ConstVecCoordId(v))->getValue();
        break;
    case sofa::core::V_DERIV:
        out << this->read(ConstVecDerivId(v))->getValue();
        break;
    case sofa::core::V_MATDERIV:
        out << this->read(ConstMatrixDerivId(v))->getValue();
        break;
    default:
        break;
    }

    cur = out.str();

    double error=0;
    std::istringstream compare_ref(ref);
    std::istringstream compare_cur(cur);

    Real value_ref, value_cur;
    unsigned int count=0;
    while (compare_ref >> value_ref && compare_cur >> value_cur )
    {
        error += fabs(value_ref-value_cur);
        count ++;
    }
    return error/count;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::writeState(std::ostream& out)
{
    writeX(out); out << " "; writeV(out);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::beginIntegration(Real /*dt*/)
{
    this->forceMask.activate(false);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::endIntegration(Real /*dt*/, const core::ExecParams*
#ifdef SOFA_SMP
        params
#endif
                                                )
{
    this->forceMask.clear();
    //By default the mask is disabled, the user has to enable it to benefit from the speedup
    this->forceMask.setInUse(this->useMask.getValue());

#ifdef SOFA_SMP
    if (params->execMode() == core::ExecParams::EXEC_KAAPI)
    {
        BaseObject::Task<vClear<VecDeriv,Deriv> >
        (this,**defaulttype::getShared(*this->write(VecDerivId::externalForce())),0);
    }
    else
#endif /* SOFA_SMP */
    {
        this->externalForces.beginEdit()->clear();
        this->externalForces.endEdit();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accumulateForce(const core::ExecParams*
#ifdef SOFA_SMP
        params
#endif
                                                 )
{
#ifdef SOFA_SMP
    if (params->execMode() == core::ExecParams::EXEC_KAAPI)
    {
        BaseObject::Task < vPEq2 <  VecDeriv, VecDeriv > >
        (this, **defaulttype::getShared(*this->write(VecDerivId::force())),
         **defaulttype::getShared(*this->read(ConstVecDerivId::externalForce())));
    }
    else
#endif /* SOFA_SMP */
    {
        helper::ReadAccessor< Data<VecDeriv> > extForces_rA = *this->read(ConstVecDerivId::externalForce());

        if (!extForces_rA.empty())
        {
            helper::WriteAccessor< Data<VecDeriv> > f_wA = *this->write(VecDerivId::force());

            if (!this->forceMask.isInUse())
            {
                for (unsigned int i=0; i < extForces_rA.size(); i++)
                    f_wA[i] += extForces_rA[i];
            }
            else
            {
                typedef helper::ParticleMask ParticleMask;
                const ParticleMask::InternalStorage &indices = this->forceMask.getEntries();
                ParticleMask::InternalStorage::const_iterator it;
                for (it = indices.begin(); it != indices.end(); it++)
                {
                    const int i = (*it);
                    f_wA[i] += extForces_rA[i];
                }
            }
        }
    }
}

template <class DataTypes>
Data<typename MechanicalObject<DataTypes>::VecCoord>* MechanicalObject<DataTypes>::write(VecCoordId v)
{
#ifdef SOFA_SMP_NUMA
//	if(this->getContext()->getProcessor()!=-1)
//		numa_set_preferred(this->getContext()->getProcessor()/2);
#endif

    if (v.index >= vectorsCoord.size())
    {
        vectorsCoord.resize(v.index + 1, 0);
    }

    if (vectorsCoord[v.index] == NULL)
    {
        vectorsCoord[v.index] = new Data< VecCoord >;
        if (f_reserve.getValue() > 0)
        {
            vectorsCoord[v.index]->beginEdit()->reserve(f_reserve.getValue());
            vectorsCoord[v.index]->endEdit();
        }
    }

    return vectorsCoord[v.index];
}

template <class DataTypes>
const Data<typename MechanicalObject<DataTypes>::VecCoord>* MechanicalObject<DataTypes>::read(ConstVecCoordId v) const
{
    if (v.isNull())
    {
        serr << "Accessing null VecCoord" << sendl;
    }
    if (v.index < vectorsCoord.size() && vectorsCoord[v.index] != NULL)
        return vectorsCoord[v.index];
    else
    {
        serr << "Vector " << v << " does not exist" << sendl;
        return NULL;
    }
}

template <class DataTypes>
Data<typename MechanicalObject<DataTypes>::VecDeriv>* MechanicalObject<DataTypes>::write(VecDerivId v)
{
#ifdef SOFA_SMP_NUMA
//	if(this->getContext()->getProcessor()!=-1)
//		numa_set_preferred(this->getContext()->getProcessor()/2);
#endif

    if (v.index >= vectorsDeriv.size())
    {
        vectorsDeriv.resize(v.index + 1, 0);
    }

    if (vectorsDeriv[v.index] == NULL)
    {
        vectorsDeriv[v.index] = new Data< VecDeriv >;
        if (f_reserve.getValue() > 0)
        {
            vectorsDeriv[v.index]->beginEdit()->reserve(f_reserve.getValue());
            vectorsDeriv[v.index]->endEdit();
        }
    }

    return vectorsDeriv[v.index];
}

template <class DataTypes>
const Data<typename MechanicalObject<DataTypes>::VecDeriv>* MechanicalObject<DataTypes>::read(ConstVecDerivId v) const
{
    if (v.index < vectorsDeriv.size())
        return vectorsDeriv[v.index];
    else
    {
        serr << "Vector " << v << "does not exist" << sendl;
        return NULL;
    }
}

template <class DataTypes>
Data<typename MechanicalObject<DataTypes>::MatrixDeriv>* MechanicalObject<DataTypes>::write(MatrixDerivId v)
{
#ifdef SOFA_SMP_NUMA
//	if(this->getContext()->getProcessor()!=-1)
//		numa_set_preferred(this->getContext()->getProcessor()/2);
#endif

    if (v.index >= vectorsMatrixDeriv.size())
    {
        vectorsMatrixDeriv.resize(v.index + 1, 0);
    }

    if (vectorsMatrixDeriv[v.index] == NULL)
    {
        vectorsMatrixDeriv[v.index] = new Data< MatrixDeriv >;
    }

    return vectorsMatrixDeriv[v.index];
}

template <class DataTypes>
const Data<typename MechanicalObject<DataTypes>::MatrixDeriv>* MechanicalObject<DataTypes>::read(ConstMatrixDerivId v) const
{
    if (v.index < vectorsMatrixDeriv.size())
        return vectorsMatrixDeriv[v.index];
    else
    {
        serr << "Vector " << v << "does not exist" << sendl;
        return NULL;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecCoord(unsigned int index, Data< VecCoord > *v)
{
    if (index >= vectorsCoord.size())
    {
        vectorsCoord.resize(index + 1, 0);
    }

    vectorsCoord[index] = v;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecDeriv(unsigned int index, Data< VecDeriv > *v)
{
    if (index >= vectorsDeriv.size())
    {
        vectorsDeriv.resize(index + 1, 0);
    }

    vectorsDeriv[index] = v;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecMatrixDeriv(unsigned int index, Data < MatrixDeriv > *m)
{
    if (index >= vectorsMatrixDeriv.size())
    {
        vectorsMatrixDeriv.resize(index + 1, 0);
    }

    vectorsMatrixDeriv[index] = m;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAvail(core::VecCoordId& v, const core::ExecParams* /* params */)
{
    for (unsigned int i = v.index; i < vectorsCoord.size(); ++i)
    {
        if (vectorsCoord[i] && vectorsCoord[i]->isSet())
            v.index = i+1;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAvail(core::VecDerivId& v, const core::ExecParams* /* params */)
{
    for (unsigned int i = v.index; i < vectorsDeriv.size(); ++i)
    {
        if (vectorsDeriv[i] && vectorsDeriv[i]->isSet())
            v.index = i+1;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAlloc(core::VecCoordId v, const core::ExecParams*
#ifdef SOFA_SMP
        params
#endif
                                        )
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
    if (v.index >= sofa::core::VecCoordId::V_FIRST_DYNAMIC_INDEX)
    {
        helper::WriteAccessor< Data<VecCoord> > vec = *this->write(v);

        vec.resize(vsize);

#ifdef SOFA_SMP
        if (params->execMode() == core::ExecParams::EXEC_KAAPI)
        {
            BaseObject::Task< VecInitResize < VecCoord > >(this,**defaulttype::getShared(*this->write(VecCoordId(v))), this->vsize);
        }
#endif /* SOFA_SMP */
    }

    //vOp(v); // clear vector
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAlloc(core::VecDerivId v, const core::ExecParams*
#ifdef SOFA_SMP
        params
#endif
                                        )
{
#ifdef SOFA_SMP_NUMA
    if(this->getContext()->getProcessor()!=-1)
        numa_set_preferred(this->getContext()->getProcessor()/2);
#endif

    if (v.index >= sofa::core::VecDerivId::V_FIRST_DYNAMIC_INDEX)
    {
        helper::WriteAccessor< Data<VecDeriv> > vec = *this->write(v);

        vec.resize(vsize);
#ifdef SOFA_SMP
        if (params->execMode() == core::ExecParams::EXEC_KAAPI)
        {
            BaseObject::Task < VecInitResize < VecDeriv > >(this,**defaulttype::getShared(*this->write(VecDerivId(v))), this->vsize);
        }
#endif /* SOFA_SMP */
    }

    //vOp(v); // clear vector
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vFree(VecCoordId vId, const core::ExecParams* /* params */)
{
    if (vId.index >= sofa::core::VecCoordId::V_FIRST_DYNAMIC_INDEX)
    {
        Data< VecCoord >* vec_d = this->write(vId);

        VecCoord *vec = vec_d->beginEdit();
        vec->resize(0);
        vec_d->endEdit();

        vec_d->unset();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vFree(VecDerivId vId, const core::ExecParams* /* params */)
{
    if (vId.index >= sofa::core::VecDerivId::V_FIRST_DYNAMIC_INDEX)
    {
        Data< VecDeriv >* vec_d = this->write(vId);

        VecDeriv *vec = vec_d->beginEdit();
        vec->resize(0);
        vec_d->endEdit();

        vec_d->unset();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vInit(VecCoordId vId, ConstVecCoordId vSrcId, const core::ExecParams* /* params */)
{
    Data< VecCoord >* vec_d = this->write(vId);

    if (!vec_d->isSet())
    {
        vec_d->forceSet();

        if (vSrcId != ConstVecCoordId::null())
            vOp(vId, vSrcId);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vInit(VecDerivId vId, ConstVecDerivId vSrcId, const core::ExecParams* /* params */)
{
    Data< VecDeriv >* vec_d = this->write(vId);

    if (!vec_d->isSet())
    {
        vec_d->forceSet();

        if (vSrcId != ConstVecDerivId::null())
            vOp(vId, vSrcId);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vOp(VecId v, ConstVecId a, ConstVecId b, double f, const core::ExecParams*
#ifdef SOFA_SMP
        params
#endif
                                     )
{
#ifdef SOFA_SMP
    if (params->execMode() == core::ExecParams::EXEC_KAAPI)
    {
        vOp(v, a, b, f, NULL, params);
    }
    else
#endif
    {

        if(v.isNull())
        {
            // ERROR
            serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
            return;
        }
        if (a.isNull())
        {
            if (b.isNull())
            {
                // v = 0
                if (v.type == sofa::core::V_COORD)
                {
                    helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                    vv.resize(this->vsize);
                    for (unsigned int i=0; i<vv.size(); i++)
                        vv[i] = Coord();
                }
                else
                {
                    helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                    vv.resize(this->vsize);
                    for (unsigned int i=0; i<vv.size(); i++)
                        vv[i] = Deriv();
                }
            }
            else
            {
                if (b.type != v.type)
                {
                    // ERROR
                    serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
                    return;
                }
                if (v == b)
                {
                    // v *= f
                    if (v.type == sofa::core::V_COORD)
                    {
                        helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                        for (unsigned int i=0; i<vv.size(); i++)
                            vv[i] *= (Real)f;
                    }
                    else
                    {
                        helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                        for (unsigned int i=0; i<vv.size(); i++)
                            vv[i] *= (Real)f;
                    }
                }
                else
                {
                    // v = b*f
                    if (v.type == sofa::core::V_COORD)
                    {
                        helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                        helper::ReadAccessor< Data<VecCoord> > vb = *this->read(ConstVecCoordId(b));
                        vv.resize(vb.size());
                        for (unsigned int i=0; i<vv.size(); i++)
                            vv[i] = vb[i] * (Real)f;
                    }
                    else
                    {
                        helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                        helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));
                        vv.resize(vb.size());
                        for (unsigned int i=0; i<vv.size(); i++)
                            vv[i] = vb[i] * (Real)f;
                    }
                }
            }
        }
        else
        {
            if (a.type != v.type)
            {
                // ERROR
                serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
                return;
            }
            if (b.isNull())
            {
                // v = a
                if (v.type == sofa::core::V_COORD)
                {
                    helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                    helper::ReadAccessor< Data<VecCoord> > va = *this->read(ConstVecCoordId(a));
                    vv.resize(va.size());
                    for (unsigned int i=0; i<vv.size(); i++)
                        vv[i] = va[i];
                }
                else
                {
                    helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                    helper::ReadAccessor< Data<VecDeriv> > va = *this->read(ConstVecDerivId(a));
                    vv.resize(va.size());
                    for (unsigned int i=0; i<vv.size(); i++)
                        vv[i] = va[i];
                }
            }
            else
            {
                if (v == a)
                {
                    if (f==1.0)
                    {
                        // v += b
                        if (v.type == sofa::core::V_COORD)
                        {
                            helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                            if (b.type == sofa::core::V_COORD)
                            {
                                helper::ReadAccessor< Data<VecCoord> > vb = *this->read(ConstVecCoordId(b));

                                if (vb.size() > vv.size())
                                    vv.resize(vb.size());

                                for (unsigned int i=0; i<vb.size(); i++)
                                    vv[i] += vb[i];
                            }
                            else
                            {
                                helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));

                                if (vb.size() > vv.size())
                                    vv.resize(vb.size());

                                for (unsigned int i=0; i<vb.size(); i++)
                                    vv[i] += vb[i];
                            }
                        }
                        else if (b.type == sofa::core::V_DERIV)
                        {
                            helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                            helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));

                            if (vb.size() > vv.size())
                                vv.resize(vb.size());

                            for (unsigned int i=0; i<vb.size(); i++)
                                vv[i] += vb[i];
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
                            return;
                        }
                    }
                    else
                    {
                        // v += b*f
                        if (v.type == sofa::core::V_COORD)
                        {
                            helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                            if (b.type == sofa::core::V_COORD)
                            {
                                helper::ReadAccessor< Data<VecCoord> > vb = *this->read(ConstVecCoordId(b));

                                if (vb.size() > vv.size())
                                    vv.resize(vb.size());

                                for (unsigned int i=0; i<vb.size(); i++)
                                    vv[i] += vb[i]*(Real)f;
                            }
                            else
                            {
                                helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));

                                if (vb.size() > vv.size())
                                    vv.resize(vb.size());

                                for (unsigned int i=0; i<vb.size(); i++)
                                    vv[i] += vb[i]*(Real)f;
                            }
                        }
                        else if (b.type == sofa::core::V_DERIV)
                        {
                            helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                            helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));

                            if (vb.size() > vv.size())
                                vv.resize(vb.size());

                            for (unsigned int i=0; i<vb.size(); i++)
                                vv[i] += vb[i]*(Real)f;
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
                            return;
                        }
                    }
                }
                else if (v == b)
                {
                    if (f==1.0)
                    {
                        // v += a
                        if (v.type == sofa::core::V_COORD)
                        {
                            helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                            if (a.type == sofa::core::V_COORD)
                            {
                                helper::ReadAccessor< Data<VecCoord> > va = *this->read(ConstVecCoordId(a));

                                if (va.size() > vv.size())
                                    vv.resize(va.size());

                                for (unsigned int i=0; i<va.size(); i++)
                                    vv[i] += va[i];
                            }
                            else
                            {
                                helper::ReadAccessor< Data<VecDeriv> > va = *this->read(ConstVecDerivId(a));

                                if (va.size() > vv.size())
                                    vv.resize(va.size());

                                for (unsigned int i=0; i<va.size(); i++)
                                    vv[i] += va[i];
                            }
                        }
                        else if (a.type == sofa::core::V_DERIV)
                        {
                            helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                            helper::ReadAccessor< Data<VecDeriv> > va = *this->read(ConstVecDerivId(a));

                            if (va.size() > vv.size())
                                vv.resize(va.size());

                            for (unsigned int i=0; i<va.size(); i++)
                                vv[i] += va[i];
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
                            return;
                        }
                    }
                    else
                    {
                        // v = a+v*f
                        if (v.type == sofa::core::V_COORD)
                        {
                            helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                            helper::ReadAccessor< Data<VecCoord> > va = *this->read(ConstVecCoordId(a));
                            vv.resize(va.size());
                            for (unsigned int i=0; i<vv.size(); i++)
                            {
                                vv[i] *= (Real)f;
                                vv[i] += va[i];
                            }
                        }
                        else
                        {
                            helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                            helper::ReadAccessor< Data<VecDeriv> > va = *this->read(ConstVecDerivId(a));
                            vv.resize(va.size());
                            for (unsigned int i=0; i<vv.size(); i++)
                            {
                                vv[i] *= (Real)f;
                                vv[i] += va[i];
                            }
                        }
                    }
                }
                else
                {
                    if (f==1.0)
                    {
                        // v = a+b
                        if (v.type == sofa::core::V_COORD)
                        {
                            helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                            helper::ReadAccessor< Data<VecCoord> > va = *this->read(ConstVecCoordId(a));
                            vv.resize(va.size());
                            if (b.type == sofa::core::V_COORD)
                            {
                                helper::ReadAccessor< Data<VecCoord> > vb = *this->read(ConstVecCoordId(b));
                                for (unsigned int i=0; i<vv.size(); i++)
                                {
                                    vv[i] = va[i];
                                    vv[i] += vb[i];
                                }
                            }
                            else
                            {
                                helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));
                                for (unsigned int i=0; i<vv.size(); i++)
                                {
                                    vv[i] = va[i];
                                    vv[i] += vb[i];
                                }
                            }
                        }
                        else if (b.type == sofa::core::V_DERIV)
                        {
                            helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                            helper::ReadAccessor< Data<VecDeriv> > va = *this->read(ConstVecDerivId(a));
                            helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));
                            vv.resize(va.size());
                            for (unsigned int i=0; i<vv.size(); i++)
                            {
                                vv[i] = va[i];
                                vv[i] += vb[i];
                            }
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
                            return;
                        }
                    }
                    else
                    {
                        // v = a+b*f
                        if (v.type == sofa::core::V_COORD)
                        {
                            helper::WriteAccessor< Data<VecCoord> > vv = *this->write(VecCoordId(v));
                            helper::ReadAccessor< Data<VecCoord> > va = *this->read(ConstVecCoordId(a));
                            vv.resize(va.size());
                            if (b.type == sofa::core::V_COORD)
                            {
                                helper::ReadAccessor< Data<VecCoord> > vb = *this->read(ConstVecCoordId(b));
                                for (unsigned int i=0; i<vv.size(); i++)
                                {
                                    vv[i] = va[i];
                                    vv[i] += vb[i]*(Real)f;
                                }
                            }
                            else
                            {
                                helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));
                                for (unsigned int i=0; i<vv.size(); i++)
                                {
                                    vv[i] = va[i];
                                    vv[i] += vb[i]*(Real)f;
                                }
                            }
                        }
                        else if (b.type == sofa::core::V_DERIV)
                        {
                            helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
                            helper::ReadAccessor< Data<VecDeriv> > va = *this->read(ConstVecDerivId(a));
                            helper::ReadAccessor< Data<VecDeriv> > vb = *this->read(ConstVecDerivId(b));
                            vv.resize(va.size());
                            for (unsigned int i=0; i<vv.size(); i++)
                            {
                                vv[i] = va[i];
                                vv[i] += vb[i]*(Real)f;
                            }
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
                            return;
                        }
                    }
                }
            }
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vMultiOp(const VMultiOp& ops, const core::ExecParams* /* params */)
{
    // optimize common integration case: v += a*dt, x += v*dt
    if (ops.size() == 2
        && ops[0].second.size() == 2
        && ops[0].first.getId(this) == ops[0].second[0].first.getId(this)
        && ops[0].first.getId(this).type == sofa::core::V_DERIV
        && ops[0].second[1].first.getId(this).type == sofa::core::V_DERIV
        && ops[1].second.size() == 2
        && ops[1].first.getId(this) == ops[1].second[0].first.getId(this)
        && ops[0].first.getId(this) == ops[1].second[1].first.getId(this)
        && ops[1].first.getId(this).type == sofa::core::V_COORD)
    {
        helper::ReadAccessor< Data<VecDeriv> > va = *this->read(ConstVecDerivId(ops[0].second[1].first.getId(this)));
        helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(ops[0].first.getId(this)));
        helper::WriteAccessor< Data<VecCoord> > vx = *this->write(VecCoordId(ops[1].first.getId(this)));

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
    if( v.type==sofa::core::V_DERIV)
    {
        helper::WriteAccessor< Data<VecDeriv> > vv = *this->write(VecDerivId(v));
        Real t2 = (Real)(t*t);
        for (unsigned int i=0; i<vv.size(); i++)
        {
            if( vv[i]*vv[i] < t2 )
                clear(vv[i]);
        }
    }
    else
    {
        serr<<"vThreshold does not apply to coordinate vectors"<<sendl;
    }
}

template <class DataTypes>
double MechanicalObject<DataTypes>::vDot(ConstVecId a, ConstVecId b, const core::ExecParams* /* params */)
{
    Real r = 0.0;

    if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
    {
        const VecCoord &va = this->read(ConstVecCoordId(a))->getValue();
        const VecCoord &vb = this->read(ConstVecCoordId(b))->getValue();

        for (unsigned int i=0; i<va.size(); i++)
        {
            r += va[i] * vb[i];
        }
    }
    else if (a.type == sofa::core::V_DERIV && b.type == sofa::core::V_DERIV)
    {
        const VecDeriv &va = this->read(ConstVecDerivId(a))->getValue();
        const VecDeriv &vb = this->read(ConstVecDerivId(b))->getValue();

        for (unsigned int i=0; i<va.size(); i++)
        {
            r += va[i] * vb[i];
        }
    }
    else
    {
        serr << "Invalid dot operation ("<<a<<','<<b<<")" << sendl;
    }

    return r;
}

#ifndef SOFA_SMP
template <class DataTypes>
void MechanicalObject<DataTypes>::printDOF( ConstVecId v, std::ostream& out, int firstIndex, int range) const
{
    const unsigned int size=this->getSize();
    if ((unsigned int) (abs(firstIndex)) >= size) return;
    const unsigned int first=((firstIndex>=0)?firstIndex:size+firstIndex);
    const unsigned int max=( ( (range >= 0) && ( (range+first)<size) ) ? (range+first):size);

    if( v.type==sofa::core::V_COORD)
    {
        const Data<VecCoord>* d_x = this->read(ConstVecCoordId(v));
        if (d_x == NULL) return;
        helper::ReadAccessor< Data<VecCoord> > x = *d_x;

        if (x.empty())
            return;

        for (unsigned i=first; i<max; ++i)
        {
            out << x[i];
            if (i != max-1)
                out <<" ";
        }
    }
    else if( v.type==sofa::core::V_DERIV)
    {
        const Data<VecDeriv>* d_x = this->read(ConstVecDerivId(v));
        if (d_x == NULL) return;
        helper::ReadAccessor< Data<VecDeriv> > x = *d_x;

        if (x.empty())
            return;

        for (unsigned i=first; i<max; ++i)
        {
            out << x[i];
            if (i != max-1)
                out <<" ";
        }
    }
    else
        out<<"MechanicalObject<DataTypes>::printDOF, unknown v.type = "<<v.type<<std::endl;
}
#endif

template <class DataTypes>
unsigned MechanicalObject<DataTypes>::printDOFWithElapsedTime(VecId v, unsigned count, unsigned time, std::ostream& out)
{
    if (v.type == sofa::core::V_COORD)
    {
        const Data<VecCoord>* d_x = this->read(ConstVecCoordId(v));
        if (d_x == NULL) return 0;
        helper::ReadAccessor< Data<VecCoord> > x = *d_x;

        for (unsigned i = 0; i < x.size(); ++i)
        {
            out << count + i << "\t" << time << "\t" << x[i] << std::endl;
        }
        out << std::endl << std::endl;

        return x.size();
    }
    else if (v.type == sofa::core::V_DERIV)
    {
        const Data<VecDeriv>* d_x = this->read(ConstVecDerivId(v));
        if (d_x == NULL) return 0;
        helper::ReadAccessor< Data<VecDeriv> > x = *d_x;

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
void MechanicalObject<DataTypes>::resetForce(const core::ExecParams*
#ifdef SOFA_SMP
        params
#endif
                                            )
{
#ifdef SOFA_SMP
    if (params->execMode() == core::ExecParams::EXEC_KAAPI)
    {
        BaseObject::Task< vClear<VecDeriv, Deriv> >(this, **defaulttype::getShared(*this->write(VecDerivId::force())));
    }
    else
#endif /* SOFA_SMP */
    {
        helper::WriteAccessor< Data<VecDeriv> > f = *this->write(VecDerivId::force());

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
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::resetAcc(const core::ExecParams*
#ifdef SOFA_SMP
        params
#endif
                                          )
{
#ifdef SOFA_SMP
    if (params->execMode() == core::ExecParams::EXEC_KAAPI)
    {
        BaseObject::Task< vClear<VecDeriv, Deriv> >(this, **defaulttype::getShared(*this->write(VecDerivId::dx())));
    }
    else
#endif /* SOFA_SMP */
    {
        helper::WriteAccessor< Data<VecDeriv> > a = *this->write(VecDerivId::dx());

        for (unsigned i = 0; i < a.size(); ++i)
        {
            a[i] = Deriv();
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::resetConstraint(const core::ExecParams* /* params */)
{
    Data<MatrixDeriv>& c_data = *this->write(MatrixDerivId::holonomicC());
    MatrixDeriv *c = c_data.beginEdit();
    c->clear();
    c_data.endEdit();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::renumberConstraintId(const sofa::helper::vector<unsigned>& /*renumbering*/)
{
    this->serr << "MechanicalObject<DataTypes>::renumberConstraintId not implemented in the MatrixDeriv constraint API" << this->sendl;
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

    const MatrixDeriv& constraints = c.getValue();

    unsigned int block_row = 0;
    for (indices_t::const_iterator rowIt = indices.begin(); rowIt != indices.end(); ++rowIt, ++block_row)
    {
        MatrixDerivRowConstIterator rowIterator = constraints.readLine(*rowIt);

        if (rowIterator != constraints.end())
        {
            MatrixDerivColConstIterator chunk = rowIterator.begin();
            MatrixDerivColConstIterator chunkEnd = rowIterator.end();

            for( ; chunk != chunkEnd ; chunk++)
            {
                const unsigned int column = chunk.index();
                if( blocks.find( column ) == blocks.end() )
                {
                    // nope: let's create it
                    matrix_t* mat = new matrix_t(indices.size(), dimensionDeriv);
                    blocks[column] = mat;
                }

                // now it's created no matter what \o/
                matrix_t& block = *blocks[column];

                // fill the right line of the block
                const Deriv curValue = chunk.val();

                for (unsigned int i = 0; i < dimensionDeriv; ++i)
                {
                    SReal value;
                    defaulttype::DataTypeInfo< Deriv >::getValue(curValue, i, value);
                    block.set(block_row, i, value);
                }
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
SReal MechanicalObject<DataTypes>::getConstraintJacobianTimesVecDeriv(unsigned int line, core::ConstVecId id)
{
    SReal result = 0;

    const MatrixDeriv& constraints = c.getValue();

    MatrixDerivRowConstIterator rowIterator = constraints.readLine(line);

    if (rowIterator == constraints.end())
        return 0;

    const VecDeriv *data = 0;

    // Maybe we should extend this to restvelocity
    if (id == ConstVecId::velocity())
    {
        data = &v.getValue();
    }
    else if (id == ConstVecId::dx())
    {
        data = &dx.getValue();
    }
    else
    {
        this->serr << "getConstraintJacobianTimesVecDeriv " << "NOT IMPLEMENTED for " << id.getName() << this->sendl;
        return 0;
    }

    MatrixDerivColConstIterator it = rowIterator.begin();
    MatrixDerivColConstIterator itEnd = rowIterator.end();

    while (it != itEnd)
    {
        result += it.val() * (*data)[it.index()];
        ++it;
    }

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

#ifndef SOFA_FLOAT
template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::draw();
#endif
#ifndef SOFA_DOUBLE
template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::draw();
#endif
template<>
void MechanicalObject<defaulttype::LaparoscopicRigid3Types>::draw();

template <class DataTypes>
void MechanicalObject<DataTypes>::draw()
{
    Mat<4,4, GLfloat> modelviewM;
    Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context;
    if (showIndices.getValue())
    {
        context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
        glColor3f(1.0,1.0,1.0);
        glDisable(GL_LIGHTING);
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (sceneMaxBBox - sceneMinBBox).norm() * showIndicesScale.getValue();

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

    if (showObject.getValue())
    {
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);
        const float& scale = showObjectScale.getValue();
        vector<Vector3> positions;
        for (int i = 0; i < vsize; ++i)
            positions[i] = Vector3(getPX(i), getPY(i), getPZ(i));
        simulation::getSimulation()->DrawUtility().drawPoints(positions,scale,Vec<4,float>(1.0,1.0,1.0,1.0));
        glPopAttrib();
    }
}

#ifdef SOFA_SMP
template < class DataTypes >
void MechanicalObject < DataTypes >::vOpMEq (VecId v, ConstVecId b, a1::Shared < double >*f, const core::ExecParams* /* params */)
{
    if (v.type == sofa::core::V_COORD)
    {
        if (b.type == sofa::core::V_COORD)
        {
            BaseObject::Task < vOpMinusEqualMult < DataTypes, VecCoord, VecCoord > >
            (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
             **defaulttype::getShared(*this->read(ConstVecCoordId(b))), *f);
        }
        else
        {
            BaseObject::Task < vOpMinusEqualMult < DataTypes, VecCoord, VecDeriv > >
            (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
             **defaulttype::getShared(*this->read(ConstVecDerivId(b))), *f);
        }
    }
    else if (b.type == sofa::core::V_DERIV)
    {
        BaseObject::Task < vOpMinusEqualMult < DataTypes, VecDeriv, VecDeriv > >
        (this, **defaulttype::getShared(*this->write(VecDerivId(v))),
         **defaulttype::getShared(*this->read(ConstVecDerivId(b))), *f);
        // helper::ReadAccessor< Data<VecCoord> > vSrc = *this->read(ConstVecCoordId(src));
        // helper::ReadAccessor< Data<VecCoord> > x_rA = *this->read(ConstVecCoordId::position());
    }
    else
    {
        // ERROR
        //  serr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")" << sendl;
        return;
    }
}

// template < class DataTypes >
// void MechanicalObject < DataTypes >::vOp(core::VecId v, core::ConstVecId a, core::ConstVecId b , double f, const core::ExecParams* params){ vOp(v,a,b,f,NULL, params); }

template < class DataTypes >
void MechanicalObject < DataTypes >::vOp (core::VecId v, core::ConstVecId a, core::ConstVecId b, double f, a1::Shared  < double >*fSh, const core::ExecParams* params )
{
    if (params->execMode() != core::ExecParams::EXEC_KAAPI)
    {
        vOp(v, a, b, f, params);
    }
    else
    {
#ifdef SOFA_SMP_NUMA
        if(this->getContext()->getProcessor()!=-1)
            numa_set_preferred(this->getContext()->getProcessor()/2);
#endif
        if (v.isNull ())
        {
            // ERROR
            serr << "Invalid vOp operation (" << v << ',' << a << ','
                    << b << ',' << f << ")" << sendl;
            return;
        }
        if (a.isNull ())
        {
            if (b.isNull ())
            {
                // v = 0
                if (v.type == sofa::core::V_COORD)
                {
                    BaseObject::Task < vClear < VecCoord,Coord > >
                    (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                     (unsigned) (this->vsize));

                    /*unsigned int vt=ExecutionGraph::add_operation("v=0");
                    ExecutionGraph::read_var(vt,vv);
                    ExecutionGraph::write_var(vt,vv); */
                }
                else
                {
                    BaseObject::Task < vClear < VecDeriv,Deriv > >
                    (this,**defaulttype::getShared(*this->write(VecDerivId(v))),
                     (unsigned) (this->vsize));
                    /*unsigned int vt=ExecutionGraph::add_operation("v=0");
                    ExecutionGraph::read_var(vt,vv);
                    ExecutionGraph::write_var(vt,vv);
                    vv->resize(this->this->vsize);
                    */
                }
            }
            else
            {
                if (b.type != v.type)
                {
                    // ERROR
                    serr << "Invalid vOp operation (" << v << ',' << a << ',' << b << ',' << f << ")" << sendl;
                    return;
                }
                if (v == b)
                {
                    // v *= f
                    if (v.type == sofa::core::V_COORD)
                    {

                        /*VecCoord* vv = getVecCoord(v.index);
                        unsigned int vt=ExecutionGraph::add_operation("v*=f");
                        ExecutionGraph::read_var(vt,vv);
                        ExecutionGraph::write_var(vt,vv); */
                        BaseObject::Task < vTEq < VecCoord, Real > >
                        (this,**defaulttype::getShared(*this->write(VecCoordId(v))), f);
                    }
                    else
                    {
                        /*
                        unsigned int vt=ExecutionGraph::add_operation("v*=f");
                        ExecutionGraph::read_var(vt,vv);
                        ExecutionGraph::write_var(vt,vv);
                        */
                        BaseObject::Task < vTEq < VecDeriv, Real > >
                        (this,**defaulttype::getShared(*this->write(VecDerivId(v))), f);
                    }
                }
                else
                {
                    // v = b*f
                    if (v.type == sofa::core::V_COORD)
                    {
                        BaseObject::Task < vEqBF < VecCoord,Real > >
                        (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                         **defaulttype::getShared(*this->read(ConstVecCoordId(b))), f);
                        // >(this,**getVecCoord (b.index), TODO : check order
                        // **getVecCoord (v.index), f);
                    }
                    else
                    {
                        BaseObject::Task < vEqBF < VecDeriv,Real > >
                        (this,**defaulttype::getShared(*this->write(VecDerivId(v))),
                         **defaulttype::getShared(*this->read(ConstVecDerivId(b))), f);
                        // *getVecDeriv (b.index), TODO : check order
                        // **getVecDeriv (v.index), f);
                    }
                }
            }
        }
        else
        {
            if (a.type != v.type)
            {
                // ERROR
                serr << "Invalid vOp operation (" << v << ',' << a <<
                        ',' << b << ',' << f << ")" << sendl;
                return;
            }
            if (b.isNull ())
            {
                // v = a
                if (v.type == sofa::core::V_COORD)
                {
                    BaseObject::Task < vAssign < VecCoord > >
                    (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                     **defaulttype::getShared(*this->read(ConstVecCoordId(a))));
                }
                else
                {
                    BaseObject::Task < vAssign < VecDeriv > >
                    (this,**defaulttype::getShared(*this->write(VecDerivId(v))),
                     **defaulttype::getShared(*this->read(ConstVecDerivId(a))));
                }
            }
            else
            {
                if (v == a)
                {
                    if (f == 1.0 && !fSh)
                    {
                        // v += b
                        if (v.type == sofa::core::V_COORD)
                        {

                            if (b.type == sofa::core::V_COORD)
                            {
                                BaseObject::Task < vPEq < VecCoord, VecCoord > >
                                (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(v))));
                            }
                            else
                            {
                                BaseObject::Task < vPEq < VecCoord, VecDeriv > >
                                (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(v))));
                            }
                        }
                        else if (b.type == sofa::core::V_DERIV)
                        {

                            BaseObject::Task < vPEq <  VecDeriv, VecDeriv >	>
                            (this,**defaulttype::getShared(*this->write(VecDerivId(v))),
                             **defaulttype::getShared(*this->read(ConstVecDerivId(v))));
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation (" << v <<
                                    ',' << a << ',' << b << ',' << f << ")" << sendl;
                            return;
                        }
                    }
                    else
                    {
                        // v += b*f
                        if (v.type == sofa::core::V_COORD)
                        {
                            if (b.type == sofa::core::V_COORD)
                            {
                                if (fSh)
                                {
                                    BaseObject::Task < vPEqBF < DataTypes, VecCoord, VecCoord > >
                                    (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                     **defaulttype::getShared(*this->read(ConstVecCoordId(b))), *fSh, f);
                                }
                                else
                                {
                                    BaseObject::Task < vPEqBF < DataTypes, VecCoord, VecCoord > >
                                    (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                     **defaulttype::getShared(*this->read(ConstVecCoordId(b))), f);
                                }
                            }
                            else
                            {
                                if (fSh)
                                {
                                    BaseObject::Task < vPEqBF < DataTypes, VecCoord, VecDeriv > >
                                    (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                     **defaulttype::getShared(*this->read(ConstVecDerivId(b))),*fSh, f);
                                }
                                else
                                {
                                    BaseObject::Task < vPEqBF < DataTypes, VecCoord, VecDeriv > >
                                    (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                     **defaulttype::getShared(*this->read(ConstVecDerivId(b))), f);
                                }
                            }
                        }// v.type == sofa::core::V_COORD
                        else if (b.type == sofa::core::V_DERIV)
                        {
                            if (fSh)
                            {
                                BaseObject::Task < vPEqBF < DataTypes, VecDeriv, VecDeriv > >
                                (this,**defaulttype::getShared(*this->write(VecDerivId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(b))),*fSh, f);
                            }
                            else
                            {
                                BaseObject::Task < vPEqBF < DataTypes, VecDeriv, VecDeriv > >
                                (this,**defaulttype::getShared(*this->write(VecDerivId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(b))), f);
                            }
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation (" << v <<
                                    ',' << a << ',' << b << ',' << f << ")" << sendl;
                            return;
                        }
                    }
                }
                else if (v == b)
                {
                    if (f == 1.0 && !fSh)
                    {
                        // v += a
                        if (v.type == sofa::core::V_COORD)
                        {

                            if (a.type == sofa::core::V_COORD)
                            {
                                BaseObject::Task < vPEq < VecCoord, VecCoord > >
                                (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(a))));
                            }
                            else
                            {
                                BaseObject::Task < vPEq < VecCoord, VecDeriv > >
                                (this,**defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(a))));
                            }
                        }
                        else if (a.type == sofa::core::V_DERIV)
                        {
                            BaseObject::Task< vPEq <VecDeriv, VecDeriv > >
                            (this, **defaulttype::getShared(*this->write(VecDerivId(v))),
                             **defaulttype::getShared(*this->read(ConstVecDerivId(a))));
                            // BaseObject::Task<vPEq <VecCoord,VecDeriv> >  TODO CHECK
                            // 		(this,**getVecDeriv(v.index),**getVecDeriv(a.index));
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation (" << v << ',' << a << ',' << b << ',' << f << ")" << sendl;
                            return;
                        }
                    }
                    else
                    {
                        // v = a+v*f
                        if (v.type == sofa::core::V_COORD)
                        {
                            if (fSh)
                            {
                                BaseObject::Task < vOpSumMult < DataTypes, VecCoord, VecCoord > >
                                (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(a))), *fSh, (Real) f);
                            }
                            else
                            {
                                BaseObject::Task < vOpSumMult < DataTypes, VecCoord, VecCoord > >
                                (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(a))), (Real) f);
                            }
                        }
                        else
                        {
                            if (fSh)
                            {
                                BaseObject::Task < vOpSumMult < DataTypes, VecDeriv, VecDeriv > >
                                (this, **defaulttype::getShared(*this->write(VecDerivId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(a))), *fSh, (Real) f);
                            }
                            else
                            {
                                BaseObject::Task < vOpSumMult < DataTypes, VecDeriv, VecDeriv > >
                                (this, **defaulttype::getShared(*this->write(VecDerivId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(a))), (Real) f);
                            }
                        }
                    }
                }
                else
                {
                    if (f == 1.0 && !fSh)
                    {
                        // v = a+b
                        if (v.type == sofa::core::V_COORD)
                        {
                            if (b.type == sofa::core::V_COORD)
                            {
                                BaseObject::Task < vAdd < VecCoord, VecCoord, VecCoord > >
                                (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(a))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(b))));
                            }
                            else
                            {
                                BaseObject::Task < vAdd < VecCoord, VecCoord, VecDeriv > >
                                (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(a))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(b))));
                            }
                        }
                        else if (b.type == sofa::core::V_DERIV)
                        {
                            BaseObject::Task < vAdd < VecDeriv, VecDeriv, VecDeriv > >
                            (this, **defaulttype::getShared(*this->write(VecDerivId(v))),
                             **defaulttype::getShared(*this->read(ConstVecDerivId(a))),
                             **defaulttype::getShared(*this->read(ConstVecDerivId(b))));
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation (" << v << ',' << a << ',' << b << ',' << f << ")" << sendl;
                            return;
                        }
                    }
                    else
                    {
                        // v = a+b*f
                        if (v.type == sofa::core::V_COORD)
                        {
                            if (b.type == sofa::core::V_COORD)
                            {
                                BaseObject::Task < vOpVeqAplusBmultF < DataTypes, VecCoord, VecCoord > >
                                (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(a))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(b))), (Real) f);
                            }
                            else
                            {
                                BaseObject::Task < vOpVeqAplusBmultF < DataTypes, VecCoord, VecDeriv > >
                                (this, **defaulttype::getShared(*this->write(VecCoordId(v))),
                                 **defaulttype::getShared(*this->read(ConstVecCoordId(a))),
                                 **defaulttype::getShared(*this->read(ConstVecDerivId(b))), (Real) f);
                            }
                        }
                        else if (b.type == sofa::core::V_DERIV)
                        {
                            BaseObject::Task < vOpVeqAplusBmultF < DataTypes, VecDeriv, VecDeriv > >
                            (this, **defaulttype::getShared(*this->write(VecDerivId(v))),
                             **defaulttype::getShared(*this->read(ConstVecDerivId(a))),
                             **defaulttype::getShared(*this->read(ConstVecDerivId(b))), (Real) f);
                        }
                        else
                        {
                            // ERROR
                            serr << "Invalid vOp operation (" << v << ',' << a << ',' << b << ',' << f << ")" << sendl;
                            return;
                        }
                    }
                }
            }
        }
    }
}

template < class DataTypes >
void MechanicalObject < DataTypes >::vDot ( a1::Shared  < double >*res, core::ConstVecId a,  core::ConstVecId b, const core::ExecParams* /* params */)
{
    //      double r = 0.0;
    if (a.type == sofa::core::V_COORD && b.type == sofa::core::V_COORD)
    {
        if (a.index == b.index)
        {
            BaseObject::Task < vDotOp < DataTypes, VecCoord > >
            (this, **defaulttype::getShared(*this->read(ConstVecCoordId(a))), *res);
        }
        else
        {
            BaseObject::Task < vDotOp < DataTypes, VecCoord > >
            (this, **defaulttype::getShared(*this->read(ConstVecCoordId(a))),
             **defaulttype::getShared(*this->read(ConstVecCoordId(b))), *res);
        }
    }
    else if (a.type == sofa::core::V_DERIV && b.type == sofa::core::V_DERIV)
    {
        if (a.index == b.index)
        {
            BaseObject::Task < vDotOp < DataTypes, VecDeriv > >
            (this, **defaulttype::getShared(*this->read(ConstVecDerivId(a))), *res);
        }
        else
        {
            BaseObject::Task < vDotOp < DataTypes, VecDeriv > >
            (this, **defaulttype::getShared(*this->read(ConstVecDerivId(a))),
             **defaulttype::getShared(*this->read(ConstVecDerivId(b))), *res);
        }
    }
    else
    {
        serr << "Invalid dot operation (" << a << ',' << b << ")" << sendl;
    }
    // return r;
}

//     template < class DataTypes >
//       void MechanicalObject < DataTypes >::setX (VecId v)
//     {
//       if (v.type == sofa::core::V_COORD)
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
//       if (v.type == sofa::core::V_COORD)
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
//       if (v.type == sofa::core::V_DERIV)
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
//       if (v.type == sofa::core::V_DERIV)
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
//       if (v.type == sofa::core::V_DERIV)
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
//       if (v.type == sofa::core::V_DERIV)
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
void MechanicalObject < DataTypes >::printDOF (core::ConstVecId /*v*/, std::ostream & /*out*/, int /* firstIndex */, int /* range */) const
{
    // 	if (v.type == sofa::core::V_COORD)
    // 	{
    // 		VecCoord & x = *getVecCoord (v.index);
    // 		Task < printDOFSh < VecCoord > >(this,*x);
    // 	}
    // 	else if (v.type == sofa::core::V_DERIV)
    // 	{
    // 		VecDeriv & x = *getVecDeriv (v.index);
    // 		Task < printDOFSh < VecDeriv > >(this,*x);
    // 	}
    // 	else
    // 		out << "ParallelMechanicalObject<DataTypes>::printDOF, unknown v.type = " <<
    // 			v.type << std::endl;

}
#endif /* SOFA_SMP */

/// Find mechanical particles hit by the given ray.
/// A mechanical particle is defined as a 2D or 3D, position or rigid DOF
/// Returns false if this object does not support picking
template <class DataTypes>
bool MechanicalObject<DataTypes>::pickParticles(double rayOx, double rayOy, double rayOz, double rayDx, double rayDy, double rayDz, double radius0, double dRadius,
        std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> >& particles, const core::ExecParams* /* params */)
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

} // namespace container

} // namespace component

} // namespace sofa

#endif

