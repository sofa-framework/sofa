/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MECHANICALOBJECT_INL
#define SOFA_COMPONENT_MECHANICALOBJECT_INL

#include <sofa/component/MechanicalObject.h>
#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/helper/io/MassSpringLoader.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>

#include <assert.h>
#include <iostream>

#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>


namespace sofa
{


namespace component
{

using namespace topology;
using namespace sofa::core::componentmodel::topology;

using namespace sofa::defaulttype;
template <class DataTypes>
MechanicalObject<DataTypes>::MechanicalObject()
    : x(new VecCoord), v(new VecDeriv), f(new VecDeriv), dx(new VecDeriv), x0(new VecCoord),reset_position(NULL), v0(NULL), xfree(new VecCoord), vfree(new VecDeriv), c(new VecConst), vsize(0), m_gnuplotFileX(NULL), m_gnuplotFileV(NULL)
    , f_X ( new XDataPtr<DataTypes>(&x,  "position coordinates of the degrees of freedom") )
    , f_V ( new VDataPtr<DataTypes>(&v,  "velocity coordinates of the degrees of freedom") )
    , f_F ( new VDataPtr<DataTypes>(&f,  "f vector of the degrees of freedom") )
    , f_Dx ( new VDataPtr<DataTypes>(&dx,  "dx vector of the degrees of freedom") )
    , f_Xfree ( new XDataPtr<DataTypes>(&xfree,  "free position coordinates of the degrees of freedom") )
    , f_Vfree ( new VDataPtr<DataTypes>(&vfree,  "free velocity coordinates of the degrees of freedom") )
    , f_X0( new XDataPtr<DataTypes>(&x0, "rest position coordinates of the degrees of freedom") )

{
    //HACK
    if (!restScale.isSet())
        restScale.setValue(1);
    initialized = false;
    this->addField(f_X, "position");
    this->addField(f_V, "velocity");
    this->addField(f_F, "force");
    this->addField(f_Dx, "derivX");
    this->addField(f_Xfree, "free_position");
    this->addField(f_Vfree, "free_velocity");
    this->addField(f_X0,"rest_position");


    f_X->init();
    f_V->init();
    f_F->init();
    f_Dx->init();
    f_Xfree->init();
    f_Vfree->init();
    f_X0->init();



    restScale = this->initData(&restScale, (SReal)1.0, "restScale","optional scaling of rest position coordinates (to simulated pre-existing internal tension)");
    translation=this->initData(&translation, Vector3(), "translation", "Translation of the DOFs");
    rotation=this->initData(&rotation, Vector3(), "rotation", "Rotation of the DOFs");
    scale=this->initData(&scale, (SReal)1.0, "scale", "Scale of the DOFs");
    filename=this->initData(&filename, std::string(""), "filename", "File corresponding to the Mechanical Object", false);

    /*    x = new VecCoord;
      v = new VecDeriv;*/
    internalForces = f; // = new VecDeriv;
    externalForces = new VecDeriv;
    //dx = new VecDeriv;

    // default size is 1
    resize(1);
    setVecCoord(VecId::position().index, this->x);
    setVecDeriv(VecId::velocity().index, this->v);
    setVecDeriv(VecId::force().index, this->f);
    setVecDeriv(VecId::dx().index, this->dx);
    setVecCoord(VecId::restPosition().index, this->x0);
    setVecCoord(VecId::freePosition().index, this->xfree);
    setVecDeriv(VecId::freeVelocity().index, this->vfree);
    /*    cerr<<"MechanicalObject<DataTypes>::MechanicalObject, x.size() = "<<x->size()<<endl;
      cerr<<"MechanicalObject<DataTypes>::MechanicalObject, v.size() = "<<v->size()<<endl;*/
}

template <class DataTypes>
void MechanicalObject<DataTypes>::initGnuplot(const std::string path)
{
    if( !this->getName().empty() )
    {
        if (m_gnuplotFileX != NULL) delete m_gnuplotFileX;
        if (m_gnuplotFileV != NULL) delete m_gnuplotFileV;
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
MechanicalObject<DataTypes>&
MechanicalObject<DataTypes>::operator = (const MechanicalObject& obj)
{
    resize( obj.getSize() );
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
        load(filename.getValue().c_str());
        filename.setValue(std::string("")); //clear the field filename: When we save the scene, we don't need anymore the filename
    }

    unsigned int size0 = getX()->size();
    Inherited::parse(arg);
    if (arg->getAttribute("size")!=NULL)
    {
        resize( atoi(arg->getAttribute("size")) );
    }
    else if (getX()->size() != size0)
        resize( getX()->size() );

    //obj->parseTransform(arg);
    if (arg->getAttribute("scale")!=NULL)
    {
        scale.setValue((SReal)atof(arg->getAttribute("scale")));
        //applyScale(scale.getValue());
    }
    if (arg->getAttribute("rx")!=NULL || arg->getAttribute("ry")!=NULL || arg->getAttribute("rz")!=NULL)
    {
        rotation.setValue(Vector3((SReal)(atof(arg->getAttribute("rx","0.0"))),(SReal)(atof(arg->getAttribute("ry","0.0"))),(SReal)(atof(arg->getAttribute("rz","0.0"))))*3.141592653/180.0);

        Quaternion q=helper::Quater<SReal>::createFromRotationVector( Vec<3,SReal>(rotation.getValue()[0],rotation.getValue()[1],rotation.getValue()[2]));
        //applyRotation(q);
    }
    if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
    {
        translation.setValue(Vector3((Real)atof(arg->getAttribute("dx","0.0")), (Real)atof(arg->getAttribute("dy","0.0")), (Real)atof(arg->getAttribute("dz","0.0"))));
        //applyTranslation(translation.getValue()[0],translation.getValue()[1],translation.getValue()[2]);
    }

}

template <class DataTypes>
MechanicalObject<DataTypes>::~MechanicalObject()
{
    delete externalForces;
    if (reset_position!=NULL)
        delete reset_position;
    //if (x0!=NULL)
    //  delete x0;
    if (v0!=NULL)
        delete v0;
    for (unsigned int i=0; i<vectorsCoord.size(); i++)
        if (vectorsCoord[i]!=NULL)
            delete vectorsCoord[i];
    for (unsigned int i=0; i<vectorsDeriv.size(); i++)
        if (vectorsDeriv[i]!=NULL)
            delete vectorsDeriv[i];
    if( m_gnuplotFileV!=NULL )
        delete m_gnuplotFileV;
    if( m_gnuplotFileX!=NULL )
        delete m_gnuplotFileX;
}


template <class DataTypes>
void MechanicalObject<DataTypes>::handleStateChange()
{

    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(this->getContext()->getMainTopology());

    std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=topology->firstStateChange();
    std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=topology->lastStateChange();

    while( itBegin != itEnd )
    {

        TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch( changeType )
        {

        case core::componentmodel::topology::POINTSADDED:
        {
            unsigned int nbPoints = ( dynamic_cast< const PointsAdded * >( *itBegin ) )->getNbAddedVertices();
            sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors = ( dynamic_cast< const PointsAdded * >( *itBegin ) )->ancestorsList;
            sofa::helper::vector< sofa::helper::vector< double       > > coefs     = ( dynamic_cast< const PointsAdded * >( *itBegin ) )->coefs;

            if ( ancestors != (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0 )
            {
                unsigned int prevSizeMechObj = getSize();
                resize( prevSizeMechObj + nbPoints );

                sofa::helper::vector< sofa::helper::vector< double > > coefs2;
                coefs2.resize(ancestors.size());

                for (unsigned int i = 0; i < ancestors.size(); ++i)
                {
                    coefs2[i].resize(ancestors[i].size());

                    for (unsigned int j = 0; j < ancestors[i].size(); ++j)
                    {
                        // constructng default coefs if none were defined
                        if (coefs == (const sofa::helper::vector< sofa::helper::vector< double > >)0 || coefs[i].size() == 0)
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
            break;
        }
        case core::componentmodel::topology::POINTSREMOVED:
        {
            const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const PointsRemoved * >( *itBegin ) )->getArray();

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
        case core::componentmodel::topology::POINTSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> &tab = ( dynamic_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();

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
    if ((*externalForces).size() > maxIndex)
        (*externalForces)[outputIndex] = (*externalForces)[inputIndex];

    // Note: the following assumes that topological changes won't be reset
    if (reset_position != NULL && (*reset_position).size() > maxIndex)
        (*reset_position)[outputIndex] = (*reset_position)[inputIndex];

    // temporary state vectors
    unsigned int i;
    for (i=0; i<vectorsCoord.size(); i++)
    {
        if(vectorsCoord[i] != NULL)
        {
            VecCoord& vector = *vectorsCoord[i];
            if (vector.size() > maxIndex)
                vector[outputIndex]=vector[inputIndex];
        }
    }
    for ( i=0; i<vectorsDeriv.size(); i++)
    {
        if(vectorsDeriv[i] != NULL)
        {
            VecDeriv& vector = *vectorsDeriv[i];
            if (vector.size() > maxIndex)
                vector[outputIndex]=vector[inputIndex];
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
    if ((*externalForces).size() > maxIndex)
    {
        tmp2 = (*externalForces)[idx1];
        (*externalForces)[idx1] = (*externalForces)[idx2];
        (*externalForces)[idx2] = tmp2;
    }

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
            VecCoord& vector = *vectorsCoord[i];
            if(vector.size() > maxIndex)
            {
                tmp = vector[idx1];
                vector[idx1] = vector[idx2];
                vector[idx2] = tmp;
            }
        }
    }
    for (i=0; i<vectorsDeriv.size(); i++)
    {
        if(vectorsDeriv[i] != NULL)
        {
            VecDeriv& vector = *vectorsDeriv[i];
            if(vector.size() > maxIndex)
            {
                tmp2 = vector[idx1];
                vector[idx1] = vector[idx2];
                vector[idx2] = tmp2;
            }
        }
    }
}

template<class V>
void renumber(V* v, V* tmp, const sofa::helper::vector< unsigned int > &index )
{
    if (v==NULL) return;
    if (v->empty()) return;
    *tmp = *v;
    for (unsigned int i = 0; i < v->size(); ++i)
        (*v)[i] = (*tmp)[ index[i] ];
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
    renumber(externalForces, &dtmp, index);
    // Note: the following assumes that topological changes won't be reset
    renumber(reset_position, &ctmp, index);
    for (unsigned int i = 0; i < vectorsCoord.size(); ++i)
        renumber(vectorsCoord[i], &ctmp, index);
    for (unsigned int i = 0; i < vectorsDeriv.size(); ++i)
        renumber(vectorsDeriv[i], &dtmp, index);
}



template <class DataTypes>
void MechanicalObject<DataTypes>::resize(const int size)
{
    (*x).resize(size);
    if (initialized && x0!=NULL)
        (*x0).resize(size);
    (*v).resize(size);
    if (v0!=NULL)
        (*v0).resize(size);
    (*f).resize(size);
    (*dx).resize(size);
    (*xfree).resize(size);
    (*vfree).resize(size);
    if (externalForces->size()>0)
        externalForces->resize(size);
    internalForces->resize(size);
    // Note: the following assumes that topological changes won't be reset
    if (reset_position!=NULL)
        (*reset_position).resize(size);

    //if (size!=vsize)
    {
        vsize=size;
        for (unsigned int i=0; i<vectorsCoord.size(); i++)
            if (vectorsCoord[i]!=NULL && vectorsCoord[i]->size()!=0)
                vectorsCoord[i]->resize(size);
        for (unsigned int i=0; i<vectorsDeriv.size(); i++)
            if (vectorsDeriv[i]!=NULL && vectorsDeriv[i]->size()!=0)
                vectorsDeriv[i]->resize(size);
    }
}



template <class DataTypes>
void MechanicalObject<DataTypes>::applyTranslation (const double dx,const double dy,const double dz)
{
    VecCoord& x = *this->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        DataTypes::add
        (x[i],dx,dy,dz);
    }
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
    //TODO: special case of rigid bodies. need to update the orientation
}

#ifndef SOFA_FLOAT
template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::applyRotation (const defaulttype::Quat q);
template <>
bool MechanicalObject<Vec1dTypes>::addBBox(double* minBBox, double* maxBBox);
#endif
#ifndef SOFA_Real
template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::applyRotation (const defaulttype::Quat q);
template <>
bool MechanicalObject<Vec1fTypes>::addBBox(double* minBBox, double* maxBBox);
#endif

template <class DataTypes>
void MechanicalObject<DataTypes>::applyScale(const double s)
{
    VecCoord& x = *this->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        x[i] *= (Real)s;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::getIndicesInSpace(sofa::helper::vector<unsigned>& indices, Real xmin, Real xmax, Real ymin, Real ymax, Real zmin, Real zmax) const
{
    const VecCoord& X = *getX();
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
        (*v0)[i]=Deriv();
        for (j=0; j<ancestors.size(); ++j)
            (*v0)[i]+=(*v0)[ancestors[j]]*(Real)coefs[j];
    }
    // Note: the following assumes that topological changes won't be reset
    if (reset_position != NULL)
    {
        (*reset_position)[i]=Coord();
        for (j=0; j<ancestors.size(); ++j)
            (*reset_position)[i]+=(*reset_position)[ancestors[j]]*(Real)coefs[j];
    }
    if (externalForces->size()>0)
    {
        (*externalForces)[i]=Deriv();
        for (j=0; j<ancestors.size(); ++j)
            (*externalForces)[i]+=(*externalForces)[ancestors[j]]*(Real)coefs[j];
    }
    for (unsigned int k=0; k<vectorsCoord.size(); k++)
    {
        if (vectorsCoord[k]!=NULL && vectorsCoord[k]->size()!=0)
        {
            (*vectorsCoord[k])[i]=Coord();
            for (j=0; j<ancestors.size(); ++j)
            {
                (*vectorsCoord[k])[i]+= (*vectorsCoord[k])[ancestors[j]]*(Real)coefs[j];
            }
        }
    }
    for (unsigned int k=0; k<vectorsDeriv.size(); k++)
    {
        if (vectorsDeriv[k]!=NULL && vectorsDeriv[k]->size()!=0)
        {
            (*vectorsDeriv[k])[i]=Deriv();
            for (j=0; j<ancestors.size(); ++j)
            {
                (*vectorsDeriv[k])[i]+= (*vectorsDeriv[k])[ancestors[j]]*(Real)coefs[j];
            }
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::computeNewPoint( const unsigned int i, const sofa::helper::vector< double >& m_x)
{
    this->resize(i+1);

    DataTypes::set((*getX())[i], m_x[0]*scale.getValue()+translation.getValue()[0], m_x[1]*scale.getValue()+translation.getValue()[1], m_x[2]*scale.getValue()+translation.getValue()[2]);
    DataTypes::set((*getXfree())[i], m_x[0]*scale.getValue()+translation.getValue()[0], m_x[1]*scale.getValue()+translation.getValue()[1], m_x[2]*scale.getValue()+translation.getValue()[2]);
    DataTypes::set((*getX0())[i], m_x[0]*scale.getValue()*restScale.getValue()+translation.getValue()[0], m_x[1]*scale.getValue()*restScale.getValue()+translation.getValue()[1], m_x[2]*scale.getValue()*restScale.getValue()+translation.getValue()[2]);

    if (reset_position != NULL)
        DataTypes::set((*reset_position)[i], m_x[0]*scale.getValue()+translation.getValue()[0], m_x[1]*scale.getValue()+translation.getValue()[1], m_x[2]*scale.getValue()+translation.getValue()[2]);
}

// Force the position of a point (and force its velocity to zero value)
template <class DataTypes>
void MechanicalObject<DataTypes>::forcePointPosition( const unsigned int i, const sofa::helper::vector< double >& m_x)
{
    DataTypes::set((*getX())[i], m_x[0], m_x[1], m_x[2]);
    DataTypes::set((*getV())[i], (Real) 0.0, (Real) 0.0, (Real) 0.0);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::contributeToMatrixDimension(unsigned int * const nbRow, unsigned int * const nbCol)
{
    if (v->size() != 0)
    {
        (*nbRow) += v->size() * DataTypeInfo<Deriv>::size();
        (*nbCol) = *nbRow;
    }
}


template <class DataTypes>
void MechanicalObject<DataTypes>::setOffset(unsigned int &offset)
{
    if (v->size() != 0)
    {
        offset += v->size() * DataTypeInfo<Deriv>::size();
    }
}


template <class DataTypes>
void MechanicalObject<DataTypes>::loadInBaseVector(defaulttype::BaseVector * dest, VecId src, unsigned int &offset)
{
    if (src.type == VecId::V_COORD)
    {
        const VecCoord* vSrc = getVecCoord(src.index);
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vSrc->size(); i++)
            for (unsigned int j=0; j<coordDim; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue((*vSrc)[i],j,tmp);
                dest->set(offset + i * coordDim + j, tmp);
            }
        offset += vSrc->size() * coordDim;
    }
    else
    {
        const VecDeriv* vSrc = getVecDeriv(src.index);
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();

        for (unsigned int i=0; i<vSrc->size(); i++)
            for (unsigned int j=0; j<derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue((*vSrc)[i],j,tmp);
                dest->set(offset + i * derivDim + j, tmp);
            }
        offset += vSrc->size() * derivDim;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addBaseVectorToState(VecId dest, defaulttype::BaseVector *src, unsigned int &offset)
{
    if (dest.type == VecId::V_COORD)
    {
        VecCoord* vDest = getVecCoord(dest.index);
        const unsigned int coordDim = DataTypeInfo<Coord>::size();

        for (unsigned int i=0; i<vDest->size(); i++)
        {
            for (unsigned int j=0; j<coordDim; j++)
            {
                Real tmp;
                DataTypeInfo<Coord>::getValue((*vDest)[i],j,tmp);
                DataTypeInfo<Coord>::setValue((*vDest)[i], j, tmp + src->element(offset + i * coordDim + j));
            }
        }

        offset += vDest->size() * coordDim;
    }
    else
    {
        VecDeriv* vDest = getVecDeriv(dest.index);
        const unsigned int derivDim = DataTypeInfo<Deriv>::size();

        for (unsigned int i=0; i<vDest->size(); i++)
            for (unsigned int j=0; j<derivDim; j++)
            {
                Real tmp;
                DataTypeInfo<Deriv>::getValue((*vDest)[i],j,tmp);
                DataTypeInfo<Deriv>::setValue((*vDest)[i], j, tmp + src->element(offset + i * derivDim + j));
            }

        offset += vDest->size() * derivDim;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addDxToCollisionModel()
{
    for (unsigned int i=0; i < this->xfree->size(); i++)
        (*this->x)[i] = (*this->xfree)[i] + (*this->dx)[i];
}


template <class DataTypes>
void MechanicalObject<DataTypes>::init()
{


    f_X->beginEdit();
    f_V->beginEdit();
    f_F->beginEdit();
    f_Dx->beginEdit();
    f_Xfree->beginEdit();
    f_Vfree->beginEdit();
    f_X0->beginEdit();



    if (getX()->size() != (std::size_t)vsize || getV()->size() != (std::size_t)vsize)
    {
        // X and/or V where user-specified
        resize(getX()->size()>getV()->size()?getX()->size():getV()->size());
    }
    else if (getX()->size() <= 1)
    {
        core::componentmodel::topology::Topology* topo = dynamic_cast<core::componentmodel::topology::Topology*>(this->getContext()->getTopology());
        if (topo!=NULL && topo->hasPos() && topo->getContext() == this->getContext())
        {
            int nbp = topo->getNbPoints();
            //std::cout<<"Setting "<<nbp<<" points from topology. " << this->getName() << " topo : " << topo->getName() <<std::endl;
            this->resize(nbp);
            for (int i=0; i<nbp; i++)
            {
                (*getX())[i] = Coord();
                DataTypes::set((*getX())[i], topo->getPX(i), topo->getPY(i), topo->getPZ(i));
            }

        }
    }

    reinit();

    if (v0 == NULL) this->v0 = new VecDeriv;
    *this->v0 = *v;
    // free position = position
    *this->xfree = *x;

    //Rest position

    if (x0->size() == 0)
    {
        *x0 = *x;
        if (restScale.getValue() != (Real)1)
        {
            Real s = (Real)restScale.getValue();
            for (unsigned int i=0; i<x0->size(); i++)
                (*x0)[i] *= s;
        }
    }

    initialized = true;

}



template <class DataTypes>
void MechanicalObject<DataTypes>::reinit()
{
    Vector3 p0;
    sofa::component::topology::RegularGridTopology *grid; this->getContext()->get(grid, BaseContext::Local);
    if (grid) p0 = grid->getP0();

    if (scale.getValue() != (SReal)1.0)
    {
        this->applyScale(scale.getValue());
        p0 *= scale.getValue();
    }

    if (rotation.getValue()[0]!=0.0 || rotation.getValue()[1]!=0.0 || rotation.getValue()[2]!=0.0)
    {
        Quaternion q=helper::Quater<SReal>::createFromRotationVector( Vec<3,SReal>(rotation.getValue()[0],rotation.getValue()[1],rotation.getValue()[2]));
        this->applyRotation(q);
//  	p0 = q.rotate(p0);
    }

    if (translation.getValue()[0]!=0.0 || translation.getValue()[1]!=0.0 || translation.getValue()[2]!=0.0)
    {
        this->applyTranslation( translation.getValue()[0],translation.getValue()[1],translation.getValue()[2]);
        p0 += translation.getValue();
    }


    if (grid) grid->setP0(p0);

    translation.setValue(Vector3());
    rotation.setValue(Vector3());
    scale.setValue((SReal)1.0);
}
template <class DataTypes>
void MechanicalObject<DataTypes>::storeResetState()
{
    // Save initial state for reset button
    if (reset_position == NULL) this->reset_position = new VecCoord;
    *this->reset_position = *x;
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
    in >> *getX();
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
    in >> *getX();
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
void MechanicalObject<DataTypes>::writeState( std::ostream& out )
{
    writeX(out); out << " "; writeV(out);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::beginIntegration(Real /*dt*/)
{
    this->f = this->internalForces;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::endIntegration(Real /*dt*/)
{
    this->f = this->externalForces;
    this->externalForces->clear();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accumulateForce()
{
    if (!this->externalForces->empty())
    {
        for (unsigned int i=0; i < this->externalForces->size(); i++)
            (*this->f)[i] += (*this->externalForces)[i];
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecCoord(unsigned int index, VecCoord* v)
{
    if (index>=vectorsCoord.size())
        vectorsCoord.resize(index+1);
    vectorsCoord[index] = v;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecDeriv(unsigned int index, VecDeriv* v)
{
    if (index>=vectorsDeriv.size())
        vectorsDeriv.resize(index+1);
    vectorsDeriv[index] = v;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVecConst(unsigned int index, VecConst* v)
{
    if (index>=vectorsConst.size())
        vectorsConst.resize(index+1);
    vectorsConst[index] = v;
}


template<class DataTypes>
typename MechanicalObject<DataTypes>::VecCoord* MechanicalObject<DataTypes>::getVecCoord(unsigned int index)
{
    if (index>=vectorsCoord.size())
        vectorsCoord.resize(index+1);
    if (vectorsCoord[index]==NULL)
        vectorsCoord[index] = new VecCoord;
    return vectorsCoord[index];
}

template<class DataTypes>
const typename MechanicalObject<DataTypes>::VecCoord* MechanicalObject<DataTypes>::getVecCoord(unsigned int index) const
{
    if (index>=vectorsCoord.size())
        return NULL;
    if (vectorsCoord[index]==NULL)
        return NULL;
    return vectorsCoord[index];
}

template<class DataTypes>
typename MechanicalObject<DataTypes>::VecDeriv* MechanicalObject<DataTypes>::getVecDeriv(unsigned int index)
{
    if (index>=vectorsDeriv.size())
        vectorsDeriv.resize(index+1);
    if (vectorsDeriv[index]==NULL)
        vectorsDeriv[index] = new VecDeriv;

    return vectorsDeriv[index];
}

template<class DataTypes>
const typename MechanicalObject<DataTypes>::VecDeriv* MechanicalObject<DataTypes>::getVecDeriv(unsigned int index) const
{
    if (index>=vectorsDeriv.size())
        return NULL;
    if (vectorsDeriv[index]==NULL)
        return NULL;

    return vectorsDeriv[index];
}

template<class DataTypes>
typename MechanicalObject<DataTypes>::VecConst* MechanicalObject<DataTypes>::getVecConst(unsigned int index)
{
    if (index>=vectorsConst.size())
        vectorsConst.resize(index+1);
    if (vectorsConst[index]==NULL)
        vectorsConst[index] = new VecConst;

    return vectorsConst[index];
}

template<class DataTypes>
const typename MechanicalObject<DataTypes>::VecConst* MechanicalObject<DataTypes>::getVecConst(unsigned int index) const
{
    if (index>=vectorsConst.size())
        return NULL;
    if (vectorsConst[index]==NULL)
        return NULL;

    return vectorsConst[index];
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAvail(VecId& v)
{
    if (v.type == VecId::V_COORD)
    {
        for (unsigned int i=v.index; i < vectorsCoord.size(); ++i)
            if (vectorsCoord[i] && ! vectorsCoord[i]->empty())
                v.index = i+1;
    }
    else if (v.type == VecId::V_DERIV)
    {
        for (unsigned int i=v.index; i < vectorsDeriv.size(); ++i)
            if (vectorsDeriv[i] != NULL && ! (*vectorsDeriv[i]).empty())
                v.index = i+1;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAlloc(VecId v)
{
    if (v.type == VecId::V_COORD && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = getVecCoord(v.index);
        vec->resize(vsize);
    }
    else if (v.type == VecId::V_DERIV && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecDeriv* vec = getVecDeriv(v.index);
        vec->resize(vsize);
    }
    else
    {
        std::cerr << "Invalid alloc operation ("<<v<<")\n";
        return;
    }
    //vOp(v); // clear vector
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vFree(VecId v)
{
    if (v.type == VecId::V_COORD && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = getVecCoord(v.index);
        vec->resize(0);
    }
    else if (v.type == VecId::V_DERIV && v.index >= VecId::V_FIRST_DYNAMIC_INDEX)
    {
        VecDeriv* vec = getVecDeriv(v.index);
        vec->resize(0);
    }
    else
    {
        std::cerr << "Invalid free operation ("<<v<<")\n";
        return;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vOp(VecId v, VecId a, VecId b, double f)
{
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
        VecDeriv* vv = getVecDeriv(v.index);
        Real t2 = (Real)(t*t);
        for (unsigned int i=0; i<vv->size(); i++)
        {
            if( (*vv)[i]*(*vv)[i] < t2 )
                clear((*vv)[i]);
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
void MechanicalObject<DataTypes>::setX(VecId v)
{
    if (v.type == VecId::V_COORD)
    {
        this->x = getVecCoord(v.index);
    }
    else
    {
        std::cerr << "Invalid setX operation ("<<v<<")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setXfree(VecId v)
{
    if (v.type == VecId::V_COORD)
    {
        this->xfree = getVecCoord(v.index);
    }
    else
    {
        std::cerr << "Invalid setXfree operation ("<<v<<")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setVfree(VecId v)
{
    if (v.type == VecId::V_DERIV)
    {
        this->vfree = getVecDeriv(v.index);
    }
    else
    {
        std::cerr << "Invalid setVfree operation ("<<v<<")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setV(VecId v)
{
    if (v.type == VecId::V_DERIV)
    {
        this->v = getVecDeriv(v.index);
    }
    else
    {
        std::cerr << "Invalid setV operation ("<<v<<")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setF(VecId v)
{
    if (v.type == VecId::V_DERIV)
    {
        this->f = getVecDeriv(v.index);
    }
    else
    {
        std::cerr << "Invalid setF operation ("<<v<<")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setDx(VecId v)
{
    if (v.type == VecId::V_DERIV)
    {
        this->dx = getVecDeriv(v.index);
    }
    else
    {
        std::cerr << "Invalid setDx operation ("<<v<<")\n";
    }
}


template <class DataTypes>
void MechanicalObject<DataTypes>::setC(VecId v)
{
    if (v.type == VecId::V_CONST)
    {
        this->c = getVecConst(v.index);
    }
    else
    {
        std::cerr << "Invalid setDx operation ("<<v<<")\n";
    }
}


template <class DataTypes>
void MechanicalObject<DataTypes>::printDOF( VecId v, std::ostream& out)
{
    if( v.type==VecId::V_COORD )
    {
        VecCoord& x= *getVecCoord(v.index);
        for( unsigned i=0; i<x.size(); ++i )
            out<<x[i]<<" ";
    }
    else if( v.type==VecId::V_DERIV )
    {
        VecDeriv& x= *getVecDeriv(v.index);
        for( unsigned i=0; i<x.size(); ++i )
            out<<x[i]<<" ";
    }
    else
        out<<"MechanicalObject<DataTypes>::printDOF, unknown v.type = "<<v.type<<std::endl;
}


template <class DataTypes>
unsigned MechanicalObject<DataTypes>::printDOFWithElapsedTime( VecId v,unsigned count, unsigned time, std::ostream& out)
{

    if( v.type==VecId::V_COORD )
    {
        VecCoord& x= *getVecCoord(v.index);

        for( unsigned i=0; i<x.size(); ++i )
        {
            out<<count+i<<"\t"<<time<<"\t"<<x[i]<<std::endl;
        }
        out<<std::endl<<std::endl;
        return x.size();
    }
    else if( v.type==VecId::V_DERIV )
    {
        VecDeriv& x= *getVecDeriv(v.index);
        for( unsigned i=0; i<x.size(); ++i )
            out<<count+i<<"\t"<<time<<"\t"<<x[i]<<std::endl;
        out<<std::endl<<std::endl;

        return x.size();
    }
    else
        out<<"MechanicalObject<DataTypes>::printDOFWithElapsedTime, unknown v.type = "<<v.type<<std::endl;

    return 0;
}


template <class DataTypes>
void MechanicalObject<DataTypes>::resetForce()
{
    VecDeriv& f= *getF();
    for( unsigned i=0; i<f.size(); ++i )
        f[i] = Deriv();
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
bool MechanicalObject<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *getX();
    if (x.size() <= 0) return false;
    Real p[3] = {0,0,0};
    for (unsigned int i=0; i<x.size(); i++)
    {
        DataTypes::get(p[0], p[1], p[2], x[i]);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}

//
// Template specializations



} // namespace component

} // namespace sofa

#endif

