#ifndef SOFA_CORE_MECHANICALOBJECT_INL
#define SOFA_CORE_MECHANICALOBJECT_INL

#include "MechanicalObject.h"
#include "Topology.h"
#include "Encoding.inl"
#include <assert.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Core
{

template <class DataTypes>
MechanicalObject<DataTypes>::MechanicalObject()
    : x(new VecCoord), v(new VecDeriv), x0(NULL), v0(NULL), vsize(0)
    , f_X( new XField<DataTypes>(&x, "position coordinates ot the degrees of freedom") )
    , f_V( new VField<DataTypes>(&v, "velocity coordinates ot the degrees of freedom") )
{
    this->addField(f_X, "position");
    f_X->beginEdit();
    this->addField(f_V, "velocity");
    f_V->beginEdit();
    /*    x = new VecCoord;
        v = new VecDeriv;*/
    internalForces = f = new VecDeriv;
    externalForces = new VecDeriv;
    dx = new VecDeriv;
    // default size is 1
    resize(1);
    setVecCoord(VecId::position().index, this->x);
    setVecDeriv(VecId::velocity().index, this->v);
    setVecDeriv(VecId::force().index, this->f);
    setVecDeriv(VecId::dx().index, this->dx);
    translation[0]=0.0;
    translation[1]=0.0;
    translation[2]=0.0;
    scale = 1.0;
    /*    cerr<<"MechanicalObject<DataTypes>::MechanicalObject, x.size() = "<<x->size()<<endl;
        cerr<<"MechanicalObject<DataTypes>::MechanicalObject, v.size() = "<<v->size()<<endl;*/
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

template <class DataTypes>
void MechanicalObject<DataTypes>::parseFields ( const std::map<std::string,std::string*>& str )
{
    Inherited::parseFields(str);
    resize( getX()->size() );
    //cerr<<"MechanicalObject<DataTypes>::parseFields, resized to "<<getX()->size()<<endl;
}

template <class DataTypes>
MechanicalObject<DataTypes>::~MechanicalObject()
{
    delete externalForces;
    if (x0!=NULL)
        delete x0;
    if (v0!=NULL)
        delete v0;
    for (unsigned int i=0; i<vectorsCoord.size(); i++)
        if (vectorsCoord[i]!=NULL)
            delete vectorsCoord[i];
    for (unsigned int i=0; i<vectorsDeriv.size(); i++)
        if (vectorsDeriv[i]!=NULL)
            delete vectorsDeriv[i];
}


template <class DataTypes>
void MechanicalObject<DataTypes>::replaceValue (const int inputIndex, const int outputIndex)
{
    // standard state vectors
    (*x) [outputIndex] = (*x) [inputIndex];
    (*x0)[outputIndex] = (*x0)[inputIndex];
    (*v) [outputIndex] = (*v) [inputIndex];
    (*v0)[outputIndex] = (*v0)[inputIndex];
    (*f) [outputIndex] = (*f) [inputIndex];
    (*dx)[outputIndex] = (*dx)[inputIndex];

    // temporary state vectors
    unsigned int i;
    for (i=0; i<vectorsCoord.size(); i++)
    {
        VecCoord& vector = *vectorsCoord[i];
        vector[outputIndex]=vector[inputIndex];
    }
    for ( i=0; i<vectorsDeriv.size(); i++)
    {
        VecDeriv& vector = *vectorsDeriv[i];
        vector[outputIndex]=vector[inputIndex];
    }

    // forces
    (*internalForces)[outputIndex] = (*internalForces)[inputIndex];
    (*externalForces)[outputIndex] = (*externalForces)[inputIndex];

}

template <class DataTypes>
void MechanicalObject<DataTypes>::swapValues (const int idx1, const int idx2)
{

    // standard state vectors
    Coord tmp = (*x)[idx1];
    (*x) [idx1] = (*x) [idx2];
    (*x) [idx2] = tmp;

    tmp = (*x0)[idx1];
    (*x0)[idx1] = (*x0)[idx2];
    (*x0)[idx2] = tmp;

    Deriv tmp2 = (*v)[idx1];
    (*v) [idx1] = (*v) [idx2];
    (*v) [idx2] = tmp2;

    tmp2 = (*v0) [idx1];
    (*v0)[idx1] = (*v0)[idx2];
    (*v0)[idx2] = tmp2;

    tmp2 = (*f) [idx1];
    (*f) [idx1] = (*f)[idx2];
    (*f) [idx2] = tmp2;

    tmp2 = (*dx) [idx1];
    (*dx)[idx1] = (*dx)[idx2];
    (*dx)[idx2] = tmp2;

    // temporary state vectors
    unsigned int i;
    for (i=0; i<vectorsCoord.size(); i++)
    {
        VecCoord& vector = *vectorsCoord[i];
        tmp = vector[idx1];
        vector[idx1] = vector[idx2];
        vector[idx2] = tmp;
    }
    for ( i=0; i<vectorsDeriv.size(); i++)
    {
        VecDeriv& vector = *vectorsDeriv[i];
        tmp2 = vector[idx1];
        vector[idx1] = vector[idx2];
        vector[idx2] = tmp2;
    }

    // forces
    tmp2 = (*internalForces)[idx1];
    (*internalForces)[idx1] = (*internalForces)[idx2];
    (*internalForces)[idx2] = tmp2;

    tmp2 = (*externalForces)[idx1];
    (*externalForces)[idx1] = (*externalForces)[idx2];
    (*externalForces)[idx2] = tmp2;

}



template <class DataTypes>
void MechanicalObject<DataTypes>::renumberValues( const std::vector< unsigned int > &index )
{

    // standard state vectors
    VecCoord x_cp  = (*x);
    VecCoord x0_cp = (*x0);

    VecDeriv v_cp  = (*v);
    VecDeriv v0_cp = (*v0);
    VecDeriv f_cp  = (*f);
    VecDeriv dx_cp = (*dx);

    // temporary state vectors
    std::vector< VecCoord > vecCoord_cp;
    vecCoord_cp.resize( vectorsCoord.size() );
    for (unsigned int i = 0; i < vectorsCoord.size(); ++i)
    {
        vecCoord_cp[i] = ( *(vectorsCoord[i]) );
    }
    std::vector< VecDeriv > vecDeriv_cp;
    vecDeriv_cp.resize( vectorsDeriv.size() );
    for (unsigned int i = 0; i < vectorsDeriv.size(); ++i)
    {
        vecDeriv_cp[i] = ( *(vectorsDeriv[i]) );
    }

    // forces
    VecDeriv intern_cp = (*internalForces);
    VecDeriv extern_cp = (*externalForces);

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        (*x )[i] = x_cp [ index[i] ];
        (*x0)[i] = x0_cp[ index[i] ];
        (*v )[i] = v_cp [ index[i] ];
        (*v0)[i] = v0_cp[ index[i] ];
        (*f )[i] = f_cp [ index[i] ];
        (*dx)[i] = dx_cp[ index[i] ];

        for (unsigned j = 0; j < vectorsCoord.size(); ++j)
            (*vectorsCoord[j])[i] = vecCoord_cp[j][ index[i] ];

        for (unsigned j = 0; j < vectorsDeriv.size(); ++j)
            (*vectorsDeriv[j])[i] = vecDeriv_cp[j][ index[i] ];

        (*internalForces)[i] = intern_cp[ index[i] ];
        (*externalForces)[i] = extern_cp[ index[i] ];


    }
}



template <class DataTypes>
void MechanicalObject<DataTypes>::resize(const int size)
{
    (*x).resize(size);
    // Note (Jeremie A.): should we really update initial position vector size ???
    if (x0!=NULL)
        (*x0).resize(size);
    (*v).resize(size);
    if (v0!=NULL)
        (*v0).resize(size);
    (*f).resize(size);
    (*dx).resize(size);
    if (size!=vsize)
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
void MechanicalObject<DataTypes>::applyTranslation (double dx, double dy, double dz)
{
    this->translation[0]+=dx;
    this->translation[1]+=dy;
    this->translation[2]+=dz;
    VecCoord& x = *this->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        DataTypes::add
        (x[i],dx,dy,dz);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::applyScale(double s)
{
    this->scale*=s;
    VecCoord& x = *this->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        x[i] *= s;
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::getIndicesInSpace(std::vector<unsigned>& /*indices*/, Real /*xmin*/, Real /*xmax*/, Real /*ymin*/, Real /*ymax*/, Real /*zmin*/, Real /*zmax*/) const
{
    std::cerr<<"ERROR: UNSUPPORTED MechanicalObject<DataTypes>::getIndicesInSpace()"<<std::endl;
    //const VecCoord& x = *getX();
    //for( unsigned i=0; i<x.size(); ++i ) {
    //	if( x[i][0] >= xmin && x[i][0] <= xmax && x[i][1] >= ymin && x[i][1] <= ymax && x[i][2] >= zmin && x[i][2] <= zmax ) {
    //		indices.push_back(i);
    //	}
    //}
}
template <class DataTypes>
void MechanicalObject<DataTypes>::computeWeightedValue( const unsigned int i, const std::vector< unsigned int >& ancestors, const std::vector< double >& coefs)
{
    /// HD interpolate position, speed,force,...
    /// assume all coef sum to 1.0
    (*x)[i]=Coord();
    (*x0)[i]=Coord();
    (*v)[i]=Deriv();
    (*f)[i]=Deriv();
    (*dx)[i]=Deriv();
    unsigned int j;
    for (j=0; j<ancestors.size(); ++j)
    {
        (*x)[i]+=(*x)[ancestors[j]]*coefs[j];
        (*x0)[i]+=(*x0)[ancestors[j]]*coefs[j];
        (*v)[i]+=(*v)[ancestors[j]]*coefs[j];
        (*f)[i]+=(*f)[ancestors[j]]*coefs[j];
        (*dx)[i]+=(*dx)[ancestors[j]]*coefs[j];

    }
    for (unsigned int k=0; k<vectorsCoord.size(); k++)
    {
        if (vectorsCoord[k]!=NULL && vectorsCoord[k]->size()!=0)
        {
            (*vectorsCoord[k])[i]=Coord();
            for (j=0; j<ancestors.size(); ++j)
            {
                (*vectorsCoord[k])[i]+= (*vectorsCoord[k])[ancestors[j]]*coefs[j];
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
                (*vectorsDeriv[k])[i]+= (*vectorsDeriv[k])[ancestors[j]]*coefs[j];
            }
        }
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::init()
{
    Topology* topo = dynamic_cast<Topology*>(this->getContext()->getTopology());
    if (topo!=NULL && topo->hasPos())
    {
        int nbp = topo->getNbPoints();
        std::cout<<"Setting "<<nbp<<" points from topology."<<std::endl;
        this->resize(nbp);
        for (int i=0; i<nbp; i++)
        {
            //DataTypes::set((*getX())[i], topo->getPX(i), topo->getPY(i), topo->getPZ(i));
            DataTypes::set
            ((*getX())[i], topo->getPX(i)*scale+translation[0], topo->getPY(i)*scale+translation[1], topo->getPZ(i)*scale+translation[2]);
        }
    }

    // Save initial state
    this->x0 = new VecCoord;
    *this->x0 = *x;
    this->v0 = new VecDeriv;
    *this->v0 = *v;

}

//
// Integration related methods
//

template <class DataTypes>
void MechanicalObject<DataTypes>::reset()
{
    if (x0 == NULL)
        return;
    // Back to initial state
    this->resize(this->x0->size());
    *this->x = *x0;
    *this->v = *v0;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::writeState( std::ostream& out )
{
    out<<*getX()<<" "<<*getV()<<" ";
}

template <class DataTypes>
void MechanicalObject<DataTypes>::beginIntegration(double /*dt*/)
{
    this->f = this->internalForces;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::endIntegration(double /*dt*/)
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


template<class DataTypes>
typename DataTypes::VecCoord* MechanicalObject<DataTypes>::getVecCoord(unsigned int index)
{
    if (index>=vectorsCoord.size())
        vectorsCoord.resize(index+1);
    if (vectorsCoord[index]==NULL)
        vectorsCoord[index] = new VecCoord;
    return vectorsCoord[index];
}

template<class DataTypes>
typename DataTypes::VecDeriv* MechanicalObject<DataTypes>::getVecDeriv(unsigned int index)
{
    if (index>=vectorsDeriv.size())
        vectorsDeriv.resize(index+1);
    if (vectorsDeriv[index]==NULL)
        vectorsDeriv[index] = new VecDeriv;

    return vectorsDeriv[index];
}

template <class DataTypes>
void MechanicalObject<DataTypes>::vAlloc(VecId v)
{
    if (v.type == V_COORD && v.index >= V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = getVecCoord(v.index);
        vec->resize(vsize);
    }
    else if (v.type == V_DERIV && v.index >= V_FIRST_DYNAMIC_INDEX)
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
    if (v.type == V_COORD && v.index >= V_FIRST_DYNAMIC_INDEX)
    {
        VecCoord* vec = getVecCoord(v.index);
        vec->resize(0);
    }
    else if (v.type == V_DERIV && v.index >= V_FIRST_DYNAMIC_INDEX)
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
            if (v.type == V_COORD)
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
                if (v.type == V_COORD)
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
                if (v.type == V_COORD)
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
            if (v.type == V_COORD)
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
                    if (v.type == V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        if (b.type == V_COORD)
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
                    else if (b.type == V_DERIV)
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
                    if (v.type == V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        if (b.type == V_COORD)
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
                    else if (b.type == V_DERIV)
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
            else
            {
                if (f==1.0)
                {
                    // v = a+b
                    if (v.type == V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        VecCoord* va = getVecCoord(a.index);
                        vv->resize(va->size());
                        if (b.type == V_COORD)
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
                    else if (b.type == V_DERIV)
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
                    if (v.type == V_COORD)
                    {
                        VecCoord* vv = getVecCoord(v.index);
                        VecCoord* va = getVecCoord(a.index);
                        vv->resize(va->size());
                        if (b.type == V_COORD)
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
                    else if (b.type == V_DERIV)
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
double MechanicalObject<DataTypes>::vDot(VecId a, VecId b)
{
    double r = 0.0;
    if (a.type == V_COORD && b.type == V_COORD)
    {
        VecCoord* va = getVecCoord(a.index);
        VecCoord* vb = getVecCoord(b.index);
        for (unsigned int i=0; i<va->size(); i++)
            r += (*va)[i] * (*vb)[i];
    }
    else if (a.type == V_DERIV && b.type == V_DERIV)
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
    if (v.type == V_COORD)
    {
        this->x = getVecCoord(v.index);
    }
    else
    {
        std::cerr << "Invalid setX operation ("<<v<<")\n";
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setV(VecId v)
{
    if (v.type == V_DERIV)
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
    if (v.type == V_DERIV)
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
    if (v.type == V_DERIV)
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
    /*
        if (v.type == V_DERIV)
        {
            this->dx = getVecDeriv(v.index);
        }
        else
        {
            std::cerr << "Invalid setC operation ("<<v<<")\n";
        }
    */
}


template <class DataTypes>
void MechanicalObject<DataTypes>::printDOF( VecId v, std::ostream& out)
{
    if( v.type==V_COORD )
    {
        VecCoord& x= *getVecCoord(v.index);
        for( unsigned i=0; i<x.size(); ++i )
            out<<x[i]<<" ";
    }
    else if( v.type==V_DERIV )
    {
        VecDeriv& x= *getVecDeriv(v.index);
        for( unsigned i=0; i<x.size(); ++i )
            out<<x[i]<<" ";
    }
    else
        out<<"MechanicalObject<DataTypes>::printDOF, unknown v.type = "<<v.type<<std::endl;
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
    VecConst& c= *getC();
    c.clear();
}

/*
template <class DataTypes>
bool MechanicalObject<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
  const VecCoord& x = *getX();
  if (x.size() <= 0) return false;
  double p[3] = {0,0,0};
  for (unsigned int i=0; i<x.size(); i++)
  {
    DataTypes::get(p[0], p[1], p[2], x[i]);
    for (int c=0;c<3;c++)
    {
      if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
      if (p[c] < minBBox[c]) minBBox[c] = p[c];
    }
  }
  return true;
}
*/

} // namespace Core

} // namespace Sofa

#endif

