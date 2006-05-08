#ifndef SOFA_CORE_MECHANICALOBJECT_INL
#define SOFA_CORE_MECHANICALOBJECT_INL

#include "MechanicalObject.h"
#include "Encoding.inl"
#include <assert.h>
#include <iostream>

namespace Sofa
{

namespace Core
{

template <class DataTypes>
MechanicalObject<DataTypes>::MechanicalObject()
    : vsize(0), mapping(NULL), topology(NULL), mass(NULL)
{
    x = new VecCoord;
    v = new VecDeriv;
    f = new VecDeriv;
    dx = new VecDeriv;
    // default size is 1
    resize(1);
    translation[0]=0.0;
    translation[1]=0.0;
    translation[2]=0.0;
    setVecCoord(VecId::position().index, this->x);
    setVecDeriv(VecId::velocity().index, this->v);
    setVecDeriv(VecId::force().index, this->f);
    setVecDeriv(VecId::dx().index, this->dx);
}

template <class DataTypes>
MechanicalObject<DataTypes>::~MechanicalObject()
{
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setMapping(Core::BasicMechanicalMapping *map)
{
    this->mapping = map;
}

/*
template <class DataTypes>
void MechanicalObject<DataTypes>::addMapping(Core::BasicMapping *mMap)
{
	assert(mMap);
	mappings.push_back(mMap);
}
*/

template <class DataTypes>
void MechanicalObject<DataTypes>::addForceField(Core::ForceField *mFField)
{
    assert(mFField);
    forcefields.push_back(mFField);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::removeForceField(Core::ForceField* mFField)
{
    ForceFieldIt it = std::find(forcefields.begin(), forcefields.end(), mFField);
    if (it!=forcefields.end())
    {
        forcefields.erase(it);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addConstraint(Core::Constraint *mConstraint)
{
    assert(mConstraint);
    constraints.push_back(mConstraint);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::removeConstraint(Core::Constraint* mConstraint)
{
    ConstraintIt it = std::find(constraints.begin(), constraints.end(), mConstraint);
    if (it!=constraints.end())
    {
        constraints.erase(it);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addMechanicalModel(BasicMechanicalObject *mmodel)
{
    assert(mmodel);
    mmodels.push_back(mmodel);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::removeMechanicalModel(BasicMechanicalObject *mmodel)
{
    MModelIt it = std::find(mmodels.begin(), mmodels.end(), mmodel);
    if (it!=mmodels.end())
    {
        mmodels.erase(it);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::resize(int size)
{
    getX()->resize(size);
    getV()->resize(size);
    getF()->resize(size);
    getDx()->resize(size);
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
void MechanicalObject<DataTypes>::init()
{
    Topology* topo = this->getTopology();
    if (topo!=NULL && topo->hasPos())
    {
        int nbp = topo->getNbPoints();
        std::cout<<"Setting "<<nbp<<" points from topology with translation "<<translation[0]<<" "<<translation[1]<<" "<<translation[2]<<std::endl;
        this->resize(nbp);
        for (int i=0; i<nbp; i++)
        {
            DataTypes::set((*getX())[i], topo->getPX(i)+translation[0], topo->getPY(i)+translation[1], topo->getPZ(i)+translation[2]);
        }
    }

    if (mapping!=NULL) mapping->init();

    if (mass!=NULL) mass->init();

    //this->propagateX();
    //this->propagateV();
    {
        ForceFieldIt it = forcefields.begin();
        ForceFieldIt itEnd = forcefields.end();
        for (; it != itEnd; it++)
            (*it)->init();
    }
    {
        ConstraintIt it = constraints.begin();
        ConstraintIt itEnd = constraints.end();
        for (; it != itEnd; it++)
            (*it)->init();
    }
    /*
    {
    	MappingIt it = mappings.begin();
    	MappingIt itEnd = mappings.end();
    	for (; it != itEnd; it++)
    		(*it)->init();
    }
    */
    {
        MModelIt it = mmodels.begin();
        MModelIt itEnd = mmodels.end();
        for (; it != itEnd; it++)
            (*it)->init();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::beginIteration(double dt)
{
    MModelIt it = mmodels.begin();
    MModelIt itEnd = mmodels.end();
    for (; it != itEnd; it++)
    {
        (*it)->beginIteration(dt);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::endIteration(double dt)
{
    MModelIt it = mmodels.begin();
    MModelIt itEnd = mmodels.end();
    for (; it != itEnd; it++)
    {
        (*it)->endIteration(dt);
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::propagateX()
{
    if (mapping != NULL)
        mapping->propagateX();
    MModelIt it = mmodels.begin();
    MModelIt itEnd = mmodels.end();
    for (; it != itEnd; it++)
    {
        (*it)->propagateX();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::propagateV()
{
    if (mapping != NULL)
        mapping->propagateV();
    MModelIt it = mmodels.begin();
    MModelIt itEnd = mmodels.end();
    for (; it != itEnd; it++)
    {
        (*it)->propagateV();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::propagateDx()
{
    if (mapping != NULL)
        mapping->propagateDx();
    MModelIt it = mmodels.begin();
    MModelIt itEnd = mmodels.end();
    for (; it != itEnd; it++)
    {
        (*it)->propagateDx();
    }
}

template <class DataTypes>
void MechanicalObject<DataTypes>::resetForce()
{
    f->resize(x->size());
    for (unsigned int i=0; i<f->size(); i++)
        (*f)[i] = typename VecDeriv::value_type();
    MModelIt it = mmodels.begin();
    MModelIt itEnd = mmodels.end();
    for (; it != itEnd; it++)
        (*it)->resetForce();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accumulateForce()
{
    if (mass != NULL)
        mass->computeForce();

    {
        MModelIt it = mmodels.begin();
        MModelIt itEnd = mmodels.end();
        for (; it != itEnd; it++)
            (*it)->accumulateForce();
    }
    {
        ForceFieldIt it = forcefields.begin();
        ForceFieldIt itEnd = forcefields.end();
        for (; it != itEnd; it++)
            (*it)->addForce();
    }
    if (mapping != NULL)
        mapping->accumulateForce();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accumulateDf()
{
    if (mass != NULL)
        mass->computeDf();

    {
        MModelIt it = mmodels.begin();
        MModelIt itEnd = mmodels.end();
        for (; it != itEnd; it++)
            (*it)->accumulateDf();
    }
    {
        ForceFieldIt it = forcefields.begin();
        ForceFieldIt itEnd = forcefields.end();
        for (; it != itEnd; it++)
            (*it)->addDForce();
    }
    if (mapping != NULL)
        mapping->accumulateDf();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::applyConstraints()
{
    ConstraintIt it = constraints.begin();
    ConstraintIt itEnd = constraints.end();
    for (; it != itEnd; it++)
        (*it)->applyConstraint();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::addMDx()
{
    if (mass != NULL)
        mass->addMDx();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::accFromF()
{
    if (mass != NULL)
        mass->accFromF();
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setObject(Abstract::BehaviorModel* obj)
{
    MModelIt it = mmodels.begin();
    MModelIt itEnd = mmodels.end();
    for (; it != itEnd; it++)
        (*it)->setObject(obj);
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setTopology(Topology* topo)
{
    this->topology = topo;
}

template <class DataTypes>
Topology* MechanicalObject<DataTypes>::getTopology()
{
    return this->topology;
}

template <class DataTypes>
void MechanicalObject<DataTypes>::setMass(Mass* m)
{
    this->mass = m;
}

template <class DataTypes>
Mass* MechanicalObject<DataTypes>::getMass()
{
    return this->mass;
}

//
// Integration related methods
//

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
    vOp(v); // clear vector
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
                        (*vv)[i] *= f;
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    for (unsigned int i=0; i<vv->size(); i++)
                        (*vv)[i] *= f;
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
                        (*vv)[i] = (*vb)[i] * f;
                }
                else
                {
                    VecDeriv* vv = getVecDeriv(v.index);
                    VecDeriv* vb = getVecDeriv(b.index);
                    vv->resize(vb->size());
                    for (unsigned int i=0; i<vv->size(); i++)
                        (*vv)[i] = (*vb)[i] * f;
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
                                (*vv)[i] += (*vb)[i]*f;
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            vv->resize(vb->size());
                            for (unsigned int i=0; i<vv->size(); i++)
                                (*vv)[i] += (*vb)[i]*f;
                        }
                    }
                    else if (b.type == V_DERIV)
                    {
                        VecDeriv* vv = getVecDeriv(v.index);
                        VecDeriv* vb = getVecDeriv(b.index);
                        vv->resize(vb->size());
                        for (unsigned int i=0; i<vv->size(); i++)
                            (*vv)[i] += (*vb)[i]*f;
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
                                (*vv)[i] += (*vb)[i]*f;
                            }
                        }
                        else
                        {
                            VecDeriv* vb = getVecDeriv(b.index);
                            for (unsigned int i=0; i<vv->size(); i++)
                            {
                                (*vv)[i] = (*va)[i];
                                (*vv)[i] += (*vb)[i]*f;
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
                            (*vv)[i] += (*vb)[i]*f;
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

} // namespace Core

} // namespace Sofa

#endif
