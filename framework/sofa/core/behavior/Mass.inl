/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_MASS_INL
#define SOFA_CORE_BEHAVIOR_MASS_INL

#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/defaulttype/DataTypeInfo.h>


namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm)
    , m_gnuplotFileEnergy(NULL)
{
}

template<class DataTypes>
Mass<DataTypes>::~Mass()
{
}

template<class DataTypes>
void Mass<DataTypes>::init()
{
    ForceField<DataTypes>::init();
}

#ifdef SOFA_SMP
template<class DataTypes>
struct ParallelMassAccFromF
{
    void	operator()( const MechanicalParams* mparams, Mass< DataTypes >*m,Shared_rw< objectmodel::Data< typename  DataTypes::VecDeriv> > _a,Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _f)
    {
        m->accFromF(mparams, _a.access(),_f.read());
    }
};

template<class DataTypes>
struct ParallelMassAddMDx
{
public:
    void	operator()(const MechanicalParams* mparams, Mass< DataTypes >*m,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > _res,Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _dx,double factor)
    {
        m->addMDx(mparams, _res.access(),_dx.read(),factor);
    }
};

// template<class DataTypes>
// void Mass<DataTypes>::addMBKv(double mFactor, double bFactor, double kFactor)
// {
//     this->ForceField<DataTypes>::addMBKv(mFactor, bFactor, kFactor);
//     if (mFactor != 0.0)
//     {
//         if (this->mstate)
//               Task<ParallelMassAddMDx < DataTypes > >(this,**this->mstate->getF(), *this->mstate->read(core::ConstVecCoordId::velocity())->getValue(),mFactor);
//     }
// }
#endif /* SOFA_SMP */


template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* mparams, MultiVecDerivId fid, double factor)
{
    if (mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<ParallelMassAddMDx< DataTypes > >(mparams, this, **defaulttype::getShared(*fid[this->mstate.get(mparams)].write()), **defaulttype::getShared(*mparams->readDx(this->mstate)), factor);
        else
#endif /* SOFA_SMP */
            addMDx(mparams, *fid[this->mstate.get(mparams)].write(), *mparams->readDx(this->mstate), factor);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* /*mparams*/, DataVecDeriv& /*f*/, const DataVecDeriv& /*dx*/ , double /*factor*/ )
{
    serr << "ERROR("<<getClassName()<< "): addMDx(const MechanicalParams* , DataVecDeriv& , const DataVecDeriv&  , double  ) not implemented." << sendl;
}


template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid)
{
    if(mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<ParallelMassAccFromF< DataTypes > >(mparams, this, **defaulttype::getShared(*aid[this->mstate.get(mparams)].write()), **defaulttype::getShared(*mparams->readF(this->mstate)));
        else
#endif /* SOFA_SMP */
            accFromF(mparams, *aid[this->mstate.get(mparams)].write(), *mparams->readF(this->mstate));
    }
    else serr <<"Mass<DataTypes>::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid) receives no mparam" << sendl;
}

template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* /*mparams*/, DataVecDeriv& /*a*/, const DataVecDeriv& /*f*/)
{
    serr << "ERROR("<<getClassName()<<"): accFromF(const MechanicalParams* , DataVecDeriv& , const DataVecDeriv& ) not implemented." << sendl;
}


template<class DataTypes>
void Mass<DataTypes>::addDForce(const MechanicalParams*
                                #ifndef NDEBUG
                                mparams
                                #endif
                                ,
                                DataVecDeriv & /*df*/, const DataVecDeriv & /*dx*/)
{
#ifndef NDEBUG
    // @TODO Remove
    // Hack to disable warning message
    mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
#endif
}

template<class DataTypes>
void Mass<DataTypes>::addMBKdx(const MechanicalParams* mparams, MultiVecDerivId dfId)
{
    this->ForceField<DataTypes>::addMBKdx(mparams, dfId);
    if (mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()) != 0.0)
    {
        addMDx(mparams, *dfId[this->mstate.get(mparams)].write(), *mparams->readDx(this->mstate), mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()));
    }
}

template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy(const MechanicalParams* /*mparams*/) const
{
    serr << "ERROR("<<getClassName()<<"): getKineticEnergy(const MechanicalParams*) not implemented." << sendl;
    return 0.0;
}

template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy(const MechanicalParams* /*mparams*/, const DataVecDeriv& /*v*/) const
{
    serr << "ERROR("<<getClassName()<<"): getKineticEnergy(const MechanicalParams*, const DataVecDeriv& ) not implemented." << sendl;
    return 0.0;
}


template<class DataTypes>
double Mass<DataTypes>::getPotentialEnergy(const MechanicalParams*) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy( const MechanicalParams* ) not implemented." << sendl;
    return 0.0;
}

template<class DataTypes>
double Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/) const
{   
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy( const MechanicalParams*, const DataVecCoord& ) not implemented." << sendl;
    return 0.0;
}


template<class DataTypes>
defaulttype::Vec6d Mass<DataTypes>::getMomentum( const MechanicalParams* mparams ) const
{
    if (this->mstate)
        return getMomentum(mparams, *mparams->readX(this->mstate), *mparams->readV(this->mstate));
    return defaulttype::Vec6d();
}

template<class DataTypes>
defaulttype::Vec6d Mass<DataTypes>::getMomentum( const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ ) const
{
    serr << "ERROR("<<getClassName()<<"): getMomentum( const MechanicalParams*, const DataVecCoord&, const DataVecDeriv& ) not implemented." << sendl;
    return defaulttype::Vec6d();
}



template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addMToMatrix(r.matrix, mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()), r.offset);
}

template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*mFact*/, unsigned int &/*offset*/)
{
    static int i=0;
    if (i < 10) {
        serr << "ERROR("<<getClassName()<<"): addMToMatrix not implemented." << sendl;
        i++;
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    this->ForceField<DataTypes>::addMBKToMatrix(mparams, matrix);
    if (mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()) != 0.0)
        addMToMatrix(mparams, matrix);
}

template<class DataTypes>
void Mass<DataTypes>::addSubMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> /*subMatrixIndex*/) {
    addMBKToMatrix(mparams,matrix); // default implementation use full addMFunction
}

template<class DataTypes>
void Mass<DataTypes>::addGravityToV(const MechanicalParams* mparams, MultiVecDerivId vid)
{
    if(this->mstate)
    {
        DataVecDeriv& v = *vid[this->mstate.get(mparams)].write();
        addGravityToV(mparams, v);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addGravityToV(const MechanicalParams* /* mparams */, DataVecDeriv& /* d_v */)
{
    static int i=0;
    if (i < 10) {
        serr << "ERROR("<<getClassName()<<"): addGravityToV not implemented." << sendl;
        i++;
    }
}


template<class DataTypes>
void Mass<DataTypes>::initGnuplot(const std::string path)
{
    if (!this->getName().empty())
    {
        if (m_gnuplotFileEnergy != NULL)
            delete m_gnuplotFileEnergy;

        m_gnuplotFileEnergy = new std::ofstream( (path+this->getName()+"_Energy.txt").c_str() );
    }
}

template<class DataTypes>
void Mass<DataTypes>::exportGnuplot(const MechanicalParams* mparams, double time)
{
    if (m_gnuplotFileEnergy!=NULL)
    {
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->getKineticEnergy(mparams)
                               <<"\t"<< this->getPotentialEnergy(mparams)
                              <<"\t"<< this->getPotentialEnergy(mparams)
                                +this->getKineticEnergy(mparams)<< std::endl;
    }
}

template <class DataTypes>
double Mass<DataTypes>::getElementMass(unsigned int ) const
{
    serr << "ERROR("<<getClassName()<<"): getElementMass with Scalar not implemented" << sendl;
    return 0.0;
}

template <class DataTypes>
void Mass<DataTypes>::getElementMass(unsigned int , defaulttype::BaseMatrix *m) const
{
    static const defaulttype::BaseMatrix::Index dimension = (defaulttype::BaseMatrix::Index) defaulttype::DataTypeInfo<Coord>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    serr << "ERROR("<<getClassName()<<"): getElementMass with Matrix not implemented" << sendl;
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
