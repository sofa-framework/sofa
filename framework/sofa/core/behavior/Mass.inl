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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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
    void	operator()( const MechanicalParams* mparams /* PARAMS FIRST */, Mass< DataTypes >*m,Shared_rw< objectmodel::Data< typename  DataTypes::VecDeriv> > _a,Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _f)
    {
        m->accFromF(mparams /* PARAMS FIRST */, _a.access(),_f.read());
    }
};

template<class DataTypes>
struct ParallelMassAddMDx
{
public:
    void	operator()(const MechanicalParams* mparams /* PARAMS FIRST */, Mass< DataTypes >*m,Shared_rw< objectmodel::Data< typename DataTypes::VecDeriv> > _res,Shared_r< objectmodel::Data< typename DataTypes::VecDeriv> > _dx,double factor)
    {
        m->addMDx(mparams /* PARAMS FIRST */, _res.access(),_dx.read(),factor);
    }
};

// template<class DataTypes>
// void Mass<DataTypes>::addMBKv(double mFactor, double bFactor, double kFactor)
// {
//     this->ForceField<DataTypes>::addMBKv(mFactor, bFactor, kFactor);
//     if (mFactor != 0.0)
//     {
//         if (this->mstate)
//               Task<ParallelMassAddMDx < DataTypes > >(this,**this->mstate->getF(), **this->mstate->getV(),mFactor);
//     }
// }
#endif /* SOFA_SMP */


template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId fid, double factor)
{
    if (mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<ParallelMassAddMDx< DataTypes > >(mparams /* PARAMS FIRST */, this, **defaulttype::getShared(*fid[this->mstate].write()), **defaulttype::getShared(*mparams->readDx(this->mstate)), factor);
        else
#endif /* SOFA_SMP */
            addMDx(mparams /* PARAMS FIRST */, *fid[this->mstate].write(), *mparams->readDx(this->mstate), factor);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx , double factor )
{
    if (this->mstate)
    {
        this->mstate->forceMask.setInUse(this->useMask());
        addMDx( *f.beginEdit() , dx.getValue(), factor);
        f.endEdit();
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(VecDeriv& /*f*/, const VecDeriv& /*dx*/, double /*factor*/)
{
    serr << "ERROR("<<getClassName()<<"): addMDx(VecDeriv& , const VecDeriv& , double ) not implemented." << sendl;
}

template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId aid)
{
    if(mparams)
    {
#ifdef SOFA_SMP
        if (mparams->execMode() == ExecParams::EXEC_KAAPI)
            Task<ParallelMassAccFromF< DataTypes > >(mparams /* PARAMS FIRST */, this, **defaulttype::getShared(*aid[this->mstate].write()), **defaulttype::getShared(*mparams->readF(this->mstate)));
        else
#endif /* SOFA_SMP */
            accFromF(mparams /* PARAMS FIRST */, *aid[this->mstate].write(), *mparams->readF(this->mstate));
    }
}

template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f)
{
    if (this->mstate)
    {
        this->mstate->forceMask.setInUse(this->useMask());
        accFromF( *a.beginEdit() , f.getValue());
        a.endEdit();
    }
}

template<class DataTypes>
void Mass<DataTypes>::accFromF(VecDeriv& /*a*/, const VecDeriv& /*f*/)
{
    serr << "ERROR("<<getClassName()<<"): accFromF(VecDeriv& /*a*/, const VecDeriv& /*f*/) not implemented." << sendl;
}

template<class DataTypes>
void Mass<DataTypes>::addDForce(const MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv & /*df*/, const DataVecDeriv & /*dx*/)
{
    // @TODO Remove
    // Hack to disable warning message
    mparams->kFactor();
}

template<class DataTypes>
void Mass<DataTypes>::addMBKdx(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId dfId)
{
    this->ForceField<DataTypes>::addMBKdx(mparams /* PARAMS FIRST */, dfId);
    if (mparams->mFactor() != 0.0)
    {
        addMDx(mparams /* PARAMS FIRST */, *dfId[this->mstate].write(), *mparams->readDx(this->mstate), mparams->mFactor());
    }
}

template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getKineticEnergy(mparams /* PARAMS FIRST */, *mparams->readV(this->mstate));
    return 0;
}

template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecDeriv& v) const
{
    return getKineticEnergy(v.getValue());
}

template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy(const VecDeriv& /*v*/ ) const
{
    serr << "ERROR("<<getClassName()<<"): getKineticEnergy( const VecDeriv& ) not implemented." << sendl;
    return 0.0;
}

template<class DataTypes>
double Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getPotentialEnergy(mparams /* PARAMS FIRST */, *mparams->readX(this->mstate));
    return 0;
}

template<class DataTypes>
double Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord& x) const
{
    return getPotentialEnergy(x.getValue());
}

template<class DataTypes>
double Mass<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/ ) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy( const VecCoord& ) not implemented." << sendl;
    return 0.0;
}

template<class DataTypes>
void Mass<DataTypes>::addKToMatrix(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/)
{
//    serr << "ERROR("<<getClassName()<<"): addKToMatrix not implemented." << sendl;
}

template<class DataTypes>
void Mass<DataTypes>::addBToMatrix(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/)
{
//	serr << "ERROR("<<getClassName()<<"): addBToMatrix not implemented." << sendl;
}

template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addMToMatrix(r.matrix, mparams->mFactor(), r.offset);
}

template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*mFact*/, unsigned int &/*offset*/)
{
    serr << "ERROR("<<getClassName()<<"): addMToMatrix not implemented." << sendl;
}

template<class DataTypes>
void Mass<DataTypes>::addMBKToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    this->ForceField<DataTypes>::addMBKToMatrix(mparams /* PARAMS FIRST */, matrix);
    if (mparams->mFactor() != 0.0)
        addMToMatrix(mparams /* PARAMS FIRST */, matrix);
}

template<class DataTypes>
void Mass<DataTypes>::addGravityToV(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId vid)
{
    if(this->mstate)
    {
        DataVecDeriv& v = *vid[this->mstate].write();
        addGravityToV(mparams /* PARAMS FIRST */, v);
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
void Mass<DataTypes>::exportGnuplot(const MechanicalParams* mparams /* PARAMS FIRST */, double time)
{
    if (m_gnuplotFileEnergy!=NULL)
    {
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->getKineticEnergy(mparams)
                <<"\t"<< this->getPotentialEnergy(mparams)
                <<"\t"<< this->getPotentialEnergy(mparams)
                +this->getKineticEnergy(mparams)<< sendl;
    }
}

/// return the mass relative to the DOF #index
template <class DataTypes>
double Mass<DataTypes>::getElementMass(unsigned int ) const
{
    serr << "ERROR("<<getClassName()<<"): getElementMass with Scalar not implemented" << sendl;
    return 0.0;
}

template <class DataTypes>
void Mass<DataTypes>::getElementMass(unsigned int , defaulttype::BaseMatrix *m) const
{
    static unsigned int dimension = defaulttype::DataTypeInfo<Coord>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    serr << "ERROR("<<getClassName()<<"): getElementMass with Matrix not implemented" << sendl;
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
