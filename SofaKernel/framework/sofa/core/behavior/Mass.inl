/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_MASS_INL
#define SOFA_CORE_BEHAVIOR_MASS_INL

#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <fstream>


namespace sofa
{

namespace core
{

namespace behavior
{


template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm)
    , d_vertexMass( initData(&d_vertexMass, "vertexMass", "Specify a vector giving the mass of each vertex. \n"
                                                                  "If unspecified or wrongly set, another mass information is used.") )
    , d_massDensity( initData(&d_massDensity, "massDensity", "Specify real and strictly positive value(s) for the mass density. \n"
                                                                       "If unspecified or wrongly set, the totalMass information is used.") )
    , d_totalMass( initData(&d_totalMass, (Real) 1.0, "totalMass","Specify the total mass resulting from all particles. \n"
                                                                  "If unspecified or wrongly set, the default value is used: totalMass = 1.0") )
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

template<class DataTypes>
bool Mass<DataTypes>::checkTotalMass()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(d_totalMass.getValue() <= 0.0)
    {
        msg_warning(this) << "totalMass data can not have a negative value.\n"
                          << "Switching back to default values: totalMass = 1.0\n"
                          << "To remove this warning, you need to set a strictly positive value to the totalMass data";
        d_totalMass.setValue(1.0) ;
        return false;
    }
    else
    {
        return true;
    }
}

template<class DataTypes>
void  Mass<DataTypes>::getVertexMass(sofa::helper::vector< Real >& vertexMass)
{
    vertexMass = d_vertexMass.getValue();
}

template<class DataTypes>
void  Mass<DataTypes>::getMassDensity(sofa::helper::vector< Real >& massDensity)
{
    massDensity = d_massDensity.getValue();
}

template<class DataTypes>
void Mass<DataTypes>::getTotalMass(Real& totalMass)
{
    totalMass = d_totalMass.getValue();
}

template<class DataTypes>
void Mass<DataTypes>::setTotalMass(Real totalMass)
{
    Real currentTotalMass = d_totalMass.getValue();
    d_totalMass.setValue(totalMass);
    if(!checkTotalMass())
    {
        msg_warning() << "Given value to setTotalMass() is not a strictly positive value\n"
                      << "Previous value is used: totalMass = " << currentTotalMass;
        d_totalMass.setValue(currentTotalMass);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* mparams, MultiVecDerivId fid, SReal factor)
{
    if (mparams)
    {
            addMDx(mparams, *fid[this->mstate.get(mparams)].write(), *mparams->readDx(this->mstate), factor);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* /*mparams*/, DataVecDeriv& /*f*/, const DataVecDeriv& /*dx*/ , SReal /*factor*/ )
{
    serr << "ERROR("<<getClassName()<< "): addMDx(const MechanicalParams* , DataVecDeriv& , const DataVecDeriv&  , SReal  ) not implemented." << sendl;
}


template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid)
{
    if(mparams)
    {
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
SReal Mass<DataTypes>::getKineticEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getKineticEnergy(mparams /* PARAMS FIRST */, *mparams->readV(this->mstate));
    return 0.0;
}

template<class DataTypes>
SReal Mass<DataTypes>::getKineticEnergy(const MechanicalParams* /*mparams*/, const DataVecDeriv& /*v*/) const
{
    serr << "ERROR("<<getClassName()<<"): getKineticEnergy(const MechanicalParams*, const DataVecDeriv& ) not implemented." << sendl;
    return 0.0;
}


template<class DataTypes>
SReal Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getPotentialEnergy(mparams /* PARAMS FIRST */, *mparams->readX(this->mstate));
    return 0.0;
}

template<class DataTypes>
SReal Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/) const
{
    serr << "ERROR("<<getClassName()<<"): getPotentialEnergy( const MechanicalParams*, const DataVecCoord& ) not implemented." << sendl;
    return 0.0;
}


template<class DataTypes>
defaulttype::Vector6 Mass<DataTypes>::getMomentum( const MechanicalParams* mparams ) const
{
    if (this->mstate)
        return getMomentum(mparams, *mparams->readX(this->mstate), *mparams->readV(this->mstate));
    return defaulttype::Vector6();
}

template<class DataTypes>
defaulttype::Vector6 Mass<DataTypes>::getMomentum( const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ ) const
{
    serr << "ERROR("<<getClassName()<<"): getMomentum( const MechanicalParams*, const DataVecCoord&, const DataVecDeriv& ) not implemented." << sendl;
    return defaulttype::Vector6();
}



template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addMToMatrix(r.matrix, mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()), r.offset);
}

template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, SReal /*mFact*/, unsigned int &/*offset*/)
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
void Mass<DataTypes>::exportGnuplot(const MechanicalParams* mparams, SReal time)
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
SReal Mass<DataTypes>::getElementMass(unsigned int ) const
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
