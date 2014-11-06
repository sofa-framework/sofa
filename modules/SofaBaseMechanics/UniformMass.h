/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_H
#define SOFA_COMPONENT_MASS_UNIFORMMASS_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/SofaBase.h>

namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class TMassType>
class UniformMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(UniformMass,DataTypes,TMassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef TMassType MassType;

//protected:
    Data<MassType> mass;    ///< the mass of each particle
    Data<double> totalMass; ///< if >0 : total mass of this body
    sofa::core::objectmodel::DataFileName filenameMass; ///< a .rigid file to automatically load the inertia matrix and other parameters
    /// to display the center of gravity of the system
    Data< bool > showCenterOfGravity;
    Data< float > showAxisSize;

    Data<bool> compute_mapping_inertia;
    Data<bool> showInitialCenterOfGravity;

    /// to display the rest positions
    Data< bool > showX0;

    /// optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)
    Data< defaulttype::Vec<2,int> > localRange;

    Data< bool > m_handleTopoChange;

protected:
    UniformMass();

    ~UniformMass();
public:
    void setMass(const MassType& mass);
    const MassType& getMass() const { return mass.getValue(); }

    double getTotalMass() const { return totalMass.getValue(); }
    void setTotalMass(double m);

    void setFileMass(const std::string& file) {filenameMass.setValue(file);}
    std::string getFileMass() const {return filenameMass.getFullPath();};

    void loadRigidMass(std::string filename);
    // -- Mass interface

    void reinit();
    void init();

    void handleTopologyChange();

    void addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

    void accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

    void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    double getKineticEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecDeriv& v) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    defaulttype::Vec6d getMomentum(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x, const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)

    void addMDxToVector(defaulttype::BaseVector *resVect, const VecDeriv *dx, SReal mFact, unsigned int& offset);

    void addGravityToV(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_v);

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const core::MechanicalParams *mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    bool isDiagonal() {return true;};

    void draw(const core::visual::VisualParams* vparams);
};

//Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::reinit();
template <>
void UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::loadRigidMass ( std::string );
template <>
void UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::draw(const core::visual::VisualParams* vparams);
template <>
void UniformMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>::draw(const core::visual::VisualParams* vparams);
template <>
double UniformMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3dMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& x ) const;
template <>
double UniformMass<defaulttype::Rigid2dTypes,defaulttype::Rigid2dMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& x ) const;
template <>
void UniformMass<defaulttype::Vec6dTypes,double>::draw(const core::visual::VisualParams* vparams);
#endif
#ifndef SOFA_DOUBLE
template<>
void UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::reinit();
template<>
void UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::loadRigidMass ( std::string );
template <>
void UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::draw(const core::visual::VisualParams* vparams);
template <>
void UniformMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>::draw(const core::visual::VisualParams* vparams);
template <>
double UniformMass<defaulttype::Rigid3fTypes,defaulttype::Rigid3fMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& x ) const;
template <>
double UniformMass<defaulttype::Rigid2fTypes,defaulttype::Rigid2fMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& x ) const;
template <>
void UniformMass<defaulttype::Vec6fTypes,float>::draw(const core::visual::VisualParams* vparams);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MASS_UNIFORMMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec3dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec2dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec1dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec6dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec3fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec2fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec1fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec6fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif

