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
#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_H
#define SOFA_COMPONENT_MASS_UNIFORMMASS_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

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

public:
    UniformMass();

    ~UniformMass();

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

    void addMDx(DataVecDeriv& f, const DataVecDeriv& dx, double factor, const core::MechanicalParams* mparams);

    void accFromF(DataVecDeriv& a, const DataVecDeriv& f, const core::MechanicalParams* mparams);

    void addForce(DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v, const core::MechanicalParams* mparams);

    double getKineticEnergy(const DataVecDeriv& v, const core::MechanicalParams* mparams) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const DataVecCoord& x, const core::MechanicalParams* mparams) const;   ///< Mgx potential in a uniform gravity field, null at origin

    void addMDxToVector(defaulttype::BaseVector *resVect, const VecDeriv *dx, SReal mFact, unsigned int& offset);

    void addGravityToV(core::MultiVecDerivId vid, const core::MechanicalParams* mparams);

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const core::MechanicalParams *mparams);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    bool isDiagonal() {return true;};

    void draw();

    bool addBBox(double* minBBox, double* maxBBox);
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_MASS_UNIFORMMASS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec3dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec2dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec1dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec6dTypes,double>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3dMass>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Rigid2dTypes,defaulttype::Rigid2dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec3fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec2fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec1fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec6fTypes,float>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Rigid3fTypes,defaulttype::Rigid3fMass>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Rigid2fTypes,defaulttype::Rigid2fMass>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif

