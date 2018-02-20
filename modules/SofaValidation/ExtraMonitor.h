/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_EXTRAMONITOR_H
#define SOFA_COMPONENT_MISC_EXTRAMONITOR_H
#include "config.h"

#include <SofaValidation/Monitor.h>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
class ExtraMonitor : public virtual Monitor<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ExtraMonitor,DataTypes), SOFA_TEMPLATE(Monitor,DataTypes));

    typedef Monitor<DataTypes> Inherit;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    ExtraMonitor();
public:
    virtual void init() override;

    //	virtual void reset();

    //	virtual void reinit();

    virtual void handleEvent( core::objectmodel::Event* ev ) override;

    ///create gnuplot files
    virtual void initGnuplot ( const std::string path ) override;

    ///write in gnuplot files the Monitored desired data (velocities,positions,forces)
    virtual void exportGnuplot ( Real time ) override;

    /// Editable Data
    Data< bool > saveWcinToGnuplot;
    Data< bool > saveWextToGnuplot; ///< export Wext of the monitored dofs as gnuplot file

    /// to compute the forces resultant on the monitored dof
    /// used only if saveFToGnuplot is set to true (ExportForces)
    Data< bool > resultantF;

    /// to get the minimum displacement of the monitored dof on a given coordinate
    /// used only if savePToGnuplot is set to true (ExportPositions)
    Data< int > minX;
    /// to get the maximum displacement of the monitored dof on a given coordinate
    /// used only if savePToGnuplot is set to true (ExportPositions)
    Data< int > maxX;

    /// to get the displacement of the set of dofs on a given coordinate
    /// used only if savePToGnuplot is set to true (ExportPositions)
    Data< int > disp;

protected:
    std::ofstream* saveGnuplotWcin;
    std::ofstream* saveGnuplotWext;

    /// store the initial position on the considered coordinate for min computation
    sofa::helper::vector< Real > initialMinPos;
    /// store the initial position on the considered coordinate for max computation
    /// the two versions are for the case we set minX != maxX
    sofa::helper::vector< Real > initialMaxPos;

    /// store the initial position on the considered coordinate for the monitored dofs
    sofa::helper::vector< Real > initialPos;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MISC_EXTRAMONITOR_CPP)
#ifndef SOFA_FLOAT
extern template class ExtraMonitor<defaulttype::Vec3dTypes>;
extern template class ExtraMonitor<defaulttype::Vec6dTypes>;
extern template class ExtraMonitor<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class ExtraMonitor<defaulttype::Vec3fTypes>;
extern template class ExtraMonitor<defaulttype::Vec6fTypes>;
extern template class ExtraMonitor<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace misc

} // namespace component

} // namespace sofa

#endif
