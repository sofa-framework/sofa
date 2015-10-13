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
#ifndef SOFA_HELPER_PARTICLEMASK_H
#define SOFA_HELPER_PARTICLEMASK_H
#include <sofa/helper/vector.h>

#include <sofa/defaulttype/Mat.h>
#ifdef Success
#undef Success // dirty workaround to cope with the (dirtier) X11 define. See http://eigen.tuxfamily.org/bz/show_bug.cgi?id=253
#endif
#include <Eigen/Dense>


namespace sofa
{

namespace helper
{

/**
 *  \brief Utility class to handle the mechanism of masks.
 *
 *  One of the most time-consuming process in Sofa is the transmission of forces and velocities through the mappings.
 *  If only a little subset of particles are used, we would like to propagate those forces (applyJT), and velocities (applyJ) to this subset only.
 *
 *  This class is used inside the BaseMechanicalState.
 *
 *  USAGE:
 *
 *     - Forcefields, Constraints
 *              which acts only on a little number of dofs should use the mask by only adding active entries
 *              in the fonction updateForceMask() (the default implementation adds every dofs in the mask)
 *
 *     - (Multi)Mappings
 *              they must propagate the mask from their child (tomodel) to their parents (frommodels)
 *              ApplyJ shoud use getActivatedEntry to check if a child dof is active (as in some case, every dofs must be updated, do not use unsafe getEntry)
 *              ApplyJT shoud use getEntry to check if a child dof is active and CAN insert parent dofs in the parent mask.
 *              ApplyDJT, getJ/getJs shoud use getEntry to check if a child dof is active
 *              updateForceMask() must insert only active parent dofs in the parent mask (or should add nothing if parents have already been added in ApplyJT)
 *
 */
class SOFA_HELPER_API StateMask
{

public:

    typedef helper::vector<bool> InternalStorage; // note this should be space-optimized (a bool = a bit) in the STL

    StateMask() : activated(false) {}

    /// filling-up (and eventuelly resize) the mask
    void assign( size_t size, bool value );

    /// the mask can be deactivated when the mappings must be applied to every dofs (e.g. propagatePosition)
    /// it must be activated when the mappings can be limited to active dofs
    void activate( bool a );
    bool isActivated() const { return activated; }

    /// add the given dof index in the mask
    void insertEntry( size_t index ) { mask[index]=true; }

    /// is the given dof index in the mask?
    /// @warning always returns the mask value w/o checking if the mask is activated (for Mapping::applyJT/getJs)
    bool getEntry( size_t index ) const { return mask[index]; } // unsafe to be use where we do not care if the mapping in deactivated

    /// is the given dof index activated? If the mask is not activated it always returns true, otherwise it gives the real mask value.
    /// useful for Mapping::applyJ
    bool getActivatedEntry( size_t index ) const;

    /// getting mask entries is useful for advanced uses.
    const InternalStorage& getEntries() const { return mask; }

    void resize( size_t size );
    void clear() { mask.clear(); }
    size_t size() const { return mask.size(); }

    inline friend std::ostream& operator<< ( std::ostream& os, const StateMask& sm )
    {
        return os << sm.mask;
    }


    /// get the mask converted to a eigen vector
    /// useful to build a projection matrix
    template<class Real>
    Eigen::Matrix<Real, Eigen::Dynamic, 1> toEigenVec() const
    {
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> vec;
        vec v( size() );
        for( size_t i=0 ; i<size() ; ++i )
            v[i] = mask[i] ? (Real)1 : (Real)0;
        return v;
    }


protected:

    InternalStorage mask; // note this should be space-optimized (a bool = a bit) in the STL
    bool activated; // automatic switch (the mask is only used for specific operations)

};


} // namespace helper

} // namespace sofa

#endif
