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
 * Forcefields, Constraints which acts only on a little number of particles should activate the mask (by redefining the method bool useMask()) and add entries in the particle mask
 */
class StateMask
{
public:
    typedef helper::set< size_t > InternalStorage;
    StateMask(Data<bool> *activator):inUse(activator), activated(true) {}

    /// Insert an entry in the mask
    void insertEntry(unsigned int index)
    {
        indices.insert(index);
    }

    const InternalStorage &getEntries() const { return indices; }

    /// Explicit activation: when during some process we need the mask, we activate it
    void activate(bool a)
    {
        activated = a;
    }

    /// Test if the mask can be used:
    ///    * the parameter of the BaseMechanicalState useMask must be active
    ///    * we must be inside a process using the mask
    ///    * all the components of the node must use the mask. If a single one has deactivated its mask, we can't use the mask for the whole node.
    bool isInUse() const
    {
        return inUse->getValue() && activated;
    }

    void clear()
    {
        indices.clear();
        activated=true;
    }

protected:
    InternalStorage indices;
    // Act as a switch, to enable or not the mask.
    Data<bool> *inUse; // manual switch
    bool activated; // automatic switch (the mask is only used for specific operations)

};


} // namespace helper

} // namespace sofa

#endif
