/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYMODIFIER_H

#include <sofa/helper/vector.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/component/topology/PointSetTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class PointSetTopology;

template< typename DataTypes >
class PointSetTopologyLoader;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID PointID;

/**
* A class that can apply basic topology transformations on a set of points.
*/
template<class DataTypes>
class PointSetTopologyModifier : public core::componentmodel::topology::TopologyModifier
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    PointSetTopologyModifier()
        : TopologyModifier()
    {}

    PointSetTopologyModifier(core::componentmodel::topology::BaseTopology *top)
        : TopologyModifier(top)
    {}

    virtual ~PointSetTopologyModifier() {}

    PointSetTopology<DataTypes>* getPointSetTopology() const
    {
        return static_cast<PointSetTopology<DataTypes>*> (this->m_basicTopology);
    }

    /** \brief Build a point set topology from a file : also modifies the MechanicalObject
    *
    */
    virtual bool load(const char *filename);

    /** \brief Swap points i1 and i2.
    *
    */
    virtual void swapPoints(const int i1,const int i2);

    /** \brief Translates the DOF : call the applyTranslation member function in the MechanicalObject
    *
    */
    virtual void applyTranslation (const double dx,const double dy,const double dz);

    /** \brief Scales the DOF : call the applyScale member function in the MechanicalObject object
    *
    */
    virtual void applyScale (const double s);

    /** \brief Sends a message to warn that some points were added in this topology.
    *
    * \sa addPointsProcess
    */
    void addPointsWarning(const unsigned int nPoints, const bool addDOF = true);

    /** \brief Sends a message to warn that some points were added in this topology.
    *
    * \sa addPointsProcess
    */
    void addPointsWarning(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double       > >& coefs,
            const bool addDOF = true);


    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    *
    * @param addDOF if true the points are actually added from the mechanical object's state vectors
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints, const bool addDOF = true);

    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    *
    * @param addDOF if true the points are actually added from the mechanical object's state vectors
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool addDOF = true);

    /** \brief Add a new point (who has no ancestors) to this topology.
    *
    * \sa addPointsWarning
    */
    virtual void addNewPoint(unsigned int i,  const sofa::helper::vector< double >& x);

    /** \brief Sends a message to warn that some points are about to be deleted.
    *
    * \sa removePointsProcess
    */
    // side effect: indices are sorted first
    void removePointsWarning(/*const*/ sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);


    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed from the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    *
    * @param indices is not const because it is actually sorted from the highest index to the lowest one.
    * @param removeDOF if true the points are actually deleted from the mechanical object's state vectors
    */
    virtual void removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);


    /** \brief Sends a message to warn that points are about to be reordered.
    *
    * \sa renumberPointsProcess
    */
    void renumberPointsWarning( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &inv_index,
            const bool renumberDOF = true);

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &/*inv_index*/,
            const bool renumberDOF = true);

protected:
    /// modifies the mechanical object and creates the point set container
    void loadPointSet(PointSetTopologyLoader<DataTypes> *);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYMODIFIER_H
