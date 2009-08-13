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
#ifndef SOFA_COMPONENT_COLLISION_LOCALMINDISTANCEFILTER_H
#define SOFA_COMPONENT_COLLISION_LOCALMINDISTANCEFILTER_H

#include <sofa/component/component.h>

#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace collision
{

class LocalMinDistanceFilter;

/**
 * @brief LocalMinDistance cone information class for an unique collision primitive.
 */
template< class TCollisionElement >
class InfoFilter
{
public:
    /**
     * @brief Default constructor.
     *
     * @param m_revision Cone information is up to date.
     * @param m_rigid Cone information is related to a rigid CollisionModel.
     * @param m_lmdFilters The localMinDistance filtration class that contains this InfoFilter.
     */
    InfoFilter(LocalMinDistanceFilter *lmdFilters)
        :	m_revision(-1),
            m_rigid(false),
            m_lmdFilters(lmdFilters)
    {};

    /**
     * @brief Default destructor.
     */
    virtual ~InfoFilter() {};

    /**
     * @brief Returns the validity of a detected contact according to the InfoFilter.
     */
    virtual bool validate(const TCollisionElement & , const defaulttype::Vector3 & /*PQ*/) = 0;

    /**
     * @brief Returns cone information validity (up to date or not?).
     */
    bool isValid(void) const;

    /**
     * @brief Sets cone information validity.
     */
    void setValid();

    /**
     * @brief Returns true if the CollisionElement is mapped to a rigid mechanical state.
     */
    bool isRigid(void) const {return m_rigid;};

    /**
     * @brief Sets the rigid property, true if the CollisionElement is mapped to a rigid mechanical state.
     */
    void setRigid(const bool rigid) {m_rigid = rigid;};

    /**
     * @brief Returns the LocalMinDistanceFilters object that contains this InfoFilter.
     */
    LocalMinDistanceFilter *getLMDFilters(void) const {return m_lmdFilters;};

    /**
     * @brief Sets the LocalMinDistanceFilters object that contains this InfoFilter.
     */
    void setLMDFilters(const LocalMinDistanceFilter *lmdFilters) {m_lmdFilters = lmdFilters;};


protected:
    /**
     * @brief If InfoFilter data is invalid, computes the region of interest cone of the collision primitive.
     *
     * If the collision primitive is mapped to a rigid MState, the computation is only an update according to the
     * rigid transformation.
     */
    virtual void buildFilter(const TCollisionElement & ) = 0;

    int m_revision; ///< Last filter update revision.
    bool m_rigid; ///< True if the CollisionElement is mapped to a rigid mechanical state.
    const LocalMinDistanceFilter	*m_lmdFilters; ///< The LocalMinDistanceFilters object that contains this InfoFilter.
};


/**
 * @brief Collision detection validation class using cones ROI method.
 *
 * This class is used by detection collision methods such as Proximity detection to tests if a detected
 * contact should be kept for collision response or not.
 * It also manages an history of computed cones that allows faster computation for an already tested collision
 * primitive.
 */
class SOFA_COMPONENT_COLLISION_API LocalMinDistanceFilter : public virtual core::objectmodel::BaseObject
{
public:
    /**
     * @brief Default constructor.
     */
    LocalMinDistanceFilter();

    /**
     * @brief Default destructor.
     */
    virtual ~LocalMinDistanceFilter();

    /**
     * @brief Get filtering cone extension angle.
     */
    double getConeExtension(void) const {return m_coneExtension.getValue();};

    /**
     * @brief Set filtering cone extension angle.
     */
    void setConeExtenstion(const double coneExtension) {m_coneExtension.setValue(coneExtension);};

    /**
     * @brief Get the minimal filtering cone angle value, independently from geometry.
     */
    double getConeMinAngle(void) const {return m_coneMinAngle.getValue();};

    /**
     * @brief Set the minimal filtering cone angle value, independently from geometry.
     */
    void setConeMinAngle(const double coneMinAngle) {m_coneMinAngle.setValue(coneMinAngle);};

    /**
     * @brief Returns the current CollisionModel update index.
     */
    int getRevision(void) const {return m_revision;};

    /**
     * @brief Sets the current CollisionModel update index.
     */
    void setRevision(const int revision) {m_revision = revision;};

    /**
     * @brief Increases LMDFilter revision number to notify a CollisionModel modification.
     * Corresponding filtrations data should be rebuilt or updated.
     */
    void invalidate();

private:

    Data< double >	m_coneExtension;	///< Filtering cone extension angle.
    Data< double >	m_coneMinAngle;		///< Minimal filtering cone angle value, independent from geometry.
    unsigned int	m_revision;			///< Simulation step index (CollisionModel revision).
};


/**
 * @brief
 */
class SOFA_COMPONENT_COLLISION_API EmptyFilter
{
public:
    /**
     * @brief Point Collision Primitive validation method.
     */
    bool validPoint(const int /*pointIndex*/, const defaulttype::Vector3 &/*PQ*/)
    {
        return true;
    }

    /**
     * @brief Line Collision Primitive validation method.
     */
    bool validLine(const int /*lineIndex*/, const defaulttype::Vector3 &/*PQ*/)
    {
        return true;
    }

    /**
     * @brief Triangle Collision Primitive validation method.
     */
    bool validTriangle(const int /*triangleIndex*/, const defaulttype::Vector3 &/*PQ*/)
    {
        return true;
    }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_LOCALMINDISTANCEFILTER_H
