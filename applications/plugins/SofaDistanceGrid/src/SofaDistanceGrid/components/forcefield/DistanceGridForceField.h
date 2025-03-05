/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_H
#include <SofaDistanceGrid/config.h>
#include <SofaDistanceGrid/DistanceGrid.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/// This class can be overridden if needed for additional storage within template specializations.
template<class DataTypes>
class DistanceGridForceFieldInternalData
{
public:
};

template<class DataTypes>
class DistanceGridForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DistanceGridForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef container::DistanceGrid DistanceGrid;

protected:
    DistanceGrid* grid;

    class Contact
    {
    public:
        int index;
        Coord normal;
        Real fact;
        Contact( int index=0, Coord normal=Coord(),Real fact=Real(0))
            : index(index),normal(normal),fact(fact)
        {
        }

        inline friend std::istream& operator >> ( std::istream& in, Contact& c )
        {
            in>>c.index>>c.normal>>c.fact;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Contact& c )
        {
            out << c.index << " " << c.normal << " " << c.fact ;
            return out;
        }

    };

    sofa::type::vector<bool> pOnBorder;
    Data<sofa::type::vector<Contact> > contacts;


    class TContact
    {
    public:
        type::fixed_array<unsigned int,3> index;
        Coord normal,B,C;
        Real fact;

        inline friend std::istream& operator >> ( std::istream& in, TContact& c )
        {
            in>>c.index>>c.normal>>c.B>>c.C>>c.fact;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const TContact& c )
        {
            out << c.index << " " << c.normal << " " << c.B << " " << c.C << " " << c.fact ;
            return out;
        }

    };

    Data<sofa::type::vector<TContact> > tcontacts;

    class VContact
    {
    public:
        type::fixed_array<unsigned int,4> index;
        Coord A,B,C;
        Real fact;

        inline friend std::istream& operator >> ( std::istream& in, VContact& c )
        {
            in>>c.index>>c.A>>c.B>>c.C>>c.fact;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const VContact& c )
        {
            out << c.index << " " << c.A << " " << c.B << " " << c.C << " " << c.fact ;
            return out;
        }

    };

    Data<sofa::type::vector<VContact> > vcontacts;

    DistanceGridForceFieldInternalData<DataTypes> data;

public:

    // Input data parameters
    sofa::core::objectmodel::DataFileName fileDistanceGrid; ///< load distance grid from specified file
    Data< double > scale; ///< scaling factor for input file
    Data< type::fixed_array<DistanceGrid::Coord,2> > box; ///< Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax
    Data< int > nx; ///< number of values on X axis
    Data< int > ny; ///< number of values on Y axis
    Data< int > nz; ///< number of values on Z axis

    Data<Real> stiffnessIn; ///< force stiffness when inside of the object
    Data<Real> stiffnessOut; ///< force stiffness when outside of the object
    Data<Real> damping; ///< force damping coefficient
    Data<Real> maxDist; ///< max distance of the surface after which no more force is applied
    Data<Real> minArea; ///< minimal area for each triangle, as seen from the direction of the local surface (i.e. a flipped triangle will have a negative area)
    Data<Real> stiffnessArea; ///< force stiffness if a triangle have an area less than minArea
    Data<Real> minVolume; ///< minimal volume for each tetrahedron (a flipped triangle will have a negative volume)
    Data<Real> stiffnessVolume; ///< force stiffness if a tetrahedron have an volume less than minVolume
    bool flipNormals;

    Data<sofa::type::RGBAColor> color; ///< display color.(default=[0.0,0.5,0.2,1.0])
    Data<bool> bDraw; ///< enable/disable drawing of distancegrid
    Data<bool> drawPoints; ///< enable/disable drawing of distancegrid
    Data<Real> drawSize; ///< display size if draw is enabled

    /// optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitioning)
    Data< type::Vec<2,int> > localRange;
protected:
    DistanceGridForceField()
        : grid(NULL)
        , fileDistanceGrid( initData( &fileDistanceGrid, "filename", "load distance grid from specified file"))
        , scale( initData( &scale, 1.0, "scale", "scaling factor for input file"))
        , box( initData( &box, "box", "Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
        , nx( initData( &nx, 64, "nx", "number of values on X axis") )
        , ny( initData( &ny, 64, "ny", "number of values on Y axis") )
        , nz( initData( &nz, 64, "nz", "number of values on Z axis") )
        , stiffnessIn(initData(&stiffnessIn, (Real)500, "stiffnessIn", "force stiffness when inside of the object"))
        , stiffnessOut(initData(&stiffnessOut, (Real)0, "stiffnessOut", "force stiffness when outside of the object"))
        , damping(initData(&damping, (Real)0.01, "damping", "force damping coefficient"))
        , maxDist(initData(&maxDist, (Real)1.0, "maxdist", "max distance of the surface after which no more force is applied"))
        , minArea(initData(&minArea, (Real)0, "minArea", "minimal area for each triangle, as seen from the direction of the local surface (i.e. a flipped triangle will have a negative area)"))
        , stiffnessArea(initData(&stiffnessArea, (Real)100, "stiffnessArea", "force stiffness if a triangle have an area less than minArea"))
        , minVolume(initData(&minVolume, (Real)0, "minVolume", "minimal volume for each tetrahedron (a flipped triangle will have a negative volume)"))
        , stiffnessVolume(initData(&stiffnessVolume, (Real)0, "stiffnessVolume", "force stiffness if a tetrahedron have an volume less than minVolume"))
        , color(initData(&color, sofa::type::RGBAColor(0.0f,0.5f,0.2f,1.0f), "color", "display color.(default=[0.0,0.5,0.2,1.0])"))
        , bDraw(initData(&bDraw, false, "draw", "enable/disable drawing of distancegrid"))
        , drawPoints(initData(&drawPoints, false, "drawPoints", "enable/disable drawing of distancegrid"))
        , drawSize(initData(&drawSize, (Real)10.0f, "drawSize", "display size if draw is enabled"))
        , localRange( initData(&localRange, type::Vec<2,int>(-1,-1), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitioning)" ) )
    {
        this->addAlias(&stiffnessIn,"stiffness");
        this->addAlias(&stiffnessOut,"stiffness");
        this->addAlias(&fileDistanceGrid,"fileDistanceGrid");
    }
public:
    void init() override;

    void setMState(  core::behavior::MechanicalState<DataTypes>* mstate ) { this->mstate = mstate; }

    void setStiffness(Real stiffIn, Real stiffOut)
    {
        stiffnessIn.setValue( stiffIn );
        stiffnessOut.setValue( stiffOut );
    }

    void setDamping(Real damp)
    {
        damping.setValue( damp );
    }

    void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Get potentialEnergy not implemented";
        return 0.0;
    }
    void draw(const core::visual::VisualParams* vparams) override;
    void drawDistanceGrid(const core::visual::VisualParams*, float size=0.0f);


};

#if  !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_CPP)
extern template class SOFA_SOFADISTANCEGRID_API DistanceGridForceField<defaulttype::Vec3Types>;
//extern template class SOFA_SOFADISTANCEGRID_API DistanceGridForceField<defaulttype::Vec2Types>;
//extern template class SOFA_SOFADISTANCEGRID_API DistanceGridForceField<defaulttype::Vec1Types>;

#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_H
