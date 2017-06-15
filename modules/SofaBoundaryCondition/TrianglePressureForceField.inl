/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TrianglePressureForceField_INL
#define SOFA_COMPONENT_FORCEFIELD_TrianglePressureForceField_INL

#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <SofaBaseTopology/TopologySparseData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>

// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

template <class DataTypes> TrianglePressureForceField<DataTypes>::~TrianglePressureForceField()
{
}

template <class DataTypes>  TrianglePressureForceField<DataTypes>::TrianglePressureForceField()    
        : pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
		, cauchyStress(initData(&cauchyStress, MatSym3(),"cauchyStress", "Cauchy Stress applied on the normal of each triangle"))
        , triangleList(initData(&triangleList,"triangleList", "Indices of triangles separated with commas where a pressure is applied"))
        , normal(initData(&normal,"normal", "Normal direction for the plane selection of triangles"))
        , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
        , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
        , p_showForces(initData(&p_showForces, (bool)false, "showForces", "draw triangles which have a given pressure"))
		, p_useConstantForce(initData(&p_useConstantForce, (bool)true, "useConstantForce", "applied force is computed as the the pressure vector times the area at rest"))
        , trianglePressureMap(initData(&trianglePressureMap, "trianglePressureMap", "map between edge indices and their pressure"))

    {
    }

template <class DataTypes> void TrianglePressureForceField<DataTypes>::init()
{

    this->core::behavior::ForceField<DataTypes>::init();
	
    _topology = this->getContext()->getMeshTopology();
    if(!_topology)
        serr << "Missing component: Unable to get MeshTopology from the current context. " << sendl;

    if(!_topology)
        return;

    if (dmin.getValue()!=dmax.getValue())
    {
        selectTrianglesAlongPlane();
    }
    if (triangleList.getValue().size()>0)
    {
        selectTrianglesFromString();
    }

    trianglePressureMap.createTopologicalEngine(_topology);
    trianglePressureMap.registerTopologicalData();
	
    initTriangleInformation();
		
}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::addForce(const core::MechanicalParams*  /*mparams*/, DataVecDeriv& d_f, const DataVecCoord&  d_x , const DataVecDeriv& /* d_v */)
{

    VecDeriv& f = *d_f.beginEdit();
    Deriv force;

	const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();

	if (p_useConstantForce.getValue()) {
		const sofa::helper::vector<TrianglePressureInformation>& my_subset = trianglePressureMap.getValue();


		for (unsigned int i=0; i<my_map.size(); ++i)
		{
			force=my_subset[i].force/3;
			f[_topology->getTriangle(my_map[i])[0]]+=force;
			f[_topology->getTriangle(my_map[i])[1]]+=force;
			f[_topology->getTriangle(my_map[i])[2]]+=force;

		}
	} else {
        //sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();
        typedef core::topology::BaseMeshTopology::Triangle Triangle;
		const sofa::helper::vector<Triangle> &ta = _topology->getTriangles();
		const  VecDeriv p = d_x.getValue();
        //Real area;
		MatSym3 cauchy=cauchyStress.getValue();
		Deriv areaVector,force;

		for (unsigned int i=0; i<my_map.size(); ++i)
		{
			const Triangle &t=ta[my_map[i]];
			areaVector=cross(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])/6.0f;
			force=cauchy*areaVector;
			for (size_t j=0;j<3;++j) {
				f[t[j]]+=force;
			}
		}
/*
		for (unsigned int i=0; i<my_map.size(); ++i)
		{
			const Triangle &t=ta[my_map[i]];
			// In case of implicit integration compute the area vector and the area and store it in the data structure 
			//	This will speed up the computation of the tangent stiffness matrix
			if (mparams->implicit()) {
				my_subset[i].force=cross(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]]);
				area=(my_subset[i].force).norm();
				my_subset[i].area=area;
				my_subset[i].force=my_subset[i].force/(2*area); // normalize normal vector and store it as a force
				force=pressure.getValue()*area/6.0f;
			} else {
				area=(Real)(sofa::component::topology::areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]]) * 0.5f);
				force=pressure.getValue()*area/3.0f;
			}

			f[t[0]]+=force;
			f[t[1]]+=force; 
			f[t[2]]+=force;

		}*/

         //trianglePressureMap.endEdit();
	}
    d_f.endEdit();
//    updateTriangleInformation();
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv&  /*d_df*/ , const DataVecDeriv&  /*d_dx*/ )
{
    //Warning fix : behaviorally equivalent without unused kfactor variable warning
	mparams->kFactor();
    //Real kfactor = mparams->kFactor();

	/*
	if (pressureScalar.getValue()!=0.0f) {

		size_t i,j,k;
		VecDeriv& df=*(d_df.beginEdit());
		const VecDeriv dx=d_dx.getValue();
		typedef sofa::component::topology::BaseMeshTopology::Triangle Triangle;
		const sofa::helper::vector<Triangle> &ta = _topology->getTriangles();
		 
		const  VecCoord p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
		const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();
		const sofa::helper::vector<TrianglePressureInformation>& my_subset = trianglePressureMap.getValue();

		Real press=	pressureScalar.getValue()/6.0f;
		Coord dp;
		
		for ( i=0; i<my_map.size(); ++i)
		{
			Deriv dForce;
			const Triangle &t=ta[my_map[i]];

			for (j=0;j<3;++j) {
				dp=p[t[(j+2)%3]]-p[t[(j+1)%3]];
				dForce+=cross(dp,dx[t[j]]);
			}
			dForce*=press*kfactor;
			for (j=0;j<3;++j) {
				df[t[j]]+=dForce;
			}
		}
//		for (i=0;i<df.size();++i) {
//			msg_info()<<"df["<<i<<"]= "<<df[i]<<std::endl;
//		} 
		d_df.endEdit();
	} */
/*
	if (p_definedOnRestPosition.getValue()==false) {
		size_t i,j,k;
		VecDeriv& df=*(d_df.beginEdit());
		const VecDeriv dx=d_dx.getValue();
		typedef sofa::component::topology::BaseMeshTopology::Triangle Triangle;
		const sofa::helper::vector<Triangle> &ta = _topology->getTriangles();
		 
		const  VecCoord p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
		const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();
		const sofa::helper::vector<TrianglePressureInformation>& my_subset = trianglePressureMap.getValue();

		Coord dp;
		Deriv dForce,press;
		Real dArea;
		Real KK,area;
		Coord areaVec,dareaVec,dareaVec2;
		press=pressure.getValue();

		for ( i=0; i<my_map.size(); ++i)
		{
			dArea=0;
			dareaVec=Coord();
			const Triangle &t=ta[my_map[i]];
			for (j=0;j<3;++j) {
				dp=p[t[(j+2)%3]]-p[t[(j+1)%3]];
				dareaVec2=cross(dp,dx[t[j]]);
				//dArea+=dot(my_subset[i].force,cross(dp,dx[t[j]]));
				dArea+=dot(my_subset[i].force,dareaVec2);//+dareaVec2.norm2()/(4*my_subset[i].area);
				dareaVec+=dareaVec2;
	//			dArea+=dot(dx[t[j]],cross(my_subset[i].force,dp));
				//dp=dx[t[j]];
			}
//			dArea+=dareaVec.norm2()/(4*my_subset[i].area);
			Real dAreatest=((cross(p[t[1]]-p[t[0]]+dx[t[1]]-dx[t[0]],p[t[2]]-p[t[0]]+dx[t[2]]-dx[t[0]])).norm()-(cross(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])).norm())/2.0;
			if (fabs(dArea-dAreatest)>1e-3){
			}
			dForce=press*dArea*kfactor/3;
			for (j=0;j<3;++j) 
				df[t[j]]+=dForce;

		}
		d_df.endEdit();
	} 
*/

	return;
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::initTriangleInformation()
{
   this->getContext()->get(triangleGeo);

    if(!triangleGeo)
        serr << "Missing component: Unable to get TriangleSetGeometryAlgorithms from the current context." << sendl;

    // FIXME: a dirty way to avoid a crash
    if(!triangleGeo)
        return;

    const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        my_subset[i].area=triangleGeo->computeRestTriangleArea(my_map[i]);
        my_subset[i].force=pressure.getValue()*my_subset[i].area;
    }

    trianglePressureMap.endEdit();
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::updateTriangleInformation()
{
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();

    for (unsigned int i=0; i<my_subset.size(); ++i)
        my_subset[i].force=(pressure.getValue()*my_subset[i].area);

    trianglePressureMap.endEdit();
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    std::vector<bool> vArray;
    unsigned int i;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();
    helper::vector<unsigned int> inputTriangles;

    for (int n=0; n<_topology->getNbTriangles(); ++n)
    {
        if ((vArray[_topology->getTriangle(n)[0]]) && (vArray[_topology->getTriangle(n)[1]])&& (vArray[_topology->getTriangle(n)[2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            t.area = 0;
            my_subset.push_back(t);
            inputTriangles.push_back(n);
        }
    }
    trianglePressureMap.endEdit();
    trianglePressureMap.setMap2Elements(inputTriangles);

    return;
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesFromString()
{
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();
    helper::vector<unsigned int> _triangleList = triangleList.getValue();

    trianglePressureMap.setMap2Elements(_triangleList);

    for (unsigned int i = 0; i < _triangleList.size(); ++i)
    {
        TrianglePressureInformation t;
        t.area = 0;
        my_subset.push_back(t);
    }

    trianglePressureMap.endEdit();

    return;
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!p_showForces.getValue())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    glColor4f(0,1,0,1);

    const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[0]]);
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[1]]);
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[2]]);
    }
    glEnd();


    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TrianglePressureForceField_INL
