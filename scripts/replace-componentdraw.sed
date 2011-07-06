#!/bin/sed -f 
#README:
#sed -f replace-componentdraw.sed < testfile > testoutput
#testfile
#TEST FILE FOR REPLACECOMPONENTDRAW.SED
# void draw ();
#  void draw (    );
#    void draw ( int );
#      void draw (  int idx    );
#          simulation::getSimulation()->DrawUtility().drawArrow(p0,p1,p2,p3);
#     sofa::simulation::getSimulation()->DrawUtility().drawArrow(p0,p1,p2,p3);
#	void Component::draw ()
#	{
#		TriangleSetGeometryAlgorithms<DataTypes>::draw ();
#	}
#		void Component::draw(   )
#		{
#		TriangleSetGeometryAlgorithms< DataTypes >::draw (   );
#		}
#	void Component::draw ( int index )
#	{
#	  CollisionModel::draw ( index  );
#	}
#	void Component::draw (int   i)
#	{
#	draw (  );
#	draw (i); it2->elem.first.draw ();
#	it2->elem.first.draw ();
#	}
# 	void draw ( const typename Out::VecCoord& out , const typename In::VecCoord& in ) ;
#	template <class In, class Out>
#void BarycentricMapperMeshTopology<In,Out>::draw ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
#	if ( mapper!=NULL ) mapper->draw ( out, in );
#	template<>
#	void MechanicalObject<defaulttype::LaparoscopicRigid3Types>::draw();
#	template<class DataTypes>
#	void DistanceOnGrid< DataTypes >::draw()
#	grid->draw();
#	template <>
#void UniformMass<Rigid2dTypes, Rigid2dMass>::draw();



#void draw(int index) --> draw(const sofa::core::visual::VisualParams*, int index)
#void draw(int )      --> draw(const sofa::core::visual::VisualParams*, int)
s/\(void[[:blank:]]\)\([[:blank:]]*\)\(draw[[:blank:]]*\)(\([[:blank:]]*\)\([[:graph:]][[:graph:]]*\)\(.*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1\2\3(const sofa::core::visual::VisualParams*,\5\6);/g
# void draw() --> void draw(const sofa::core::visual::VisualParams* vparams)
s/\(void[[:blank:]]\)\([[:blank:]]*\)\(draw[[:blank:]]*\)(\([[:blank:]]*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1\2\3(const sofa::core::visual::VisualParams*);/g

#void Component::draw() --> void Component::draw(const sofa::core::visual::VisualParams* vparams)
#void BarycentricMapperMeshTopology<In,Out>::draw(const typename Out::VecCoord& out, const typename In::VecCoord& in ) 
# --> void BarycentricMapperMeshTopology<In,Out>::draw (const sofa::core::visual::VisualParams* vparams,const typename Out::VecCoord& out, const typename In::VecCoord& in )
s/\(void[[:blank:]][[:blank:]]*\)\([[:graph:]][[:print:]]*\)\(::draw[[:blank:]]*\)(\([[:blank:]]*\)\([[:graph:]][[:graph:]]*\)\(.*\))\([[:blank:]]*\)$/\1\2\3(const sofa::core::visual::VisualParams* vparams,\4\5\6\7)/g
s/\(void[[:blank:]][[:blank:]]*\)\([[:graph:]][[:print:]]*\)\(::draw[[:blank:]]*\)(\([[:blank:]]*\))\([[:blank:]]*\)$/\1\2\3\4(const sofa::core::visual::VisualParams* vparams)/g
s/\(void[[:blank:]][[:blank:]]*\)\([[:graph:]][[:print:]]*\)\(::draw[[:blank:]]*\)(\([[:blank:]]*\)\([[:graph:]][[:graph:]]*\)\(.*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1\2\3\4(const sofa::core::visual::VisualParams* vparams,\5\6\7);/g
s/\(void[[:blank:]][[:blank:]]*\)\([[:graph:]][[:print:]]*\)\(::draw[[:blank:]]*\)(\([[:blank:]]*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1\2\3\4(const sofa::core::visual::VisualParams* vparams);/g


#::draw( index  ); --> ::draw(vparams,index);
#mapper->draw ( out, in ); --> mapper->draw(vparams,out, in );
s/\([[:graph:]][[:graph:]]*\)\(draw[[:blank:]]*\)(\([[:blank:]]*\)\([[:graph:]][[:print:]]*\)\(.*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1draw(vparams,\4\5);/g
#TriangleSetGeometryAlgorithms<DataTypes>::draw(); --> TriangleSetGeometryAlgorithms<DataTypes>::draw(vparams);
#it2->elem.first.draw(); --> it2->elem.first.draw();
s/\([[:graph:]][[:print:]]*\)\(draw[[:blank:]]*\)(\([[:blank:]]*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1draw(vparams);/g

s/\(void[[:blank:]][[:blank:]]*\)\([[:graph:]][[:print:]]*\)\(::draw[[:blank:]]*\)(\([[:blank:]]*\)\(vparams,\)\([[:blank:]]*\)\([[:graph:]][[:graph:]]*\)\(.*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1\2\3(\6\7\8);/g


# getSimulation()->drawUtility().                   --> vparams->drawTool()->
# simulation::getSimulation()->drawUtility().       --> vparams->drawTool()->
# sofa::simulation::getSimulation()->drawUtility(). --> vparams->drawTool()->
s/[[:alnum:]][[:graph:]]*->DrawUtility()\./vparams->drawTool()->/g

# draw(); --> draw(vparams);
# draw(i); --> draw(vparams,i);
s/\(^[[:blank:]]*\)\(draw[[:blank:]]*\)(\([[:blank:]]*\)\([[:alnum:]][[:graph:]]*\)\(.*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1draw(vparams,\4\5);/g
s/\(^[[:blank:]]*\)\(draw[[:blank:]]*\)(\([[:blank:]]*\))\([[:blank:]]*\);\([[:blank:]]*\)$/\1draw(vparams);/g

# Revert duplicated changes
s/\(([[:space:]]*\)\(const sofa::core::visual::VisualParams\*,\|const sofa::core::visual::VisualParams\* vparams,\)\([^)]*core::visual::VisualParams\*\)/\1\3/g
s/vparams,vparams/vparams/g

# Revert changes to Axis::draw(), Cylinder::draw(), GlText::draw(), BBox.draw(), getSimulation()->draw()
s/\(Axis::\|Cylinder::\|GlText::\|getSimulation()->\)draw(vparams,/\1draw(/g
# Revert changes to BBox.draw(), vmsubvol[i].draw()
s/\(BBox\.\|vmsubvol\[i\]\.\)draw(vparams)/\1draw()/g
