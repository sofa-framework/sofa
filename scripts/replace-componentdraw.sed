#!bin/sed -f 
# void draw() --> void draw(const core::visual::VisualParams* vparams)
s/\(void.*\)\(draw[[:blank:]]*([[:blank:]]*)\)/\1draw(const core::visual::VisualParams* vparams)/g
#void draw(int index) --> draw(int index, const core::visual::VisualParams* )
#void draw(int )      --> draw(int, const core::visual::VisualParams*)
s/\(void.*\)\(draw[[:blank:]]*([[:blank:]]*int[[:blank:]]*\)\([[:alnum:]]*\)\([[:blank:]]*)\)/\1\2\3, const core::visual::VisualParams*)/g

#CollisionModel::draw( index  ); --> CollisionModel::draw(index,vparams);
s/\([[:graph:]]*\)\(draw[[:blank:]]*([[:blank:]]*\)\([[:graph:]][[:graph:]]*\)\([[:blank:]]*)\)\([[:blank:]]*;\)/\1draw(\3,vparams);/g
#TriangleSetGeometryAlgorithms<DataTypes>::draw(); --> TriangleSetGeometryAlgorithms<DataTypes>::draw(vparams);
s/\([[:graph:]]*\)\(draw[[:blank:]]*([[:blank:]]*)[[:blank:]]*;\)/\1draw(vparams);/g

# getSimulation()->drawUtility().                   --> vparams->drawTool()->
# simulation::getSimulation()->drawUtility().       --> vparams->drawTool()->
# sofa::simulation::getSimulation()->drawUtility(). --> vparams->drawTool()->
s/[[:graph:]]*getSimulation()->DrawUtility()\./vparams->drawTool()->/g


