#!/bin/sed -f 

# cath a block comprised between MeshLoader and the next Node. 

/.*MeshLoader/,/^[[:blank:]]*<[[:graph:]]*Node/ {
  
  :changeName
  /MeshLoader/ {
    /[[:blank:]]name/! {
      # MeshLoader has not got any name
      s/\/>[[:blank:]]*$/ name="loader"\/>/
    }
   s/\([[:blank:]]name[[:blank:]]*=\)\("[[:graph:]]*"\)/\1"loader"/
  }
  
  :changeLoader 
  /\.obj/{
    s/MeshLoader/MeshObjLoader/
  }
  /\.msh/{
    s/MeshLoader/MeshGmshLoader/
  }
  \/.vtk/{
    s/MeshLoader/MeshVTKLoader/
  }

  :changeMechanicalObject
  s/\(type[[:blank:]]*=[[:blank:]]*"MechanicalObject"\)/& src="@loader" /
  s/<[[:blank:]]*MechanicalObject[[:blank:]]*/<MechanicalObject src="@loader" /

  :changeTopology
  s/type[[:blank:]]*=[[:blank:]]*"Mesh"/type="Mesh" src="@loader" /
  s/<[[:blank:]]*Mesh[[:blank:]][[:blank:]]*/<Mesh src="@loader" /
  s/<[[:blank:]]*Mesh\/>/<Mesh src="@loader"\/>/
  #
  s/type[[:blank:]]*=[[:blank:]]*"MeshTopology"/type="MeshTopology" src="@loader"/
  s/<[[:blank:]]*MeshTopology[[:blank:]][[:blank:]]*/<MeshTopology src="@loader" /
  s/<[[:blank:]]*MeshTopology\/>/<MeshTopology src="@loader"\/>/
  #
  s/\(type[[:blank:]]*=[[:graph:]]*TopologyContainer"\)/& src="@loader"/
  s/\(<[[:graph:]]*TopologyContainer\)/& src="@loader"/
  #
  s/\(href[[:blank:]]*=[[:graph:]]*Topology.xml"\)/& src="@loader"/
}