#!/bin/bash
# use: ./changeTopologyNames.sh directoryPath

SOFA_DIR=$1

# VertexEdges => EdgesAroundVertex
# EdgeVertexShell => EdgesAroundVertex
# edgeVertexShell => edgesAroundVertex

grep -rl '\(VertexEdges\|EdgeVertexShell\|edgeVertexShell\)' ${SOFA_DIR} | grep -v svn >> listEdge.log

while read line
do
  echo ${line} | xargs sed -i 's/VertexEdges/EdgesAroundVertex/g'
  echo ${line} | xargs sed -i 's/EdgeVertexShell/EdgesAroundVertex/g'
  echo ${line} | xargs sed -i 's/edgeVertexShell/edgesAroundVertex/g'
done < "listEdge.log"




# TriangleVertexShell => TrianglesAroundVertex
# triangleVertexShell => trianglesAroundVertex
# TriangleEdgeShell => TrianglesAroundEdge
# triangleEdgeShell => trianglesAroundEdge

# TriangleEdges => EdgesInTriangle
# VertexTriangles => TrianglesAroundVertex
# EdgeTriangles => TrianglesAroundEdge

# TriangleEdge => EdgesInTriangle
# triangleEdge => edgesInTriangle

grep -rl '\(TriangleVertexShell\|triangleVertexShell\|TriangleEdgeShell\|triangleEdgeShell\|TriangleEdges\|VertexTriangles\|EdgeTriangles\|TriangleEdge\|triangleEdge\)' ${SOFA_DIR} | grep -v svn >> listTriangle.log

while read line
do
  echo ${line} | xargs sed -i 's/TriangleVertexShell/TrianglesAroundVertex/g'
  echo ${line} | xargs sed -i 's/triangleVertexShell/trianglesAroundVertex/g'
  echo ${line} | xargs sed -i 's/TriangleEdgeShell/TrianglesAroundEdge/g'
  echo ${line} | xargs sed -i 's/triangleEdgeShell/trianglesAroundEdge/g'

  echo ${line} | xargs sed -i 's/TriangleEdges/EdgesInTriangle/g'
  echo ${line} | xargs sed -i 's/VertexTriangles/TrianglesAroundVertex/g'
  echo ${line} | xargs sed -i 's/EdgeTriangles/TrianglesAroundEdge/g'

  echo ${line} | xargs sed -i 's/TriangleEdge/EdgesInTriangle/g'
  echo ${line} | xargs sed -i 's/triangleEdge/edgesInTriangle/g'

done < "listTriangle.log"







# QuadVertexShell => QuadsAroundVertex
# quadVertexShell => quadsAroundVertex
# QuadEdgeShell => QuadsAroundEdge
# quadEdgeShell => quadsAroundEdge

# QuadEdges => EdgesInQuad
# VertexQuads => QuadsAroundVertex
# EdgeQuads => QuadsAroundEdge

# QuadEdge => EdgesInQuad
# quadEdge => edgesInQuad

grep -rl '\(QuadVertexShell\|quadVertexShell\|QuadEdgeShell\|quadEdgeShell\|QuadEdges\|VertexQuads\|EdgeQuads\|QuadEdge\|quadEdge\)' ${SOFA_DIR} | grep -v svn >> listQuad.log


while read line
do
  echo ${line} | xargs sed -i 's/QuadVertexShell/QuadsAroundVertex/g'
  echo ${line} | xargs sed -i 's/quadVertexShell/quadsAroundVertex/g'
  echo ${line} | xargs sed -i 's/QuadEdgeShell/QuadsAroundEdge/g'
  echo ${line} | xargs sed -i 's/quadEdgeShell/quadsAroundEdge/g'

  echo ${line} | xargs sed -i 's/QuadEdges/EdgesInQuad/g'
  echo ${line} | xargs sed -i 's/VertexQuads/QuadsAroundVertex/g'
  echo ${line} | xargs sed -i 's/EdgeQuads/QuadsAroundEdge/g'

  echo ${line} | xargs sed -i 's/QuadEdge/EdgesInQuad/g'
  echo ${line} | xargs sed -i 's/quadEdge/edgesInQuad/g'

done < "listQuad.log"





# TetraVertexShell => TetrahedraAroundVertex
# tetraVertexShell => tetrahedraAroundVertex
# TetrahedronVertexShell => TetrahedraAroundVertex
# tetrahedronVertexShell => tetrahedraAroundVertex

# TetraEdgeShell => TetrahedraAroundEdge
# tetraEdgeShell => tetrahedraAroundEdge
# TetrahedronEdgeShell => TetrahedraAroundEdge
# tetrahedronEdgeShell => tetrahedraAroundEdge

# TetraTriangleShell => TetrahedraAroundTriangle
# tetraTriangleShell => tetrahedraAroundTriangle
# TetrahedronTriangleShell => TetrahedraAroundTriangle
# tetrahedronTriangleShell => tetrahedraAroundTriangle

# VertexTetras => TetrahedraAroundVertex
# EdgeTetras => TetrahedraAroundEdge
# TriangleTetras => TetrahedraAroundTriangle
# TetraEdges => EdgesInTetrahedron
# TetraTriangles => TrianglesInTetrahedron

# TetrahedronEdges => EdgesInTetrahedron
# TetrahedronEdge => EdgesInTetrahedron
# tetrahedronEdge => edgesInTetrahedron
# TetrahedronTriangles => TrianglesInTetrahedron
# TetrahedronTriangle => TrianglesInTetrahedron
# tetrahedronTriangle => trianglesInTetrahedron

# SeqTetras => SeqTetrahedra
# getTetras => getTetrahedra
# getNbTetras => getNbTetrahedra
# getTetra( => getTetrahedron(


grep -rl '\(tetras\|Tetras\|tetra\|Tetra\)' ${SOFA_DIR} | grep -v svn >> listTetra.log

while read line
do
  echo ${line} | xargs sed -i 's/TetraVertexShell/TetrahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/tetraVertexShell/tetrahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/TetrahedronVertexShell/TetrahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/tetrahedronVertexShell/tetrahedraAroundVertex/g'

  echo ${line} | xargs sed -i 's/TetraEdgeShell/TetrahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/tetraEdgeShell/tetrahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/TetrahedronEdgeShell/TetrahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/tetrahedronEdgeShell/tetrahedraAroundEdge/g'

  echo ${line} | xargs sed -i 's/TetraTriangleShell/TetrahedraAroundTriangle/g'
  echo ${line} | xargs sed -i 's/tetraTriangleShell/tetrahedraAroundTriangle/g'
  echo ${line} | xargs sed -i 's/TetrahedronTriangleShell/TetrahedraAroundTriangle/g'
  echo ${line} | xargs sed -i 's/tetrahedronTriangleShell/tetrahedraAroundTriangle/g'

  echo ${line} | xargs sed -i 's/VertexTetras/TetrahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/EdgeTetras/TetrahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/TriangleTetras/TetrahedraAroundTriangle/g'
  echo ${line} | xargs sed -i 's/TetraEdges/EdgesInTetrahedron/g'
  echo ${line} | xargs sed -i 's/TetraTriangles/TrianglesInTetrahedron/g'

  echo ${line} | xargs sed -i 's/TetrahedronEdges/EdgesInTetrahedron/g'
  echo ${line} | xargs sed -i 's/TetrahedronEdge/EdgesInTetrahedron/g'
  echo ${line} | xargs sed -i 's/tetrahedronEdge/edgesInTetrahedron/g'
  echo ${line} | xargs sed -i 's/TetrahedronTriangles/TrianglesInTetrahedron/g'
  echo ${line} | xargs sed -i 's/TetrahedronTriangle/TrianglesInTetrahedron/g'
  echo ${line} | xargs sed -i 's/tetrahedronTriangle/trianglesInTetrahedron/g'

  echo ${line} | xargs sed -i 's/SeqTetras/SeqTetrahedra/g'
  echo ${line} | xargs sed -i 's/getTetras/getTetrahedra/g'
  echo ${line} | xargs sed -i 's/getNbTetras/getNbTetrahedra/g'
  echo ${line} | xargs sed -i 's/getTetra(/getTetrahedron(/g'
  
done < "listTetra.log"







# HexaVertexShell => HexahedraAroundVertex
# hexaVertexShell => hexahedraAroundVertex
# HexahedronVertexShell=> HexahedraAroundVertex
# hexahedronVertexShell => hexahedraAroundVertex

# HexaEdgeShell => HexahedraAroundEdge
# hexaEdgeShell => hexahedraAroundEdge
# HexahedronEdgeShell => HexahedraAroundEdge
# hexahedronEdgeShell => hexahedraAroundEdge

# HexaQuadShell => HexahedraAroundQuad
# hexaQuadShell => hexahedraAroundQuad
# HexahedronQuadShell => HexahedraAroundQuad
# hexahedronQuadShell => hexahedraAroundQuad

# VertexHexas => HexahedraAroundVertex
# EdgeHexas => HexahedraAroundEdge
# QuadHexas => HexahedraAroundQuad
# HexaEdges => EdgesInHexahedron
# HexaQuads => QuadsInHexahedron

# HexahedronEdges => EdgesInHexahedron
# HexahedronEdge => EdgesInHexahedron
# hexahedronEdge => edgesInHexahedron
# HexahedronQuads => QuadsInHexahedron
# HexahedronQuad => QuadsInHexahedron
# hexahedronQuad => quadsInHexahedron

# SeqHexas => SeqHexahedra
# getHexas => getHexahedra
# getNbHexas => getNbHexahedra
# getHexa( => getHexahedron(

grep -rl '\(Hexas\|hexas\|Hexa\|hexa\)' ${SOFA_DIR} | grep -v svn >> listHexa.log

while read line
do
  echo ${line} | xargs sed -i 's/HexaVertexShell/HexahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/hexaVertexShell/hexahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/HexahedronVertexShell/HexahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/hexahedronVertexShell/hexahedraAroundVertex/g'

  echo ${line} | xargs sed -i 's/HexaEdgeShell/HexahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/hexaEdgeShell/hexahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/HexahedronEdgeShell/HexahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/hexahedronEdgeShell/hexahedraAroundEdge/g'

  echo ${line} | xargs sed -i 's/HexaQuadShell/HexahedraAroundQuad/g'
  echo ${line} | xargs sed -i 's/hexaQuadShell/hexahedraAroundQuad/g'
  echo ${line} | xargs sed -i 's/HexahedronQuadShell/HexahedraAroundQuad/g'
  echo ${line} | xargs sed -i 's/hexahedronQuadShell/hexahedraAroundQuad/g'

  echo ${line} | xargs sed -i 's/VertexHexas/HexahedraAroundVertex/g'
  echo ${line} | xargs sed -i 's/EdgeHexas/HexahedraAroundEdge/g'
  echo ${line} | xargs sed -i 's/QuadHexas/HexahedraAroundQuad/g'
  echo ${line} | xargs sed -i 's/HexaEdges/EdgesInHexahedron/g'
  echo ${line} | xargs sed -i 's/HexaQuads/QuadsInHexahedron/g'

  echo ${line} | xargs sed -i 's/HexahedronEdges/EdgesInHexahedron/g'
  echo ${line} | xargs sed -i 's/HexahedronEdge/EdgesInHexahedron/g'
  echo ${line} | xargs sed -i 's/hexahedronEdge/edgesInHexahedron/g'
  echo ${line} | xargs sed -i 's/HexahedronQuads/QuadsInHexahedron/g'
  echo ${line} | xargs sed -i 's/HexahedronQuad/QuadsInHexahedron/g'
  echo ${line} | xargs sed -i 's/hexahedronQuad/quadsInHexahedron/g'
  
  echo ${line} | xargs sed -i 's/SeqHexas/SeqHexahedra/g'
  echo ${line} | xargs sed -i 's/getHexas/getHexahedra/g'
  echo ${line} | xargs sed -i 's/getNbHexas/getNbHexahedra/g'
  echo ${line} | xargs sed -i 's/getHexa(/getHexahedron(/g'

done < "listHexa.log"



# getEdgeTriangleShell  getEdgesInTriangle
# getEdgeQuadShell   getEdgesInQuad
# getEdgeTetraShell  getEdgesInTetrahedron

# getEdgeHexaShell   getEdgesInHexahedron
# getTriangleTetraShell  getTrianglesInTetrahedron
# getQuadHexaShell    getQuadsInHexahedron
# getVertexVertexShell   getVerticesAroundVertex


grep -rl '\(getEdgeTriangleShell\|getEdgeQuadShell\|getEdgeTetraShell\|getEdgeHexaShell\|getTriangleTetraShell\|etQuadHexaShell\|getVertexVertexShell\)' ${SOFA_DIR} | grep -v svn >> listautres.log

while read line
do

  echo ${line} | xargs sed -i 's/getEdgeTriangleShell/getEdgesInTriangle/g'
  echo ${line} | xargs sed -i 's/getEdgeQuadShell/getEdgesInQuad/g'
  echo ${line} | xargs sed -i 's/getEdgeTetraShell/getEdgesInTetrahedron/g'

  echo ${line} | xargs sed -i 's/getEdgeHexaShell/getEdgesInHexahedron/g'
  echo ${line} | xargs sed -i 's/getTriangleTetraShell/getTrianglesInTetrahedron/g'
  echo ${line} | xargs sed -i 's/getQuadHexaShell/getQuadsInHexahedron/g'
  echo ${line} | xargs sed -i 's/getVertexVertexShell/getVerticesAroundVertex/g'

  
done < "listautres.log"
# 
# cd ${SOFA_DIR}
# maake
