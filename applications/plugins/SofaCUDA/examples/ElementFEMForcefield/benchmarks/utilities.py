import numpy as np

def generate_regular_grid(nx=10, ny=10, nz=10, min_corner=(0, 0, 0), max_corner=(1, 1, 1)):                                                                                                                       
    """                                                                                                                                                                                                  
    Generate a regular grid of hexahedra.                                                                                                                                                                
                                                                                                                                                                                                       
    Args:                                                                                                                                                                                                
      nx, ny, nz: Number of vertices in each direction (grid resolution)                                                                                                                               
      min_corner: (xmin, ymin, zmin) tuple                                                                                                                                                             
      max_corner: (xmax, ymax, zmax) tuple                                                                                                                                                             
                                                                                                                                                                                                       
    Returns:                                                                                                                                                                                             
      points: array of shape (nx*ny*nz, 3) - vertex positions                                                                                                                                          
      hexahedra: array of shape ((nx-1)*(ny-1)*(nz-1), 8) - hexahedra indices                                                                                                                          
    """
    xmin, ymin, zmin = min_corner                                                                                                                                                                        
    xmax, ymax, zmax = max_corner                                                                                                                                                                        
                                                                                                                                                                                                       
    # Compute spacing                                                                                                                                                                                    
    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 0                                                                                                                                                       
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 0                                                                                                                                                       
    dz = (zmax - zmin) / (nz - 1) if nz > 1 else 0                                                                                                                                                       
                                                                                                                                                                                                       
    # Generate points                                                                                                                                                                                    
    points = []                                                                                                                                                                                          
    for k in range(nz):                                                                                                                                                                                  
      for j in range(ny):                                                                                                                                                                              
          for i in range(nx):                                                                                                                                                                          
              points.append([xmin + i*dx, ymin + j*dy, zmin + k*dz])                                                                                                                                   
    points = np.array(points)                                                                                                                                                                            
                                                                                                                                                                                                       
    # Helper to get point index from grid coordinates                                                                                                                                                    
    def point_index(i, j, k):                                                                                                                                                                            
      return nx * (ny * k + j) + i                                                                                                                                                                     
                                                                                                                                                                                                       
    # Generate hexahedra (8 vertices per hexa, in SOFA convention)                                                                                                                                       
    hexahedra = []                                                                                                                                                                                       
    for k in range(nz - 1):                                                                                                                                                                              
      for j in range(ny - 1):                                                                                                                                                                          
          for i in range(nx - 1):                                                                                                                                                                      
              hexa = [                                                                                                                                                                                 
                  point_index(i,   j,   k),                                                                                                                                                            
                  point_index(i+1, j,   k),                                                                                                                                                            
                  point_index(i+1, j+1, k),                                                                                                                                                            
                  point_index(i,   j+1, k),                                                                                                                                                            
                  point_index(i,   j,   k+1),                                                                                                                                                          
                  point_index(i+1, j,   k+1),                                                                                                                                                          
                  point_index(i+1, j+1, k+1),                                                                                                                                                          
                  point_index(i,   j+1, k+1),                                                                                                                                                          
              ]                                                                                                                                                                                        
              hexahedra.append(hexa)                                                                                                                                                                   
    hexahedra = np.array(hexahedra)                                                                                                                                                                      
                                                                                                                                                                                                       
    return points, hexahedra

def hexa_to_tetra(hexahedra):                                                                                                                                                                            
    """                                                                                                                                                                                                  
    Convert hexahedra to tetrahedra.                                                                                                                                                                     
                                                                                                                                                                                                       
    Each hexahedron is split into 5 tetrahedra.                                                                                                                                                          
                                                                                                                                                                                                       
    Args:       
      hexahedra: array of shape (N, 8) - hexahedra vertex indices

    Returns:
      tetrahedra: array of shape (N*5, 4) - tetrahedra vertex indices
    """
    tetrahedra = []

    # 5-tetra decomposition using diagonal 1-3-4-6
    splits = [
      [0, 1, 3, 4],
      [1, 2, 3, 6],
      [1, 4, 5, 6],
      [3, 4, 6, 7],
      [1, 3, 4, 6],  # central tetrahedron                                                                                                                                                             
    ]
                                                                                                                                                                                                       
    for hexa in hexahedra:
      for split in splits:                                                                                                                                                                             
          tetrahedra.append([hexa[i] for i in split])
                                                                                                                                                                                                       
    return np.array(tetrahedra)

def hexa_to_tetra_symmetric(hexahedra):                                                                                                                                                                  
    """                                                                                                                                                                                                  
    Convert hexahedra to tetrahedra using symmetric 6-tetra decomposition.                                                                                                                               

    Each hexahedron is split into 6 tetrahedra around the space diagonal (0-6).                                                                                                                          
    Better symmetry properties for FEM simulations.

    Args:
      hexahedra: array of shape (N, 8) - hexahedra vertex indices

    Returns:
      tetrahedra: array of shape (N*6, 4) - tetrahedra vertex indices
    """
    tetrahedra = []

    # 6-tetra symmetric decomposition around diagonal 0-6
    splits = [
      [0, 1, 2, 6],
      [0, 2, 3, 6],
      [0, 3, 7, 6],
      [0, 7, 4, 6],
      [0, 4, 5, 6],
      [0, 5, 1, 6],
    ]

    for hexa in hexahedra:
      for split in splits:
          tetrahedra.append([hexa[i] for i in split])

    return np.array(tetrahedra)