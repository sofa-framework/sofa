# SofaMatrix

SofaMatrix is a SOFA plugin containing components related to linear system matrices.

## Components

The plugin contains the following components that can be placed into a SOFA scene:

### GlobalSystemMatrixExporter

GlobalSystemMatrixExporter is a component to export the matrix of a linear system.
It supports the following file formats:
* **txt**: exports a Python-like list of lists, containing all the matrix values including the zeros.
It uses the `operator<<` in `BaseMatrix`.
* **csv**: exports a simple comma-separated values file, containing all the matrix values including the zeros.
* **jpg**: exports a binary image where white pixel means the value is zero, and black
pixel means the value is non-zero.
* **png**: exports a binary image where white pixel means the value is zero, and black
  pixel means the value is non-zero.
* **bmp**: exports a binary image where white pixel means the value is zero, and black
  pixel means the value is non-zero.

GlobalSystemMatrixExporter must have a link to a linear solver (a component inhereting from `sofa::core::behavior::LinearSolver`).

The link can be explicit. Example:
```xml
<GlobalSystemMatrixExporter linearSolver="@pathToLinearSolver"/>
```

The link can be implicit. Example:
```xml
<SparseLDLSolver/>
<GlobalSystemMatrixExporter/>
```

A use case is available in the examples folder.

### GlobalSystemMatrixImage

GlobalSystemMatrixImage is a component to visualize the matrix of a linear system in a
Qt widget.

GlobalSystemMatrixImage must have a link to a linear solver (a component inhereting from `sofa::core::behavior::LinearSolver`).

The link can be explicit. Example:
```xml
<GlobalSystemMatrixImage linearSolver="@pathToLinearSolver"/>
```

The link can be implicit. Example:
```xml
<SparseLDLSolver/>
<GlobalSystemMatrixImage/>
```

A use case is available in the examples folder.

#### Screenshot

![GlobalSystemMatrixImage](doc/GlobalSystemMatrixImage.png)
