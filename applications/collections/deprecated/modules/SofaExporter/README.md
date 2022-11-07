= SofaExporter

This plugin provide several exporters to save simulation. There is several classes of exporters.
Some of them are saving object surfaces while other are saving complete topologies.

Have a look at the examples provided in:
```
examples/MeshExporter.scn
examples/OBJExporter.scn
examples/STLExporter.scn
```

For these three exporters. The filename property can have the following pattern:
```
/absolute/path/file
./relative/path/file
./relative/path/nofilename/ (use the object name as filename)
```

