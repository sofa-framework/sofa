HeadlessRecorder is, as its name suggest, a new gui plugin who allow users to records sofa visualisation without any windows (batch only)

⚠ Linux only ⚠

## Installation

Ubuntu :
```
$ sudo apt-get install libavcodec-dev libswscale-dev libavutil-dev libavformat-dev 
```

## How to use it

Every needed information are available in runSofa helper.
```
$ ./runSofa -h
```

Here is an example for recording a 5 seconds video in 1920x1080 
```
$ ./runSofa -g hRecorder --video --width=1920 --height=1080 --fps=60 --recordTime=5 -a --filename aFileName
```
This example will record in a video file named myFileName a footage of the default runSofa scene (aka caduceus). The dimensions of the video will be 1920x1080, the framerate is set to 60fps, the recording time will be 10 seconds and the option -a animate the scene.

Here is an example for screenshots 1920x1080 
```
$ ./runSofa -g hRecorder --picture --width=1920 --height=1080 --fps=60 --recordTime=10 -a --filename aFileName
```
## Information

You have to use an InteractiveCamera component in your scene and correctly place it before recording.
By example you need to add this line to your caduceus scene :
```
    <InteractiveCamera position="0 30 90" lookAt="0 30 0"/>
```

### Authors
Douaille Erwan

### Contact information
douailleerwan@gmail.com
