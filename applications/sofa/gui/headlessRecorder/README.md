#Headless Recorder

## Installation

Ubuntu :
```
$ sudo apt-get install libavcodec-dev libswscale-dev libavutil-dev 
```

## How to use it

Every needed information are available in runSofa helper.
```
$ ./runSofa -h
```

Here is an example for recording a video in 1920x1080 
```
$ ./runSofa -g hRecorder --recordAsVideo myFileName --width 1920 --height 1080
```

## Troubleshooting

### Missing GL version
This error may be related to the nvidia driver installed on your linux OS.
The 384.90 nvidia driver has a bug with EGL (library used by this plugin)
Bug report: https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-384/+bug/1731968
Please ensure your version is different:
```
$ dpkg -l | grep nvidia
```

Here is a link to the wiki of ubuntu in case you need to downgrade/upgrade your nvidia driver : https://help.ubuntu.com/community/BinaryDriverHowto/Nvidia

## Information

### Authors
Douaille Erwan

### Contact information
douailleerwan@gmail.com
