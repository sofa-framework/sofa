#Communication

## LibOscpack installation
Please ensure oscpack version is >= 1.1.X. Do not use the default packages provided by ubuntu repository (1.0.X version).
You can find a fully working version here : http://ftp.debian.org/debian/pool/main/o/oscpack/

```
sudo dpkg -i liboscpack-dev_1.1.0-2_amd64.deb
```

## How to use ServerCommunication OSC

```
<ServerCommunicationOSC name="ServerCommunicationOSC" job="receiver" port="6000" refreshRate="2"/>
<CommunicationSubscriber name="subscriberOSC" communication="@ServerCommunicationOSC" subject="/test" arguments="x y"/>
```
