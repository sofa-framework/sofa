function Component()
{
    // default constructor
    
    installer.finishButtonClicked.connect(this, Component.prototype.installationFinished);
}

Component.prototype.createOperations = function()
{
    component.createOperations();

    if (systemInfo.productType === "windows") {
        component.addOperation("CreateShortcut", "@TargetDir@/bin/runSofa.exe", "@StartMenuDir@/runSofa.lnk",
            "workingDirectory=@TargetDir@", "iconPath=@TargetDir@/bin/runSofa.exe");
            
        component.addOperation("CreateShortcut", "@TargetDir@/maintenancetool.exe", "@StartMenuDir@/Uninstall.lnk",
            "workingDirectory=@TargetDir@", "iconPath=%SystemRoot%/system32/xwizard.exe");
    }
}

Component.prototype.installationFinished = function()
{
    if (installer.isInstaller() && installer.status == QInstaller.Success) {
        QDesktopServices.openUrl("https://www.sofa-framework.org/thank-you?sofa=" + installer.value("Version") + "&os=" + systemInfo.productType);
    }
}
