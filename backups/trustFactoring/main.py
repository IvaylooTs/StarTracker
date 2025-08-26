

def IMUdata():
    return (1,2,3,4)



def CameraCall():
    return (1,0,0,0)


def cameraAvaiable():
    return True
def IMUAvailable():
    return True

def isTrackingCalibrated():
    if(cameraAvaiable()):
        return True
    else:
        return False


def isIMUCalibrated():
    if(IMUAvailable()):
        return True
    else:
        return False

def GetTrustFactor():
    imuCalibrated = isIMUCalibrated()
    cameraTracking = isTrackingCalibrated() << 1
    finalRes = 0
    finalRes |= cameraTracking | imuCalibrated

    return finalRes

if __name__ == "__main__":
    print("Starting system architecture")

    if cameraAvaiable():
        print("camera handled")

    elif IMUAvailable():
        print("IMU handled")

    print("Trust factor:",GetTrustFactor())