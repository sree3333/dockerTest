import io as ioo, numpy as np, json
import cv2, cfg, os
import urllib.request as urllib
import boto3

def convertToBytes(imagefront,imageside):
    imagefront = ioo.BytesIO(imagefront)
    imagefront = np.fromstring(imagefront.getvalue(), dtype=np.uint8)
    imagefront = cv2.imdecode(imagefront, 1)


    imageside = ioo.BytesIO(imageside)
    imageside = np.fromstring(imageside.getvalue(), dtype=np.uint8)
    imageside = cv2.imdecode(imageside, 1)


    return imagefront,imageside

def gettoppoint( img):
    cnts,hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    return extTop[1]

def gets3images(imagefrontpath, imagesidepath):
    s3 = boto3.client('s3')
    spliturl = imagefrontpath.split(cfg.s3bucket + '/')
    url = s3.generate_presigned_url('get_object', Params={'Bucket': 'mirrorsizeandroid-userfiles-mobilehub-1901898188','Key': spliturl[1], }, ExpiresIn=3600)
    resp = urllib.urlopen(url)
    image = np.fromstring(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    front = image
    # front = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    spliturl = imagesidepath.split(cfg.s3bucket + '/')
    url = s3.generate_presigned_url('get_object', Params={'Bucket': 'mirrorsizeandroid-userfiles-mobilehub-1901898188','Key': spliturl[1], },ExpiresIn=3600)
    resp = urllib.urlopen(url)
    image = np.fromstring(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    side = image
    # side = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return front, side

def gather(userid,status):
    if cfg.s3upload == 'True' and status == True:
        os.system('aws s3 cp '+cfg.basedir+'/output/'+str(userid)+'/corrected s3://mirrorsize/trainingdata/'+str(userid)+' --recursive')
    elif cfg.s3upload == 'True' and status == False:
        os.system('aws s3 cp '+cfg.basedir+'/output/'+str(userid)+'/corrected s3://mirrorsize/trainingdata/error'+str(userid)+' --recursive')

def getPersonHeight(img):
    white_pixels_front = np.where(img > 50)
    y = white_pixels_front[0]
    ya = max(y)
    yb = min(y)
    heightinpixel = ya - yb
    return heightinpixel

def getScaleFactor(userheightmm,userheightpixel):
    scalfactor = float(userheightmm)/float(userheightpixel)
    return scalfactor

def angle(pt1,pt2):
    return np.rad2deg(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

def processErrorCode(measurements,dpdata):
    userid = dpdata['userid']
    with open(cfg.errorjson) as f:
        errordic= json.load(f)
    if dpdata['statusmin'] == False:
        # No one found in the frame
        measurements['message'] = errordic['103']
        measurements['status'] = '103'
        gather(userid,False)

    elif dpdata['statusmax'] == False:
        # Check you are alone in frame
        measurements['message'] = errordic['104']
        measurements['status'] = '104'
        gather(userid,False)

    elif dpdata['statusfullbody'] == False:
        # Segmentation error
        # 1. Check your lighting
        # 2. something is occuluding Body
        measurements['message'] = errordic['105']
        measurements['status'] = '105'
        gather(userid,False)

    elif dpdata['statusfront'] == False:
        # Front Pose Incorrect
        if dpdata['statusArm'] == False and dpdata['statusLeg'] == False:
            measurements['message'] = errordic["ArmLeg"]
        elif dpdata['statusArm'] == False:
            measurements['message'] = errordic["Arm"]
        elif dpdata['statusLeg'] == False:
            measurements['message'] = errordic["Leg"]
        measurements['status'] = '101'
        gather(userid,False)

    elif dpdata['statusside'] == False:
        # Side Pose Incorrect
        if dpdata['statusSideAngle'] == False and dpdata['statusSideOcclude'] == False:
            measurements['message'] = errordic["SideAngleOcclude"]
        elif dpdata['statusSideAngle'] == False:
            measurements['message'] = errordic["SideAngle"]
        elif dpdata['statusSideOcclude'] == False:
            measurements['message'] = errordic["SideOcclude"]
        measurements['status'] = '102'
        gather(userid,False)
    return measurements

def getintialdict():
    measurements = {}
    measurements['response'] = 'Vision Error'
    measurements['status'] = '-1'
    measurements['message'] = "The system couldn't process your request due to multiple noise in the scene. Please try again once more."
    return measurements

def getMeasurementdict(gender,measurements):
    measure ={}
    if gender == 'male':
        measure['chest']                   = -1
        measure['stomach']                 = -1
        measure['waist']                   = -1
        measure['thigh']                   = -1
        measure['calfMuscle']              = -1
        measure['kneeGirth']               = -1
        measure['neck']                    = -1
        measure['hip']                     = -1
        measure['upperwaist']              = -1
        measure['shoulderAcross']          = -1
        measure['frontChestWidth']         = -1
        measure['urise']                   = -1
        measure['backChestWidth']          = -1
        measure['legLength']               = -1
        measure['inseam']                  = -1
        measure['frontTorsoLength']        = -1
        measure['armsLength']              = -1
        measure['rise']                    = -1
        measure['backTorsoLength']         = -1
        measure['waistToKneeLength']       = -1
        measure['waistDepth']              = -1
        measure['upperArmLength']          = -1
        measure['foreArmLength']           = -1
        measure['biceps']                  = -1
        measure['jacketLength']            = -1
        measure['shirtLength']             = -1
        measure['kneetoAnkleLength']       = -1
        measure['sleeveLength']            = -1
        measure['sleeveLengthFull']        = -1
        measure['frontKneeLength']         = -1
        measure['frontWaistLength']        = -1
        measure['centerBackLength']        = -1
        measure['centerFrontLength']       = -1
        measure['cervicalLength']          = -1
        measure['sideseam']                = -1
        measure['armhole']                 = -1
        measure['armholeLevel']            = -1
        measure['naturalWaistLength']      = -1
        measure['naturalWaistGirth']       = -1
        measure['halfSleeveLength']        = -1
        measure['chestDepth']              = -1
        measure['stomachDepth']            = -1
        measure['hipDepth']                = -1
        measure['armInseam']               = -1
        measure['halfBackChestWidth']      = -1
        measure['upperNeck']               = -1
        measure['chestDepth']              = -1
        measure['frontChestWidth']         = -1
        measure['strappedLinear']          = -1
        measure['strappedSlant']           = -1
        measure['scyeDepth']               = -1
        measure['shoulderSlope']           = -1
        measure['hipHeight']               = -1
        measure['waistHeight']             = -1
        measure['stomachHeight']           = -1
        measure['chestHeight']             = -1
        measure['thighSlant']              = -1
        measure['wrist']                   = -1
        measure['vestFront']               = -1
        measure['vestBack']                = -1
        measure['upperHip']                = -1
        measure['shoulderBust']            = -1
        measure['shoulderWaist']           = -1
        measure['upperHip']                = -1


    elif gender == 'female':
        measure['chest']                   = -1
        measure['stomach']                 = -1
        measure['waist']                   = -1
        measure['thigh']                   = -1
        measure['calfMuscle']              = -1
        measure['kneeGirth']               = -1
        measure['overBust']                = -1
        measure['underBust']               = -1
        measure['neck']                    = -1
        measure['hip']                     = -1
        measure['upperwaist']              = -1
        measure['shoulderAcross']          = -1
        measure['frontChestWidth']         = -1
        measure['urise']                   = -1
        measure['backChestWidth']          = -1
        measure['legLength']               = -1
        measure['inseam']                  = -1
        measure['frontTorsoLength']        = -1
        measure['armsLength']              = -1
        measure['rise']                    = -1
        measure['backTorsoLength']         = -1
        measure['waistToKneeLength']       = -1
        measure['waistDepth']              = -1
        measure['upperArmLength']          = -1
        measure['foreArmLength']           = -1
        measure['biceps']                  = -1
        measure['jacketLength']            = -1
        measure['shirtLength']             = -1
        measure['kneetoAnkleLength']       = -1
        measure['sleeveLength']            = -1
        measure['sleeveLengthFull']        = -1
        measure['frontKneeLength']         = -1
        measure['frontWaistLength']        = -1
        measure['centerBackLength']        = -1
        measure['centerFrontLength']       = -1
        measure['cervicalLength']          = -1
        measure['sideseam']                = -1
        measure['armhole']                 = -1
        measure['armholeLevel']            = -1
        measure['naturalWaistLength']      = -1
        measure['naturalWaistGirth']       = -1
        measure['halfSleeveLength']        = -1
        measure['chestDepth']              = -1
        measure['stomachDepth']            = -1
        measure['hipDepth']                = -1
        measure['armInseam']               = -1
        measure['halfBackChestWidth']      = -1
        measure['upperNeck']               = -1
        measure['chestDepth']              = -1
        measure['frontChestWidth']         = -1
        measure['strappedLinear']          = -1
        measure['strappedSlant']           = -1
        measure['scyeDepth']               = -1
        measure['shoulderSlope']           = -1
        measure['hipHeight']               = -1
        measure['waistHeight']             = -1
        measure['stomachHeight']           = -1
        measure['chestHeight']             = -1
        measure['thighSlant']              = -1
        measure['wrist']                   = -1
        measure['vestFront']               = -1
        measure['vestBack']                = -1
        measure['upperHip']                = -1
        measure['shoulderBust']            = -1
        measure['shoulderWaist']           = -1
        measure['upperHip']                = -1

    measurements['measurements'] = measure
    return measurements

def getmodifiedkeypoints(keypoints,pose):
    if pose == 'f':
        points = []
        for pt in keypoints:
            points.append([pt[0]+30,pt[1]+30])
        return points
    if pose == 's':
        points = []
        for pt in keypoints:
            points.append([pt[0] + 30, pt[1] + 30])
        return points
