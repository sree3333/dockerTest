import os
import glob

basedir = os.getcwd()
body_measure_path = basedir + '/data/3D_Body/'
output_base_dir = basedir + '/output'
modeldir = basedir+'/modelspose'
tightfit_indices = basedir+ "/data/indice_files.npy"
loosefit_indices = basedir+ "/data/indice_files_loosefit.npy"
archetype_indices = basedir + "/data/archetype_indice_file.npy"
input_body_indice = basedir +"/data/3d_body_indice.npy"
loosefit_indices_bmi = basedir+ "/data/bmi_based_data/archetype_indice_file.npy"
debug = 'False'
export = 'False'
s3upload = 'False'

s3bucket = "mirrorsizeandroid-userfiles-mobilehub-1901898188"
errorjson = basedir+'/data/errormsg'
merchantspresentid=[]
merchantspresent = glob.glob(basedir + '/data/merchants/*')
for i in merchantspresent:
    merchantspresentid.append(os.path.basename(i))


# yolact_weight       = modeldir+'/yolact/yolact_darknet_2_231_80000.pth'
# yolact_weight       = modeldir+'/yolact/yolact_darknet_2_399_70000.pth'
yolact_weight       = modeldir+'/yolact/yolact_darknet_2_333_80000.pth'
ssdinceptioncoco    = modeldir+'/detection/ssd_inception_v2.pb'
frcnninceptioncoco  = modeldir+'/detection_2/frcnn_inception_v2.pb'
outputdir           = basedir+'/data_runtime'
face_path           = modeldir+'/hmr/smpl_faces.npy'
sconfig             = modeldir+'/detection_2/faster_rcnn_inception_v2_coco.config'
classes             = modeldir+'/detection_2/object_detection_classes_coco.txt'
maskrcnnpath        = modeldir+'/maskrcnn/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml'
merchantsjson       = basedir+'/data/merchants/merchants.json'


# Indices for circumferential measurement
archetypr_neck      =  basedir+'/data/archetype_neck.txt'
chest_indice        =  '/circumferential/chest.txt'
stomach_indice      =  '/circumferential/stomach.txt'
underchest_indice   =  '/circumferential/underchest.txt'
waist_indice        =  '/circumferential/s4.txt'
circum_indice       =  '/circum.txt'
hip                 =  '/circumferential/hip.txt'
upper_arm_circum    =  '/circumferential/upper_arm_circum.txt'
confirm_bust        =  '/circumferential/confirm_bust.txt'
thigh               =  '/circumferential/thigh_v2.txt'
calf                =  '/circumferential/calf.txt'
u8                  =  '/circumferential/u8.txt'
neck                =  '/circumferential/neck_v3.txt'
knee                =  '/circumferential/knee.txt'
s0                  =  '/circumferential/s0.txt'
s3                  =  '/circumferential/s3.txt'
nw1                 =  '/circumferential/nw1.txt'
lower_body_indice = '/lower_body.txt'
upper_body_indice = '/upper_body.txt'
arm_indice = '/arms_index.txt'

front_deform ={
    'front_torso_right':{'region':'front_right_region_torso_index'
    ,'fixed_edge':'front_midline_torso_index','old_handle':'front_right_boundary_torso_index',
    'new_handle':'right'},
               'front_torso_left':{'region':'front_left_region_torso_index'
    ,'fixed_edge':'front_midline_torso_index','old_handle':'front_left_boundary_torso_index',
    'new_handle':'left'},
               'back_torso_left':{'region':'rt_back_index '
    ,'fixed_edge':'back_midline_index','old_handle':'rt_back_bound_index ',
    'new_handle':'right'},
               'back_torso_right':{'region':'lft_back_index'
    ,'fixed_edge':'back_midline_index','old_handle':'lft_back_bound_index',
    'new_handle':'left'}}

side_deform ={
        'side_torso_front_right':{'region':'side_torso_right_front_region_index'
    ,'fixed_edge':'side_torso_right_fixed_line_index','old_handle':'side_front_right_old_handle_torso_index',
    'new_handle':'front'},
        'side_torso_front_left':{'region':'side_torso_left_front_region_index'
    ,'fixed_edge':'side_torso_left_fixed_line_index','old_handle':'side_front_left_old_handle_torso_index',
    'new_handle':'front'},
        'side_torso_back_right':{'region':'side_torso_right_back_region_index'
    ,'fixed_edge':'side_torso_right_fixed_line_index','old_handle':'side_torso_right_back_old_handle_index',
    'new_handle':'back'},
        'side_torso_back_left':{'region':'side_torso_left_back_region_index'
    ,'fixed_edge':'side_torso_left_fixed_line_index','old_handle':'side_torso_left_back_old_handle_index',
    'new_handle':'back'}
              }

leg_front_deform = {
    'leg_right':{'region':'region2'
    ,'fixed_edge':'right_leg_midline_index','old_handle':'right_leg_right_boundary_index',
    'new_handle':'right'},
    'leg_left':{'region':'region'
    ,'fixed_edge':'right_leg_midline_index','old_handle':'right_leg_left_boundary_index',
    'new_handle':'left'}
}

leg_side_deform = {'side_leg_front_right':{'region':'right_leg_right_region_index'
    ,'fixed_edge':'right_leg_right_boundary_index','old_handle':'right_leg_midline_index',
    'new_handle':'front'},
        'side_leg_front_left':{'region':'right_leg_left_region_index'
    ,'fixed_edge':'right_leg_left_boundary_index','old_handle':'right_leg_midline_index',
    'new_handle':'front'},
        'side_leg_back_right':{'region':'right_leg_right_back_region_index'
    ,'fixed_edge':'right_leg_right_boundary_index','old_handle':'right_leg_back_midline_index',
    'new_handle':'back'},
        'side_leg_back_left':{'region':'right_leg_left_back_region_index'
    ,'fixed_edge':'right_leg_left_boundary_index','old_handle':'right_leg_back_midline_index',
    'new_handle':'back'}}

neck_front_deform = {
    'neck_right':{'region':'neck_right'
    ,'fixed_edge':'neck_midline','old_handle':'neck_right_boundary',
    'new_handle':'right'},
    'neck_left':{'region':'neck_left'
    ,'fixed_edge':'neck_midline','old_handle':'neck_left_boundary',
    'new_handle':'left'}
}



frontrefpoints  = [[705, 260], [722, 512], [546, 513], [395, 731], [244, 916], [899, 512],
                  [1058, 731], [1210, 924], [605, 1042], [563, 1453], [512, 1814], [831, 1041],
                  [865, 1445], [924, 1822], [663, 235], [739, 235], [622, 286], [798, 277]]

siderefpoints   = [[110, 238], [269, 418], [268, 418], [0, 0], [0, 0], [279, 427], [278, 716],
                 [199, 985], [219, 994], [249, 1442], [258, 1840], [229, 1004], [259, 1452],
                 [268, 1870], [0, 0], [139, 199], [0, 0], [249, 209]]

front_thresh_tightfit = {'f_euclDis_tresh_torso':0.062,'f_rotation_tresh_torso':24.0,'f_euclDis_tresh_legs': 0.05,
                         'f_rotation_tresh_legs':18.0,'f_schouder_tresh':0.062}

front_thresh_loosefit = {'f_euclDis_tresh_torso':0.07,'f_rotation_tresh_torso':24.0,'f_euclDis_tresh_legs': 0.05,
                         'f_rotation_tresh_legs':18.0,'f_schouder_tresh':0.07}


natural_waist_girth = {'nw2':[4961,1489],'nw3':[4962,1487],'nw4':[4804,1323],'nw5':[4120,632]}
hip_dict      = {'h2': 3119, 'h3': 3141, 'h4': 3484}
upper_waist   = {'s1':[4119,631],'s2':[4165,676],'s3':[4317,830]}
waist_item    = {'s4':[1783],'s5':[1784],'w1':[3021]}
over_bust     = {'u1': [1329, 3017], 'u2': [1330, 3016], 'u3': [3079, 3505],
                 'u4': [3077, 3015], 'u5': [3076, 3014],'u6': [3506, 3508], 'u7': [3498, 3027]}
chest         = {"u7":3498,"u6":3506,"u5":3076,"u4":3077}
store_3dbody_measure = {"biceps":"bicepsIndices","wrist":"wristIndices","calfMuscle":"calfIndices","thigh":"thighIndices",
                        "kneeGirth":"kneeIndices","thighSlant":"slantThighIndices",
                        "upperKneeGirth":"upperkneeindices","foreArmGirth":"foreArmGirth","ankleGirth":"anklegirth","midThighGirth":"midthighgirth"}

store_3dbody_measure_loosefit ={"upperHip":"upperHipIndices","hip":"hipIndices","biceps":"bicepsIndices","wrist":"wristIndices",
                                "elbowGirth":"elbowgirth"}

store_circum_loosefit   = {"chest": "chest","stomach":"stomach_loose","waist":"s4","thigh":"thigh_v2","calfMuscle":"calf","kneeGirth":"knee"}

linear_prior            = {'armsLength':'shouldertowrist','upperArmLength':'upper_arm','foreArmLength':'lower_arm','sleeveLength':'sleevelength',
                        'armhole':'armhole','sideseam':'sideseam','kneetoAnkleLength':'kneetoankle','waistToKneeLength':'waisttoknee',
                            'halfSleeveLength':'halfsleeve','armInseam':'arminseam','inseam':'inseam'}

linear_prior_loosefit           = {'armsLength':'shouldertowrist','upperArmLength':'upper_arm','foreArmLength':'lower_arm','sleeveLength':'sleevelength',
                        'armhole':'armhole','sideseam':'sideseam','kneetoAnkleLength':'kneetoankle','waistToKneeLength':'waisttoknee',
                            'halfSleeveLength':'halfsleeve','armInseam':'arminseam','inseam':'inseam','legLength':'hiptoankle'}



linear_post             = {'rise':'rise','backTorsoLength':'backtorso',
                            'centerBackLength':'centerback',
                            'centerFrontLength':'centerfront','armholeLevel':'armhole_level','naturalWaistLength':'natural_waist','yoke':'yoke'
                           ,'vestBack':'vestback','vestFront':'vestfront','strappedLinear':'strappedlinear','strappedSlant':'strappedslant','scyeDepth':'scyedepth',
                           'shoulderNaturalwaist':'shouldernaturalwaist','naturalwaistWaist':'naturalwaistWaist','cervicalNaturalwaist':'cervicalNaturalwaist',
                          'naturalwaistWaistBack':'naturalwaistWaistBack'}
linear_post_female            = {'rise':'rise','backTorsoLength':'backtorso',
                            'centerBackLength':'centerback',
                            'centerFrontLength':'centerfront','armholeLevel':'armhole_level','naturalWaistLength':'natural_waist','yoke':'yoke'
                           ,'vestBack':'vestback','vestFront':'vestfront','strappedLinear':'strappedlinear','strappedSlant':'strappedslant','scyeDepth':'scyedepth',
                           'shoulderNaturalwaist':'shouldernaturalwaist','naturalwaistWaist':'naturalwaistWaist','cervicalNaturalwaist':'cervicalNaturalwaist',
                          'naturalwaistWaistBack':'naturalwaistWaistBack','shoulderWaist':'shoulderwaist','shoulderBust':'shoulderbust','bustToBust':'busttobust','bustToNaturalwaist':'busttonaturalwaist',
               'neckToBust':'necktobust','neckToMidwaist':'necktomidwaist' }


linear_post_loose             = {'rise':'rise','frontTorsoLength':'fronttorso','backTorsoLength':'backtorso','frontKneeLength':'kneelength',
                            'frontWaistLength':'frontwaistlength','centerBackLength':'centerback',
                            'centerFrontLength':'centerfront','armholeLevel':'armhole_level','naturalWaistLength':'natural_waist','yoke':'yoke'
                           ,'vestBack':'vestback','vestFront':'vestfront','strappedLinear':'strappedlinear','strappedSlant':'strappedslant','scyeDepth':'scyedepth',
                           'shoulderNaturalwaist':'shouldernaturalwaist','naturalwaistWaist':'naturalwaistWaist','cervicalNaturalwaist':'cervicalNaturalwaist',
                          'naturalwaistWaistBack':'naturalwaistWaistBack','jacketLength':'jacketlength'}

waist_detection ={"midwaist":[2915,6375],"lowwaist":[1793,5256]}

linear_post_depth       = {'waistDepth':'waistdepth','chestDepth':'chestdepth','stomachDepth':'stomachdepth',
                           'hipDepth':'hipdepth'}

curve_dict_male             = {"backChestWidth":"backchestwidth","frontChestWidth":"frontchestwidth","urise":"urise",
                           "halfBackChestWidth":"halfback","uriseLowHigh":"uriselowhigh","uriseHighLow":"urisehighlow"}

curve_dict_female              = {"backChestWidth":"backchestwidth","frontChestWidth":"frontchestwidth","urise":"urise",
                           "halfBackChestWidth":"halfback","uriseLowHigh":"uriselowhigh","uriseHighLow":"urisehighlow","backRise":"backrise",
                           "naturalwaistWidth":"naturalwaistwidth","midwaistWidth":"midwaistwidth","backShoulderWidth":"acrossbackwidth"}


curve_dict_loosefit              = {"backChestWidth":"backchestwidth","frontChestWidth":"frontchestwidth","urise":"urise",
                           "halfBackChestWidth":"halfback"}


store_circum_tightfit   = {"stomach":"stomach","neck":"neck_v3","upperNeck":"neck_v3","upperwaist":"stomach","upperWaist":"stomach",
                           "upperHip":"upperhip" }

curve_list_tightfit_male             = ['shoulderAcross','frontChestWidth','urise','backChestWidth','halfBackChestWidth',
                                    'shoulderSlope','frontSoulderAcross','uriseLowHigh','uriseHighLow']
curve_list_tightfit_female              = ['shoulderAcross','frontChestWidth','urise','backChestWidth','halfBackChestWidth',
                                    'shoulderSlope','frontSoulderAcross','uriseLowHigh','uriseHighLow',"naturalwaistWidth","midwaistWidth",
                                    "backShoulderWidth","backRise"]

curve_list_loosefit              = ['shoulderAcross','frontChestWidth','urise','backChestWidth','halfBackChestWidth','frontSoulderAcross']

linear_list_tightfit             = ['legLength','inseam','frontTorsoLength','armsLength','rise','backTorsoLength',
                           'waistToKneeLength','waistDepth','upperArmLength','foreArmLength','jacketLength','shirtLength','kneetoAnkleLength',
                            'sleeveLength','frontKneeLength','frontWaistLength','centerBackLength','centerFrontLength','cervicalLength',
                            'sideseam','armholeLevel','naturalWaistLength','halfSleeveLength','chestDepth',
                            'stomachDepth','hipDepth','armInseam','sleeveLengthFull','vestBack','vestFront','strappedLinear','strappedSlant','scyeDepth','chestHeight'
                                    ,'waistHeight','hipHeight','stomachHeight','height','weight','shoulderNaturalwaist','naturalwaistWaist',
                                    'cervicalNaturalwaist','naturalwaistWaistBack','shoulderSeam','legLengthLowWaist','legLengthHighWaist'
                                    ,'jacketLengthShort','jacketLengthLong','midwaistHeight','midwaistToHipHeight']

linear_list_female_tightfit      = ['naturalwaistToHip','bustToNaturalwaist','neckToBust','neckToMidwaist','legLength','inseam','frontTorsoLength','armsLength','rise','backTorsoLength',
                           'waistToKneeLength','waistDepth','upperArmLength','foreArmLength','jacketLength','shirtLength','kneetoAnkleLength',
                            'sleeveLength','frontKneeLength','frontWaistLength','centerBackLength','centerFrontLength','cervicalLength',
                            'sideseam','armholeLevel','naturalWaistLength','halfSleeveLength','chestDepth',
                            'stomachDepth','hipDepth','armInseam','sleeveLengthFull','vestBack','vestFront','shoulderWaist','shoulderBust'
                           ,'strappedLinear','strappedSlant','scyeDepth','chestHeight'
                                    ,'waistHeight','hipHeight','stomachHeight','height','weight','shoulderNaturalwaist','naturalwaistWaist',
                                    'cervicalNaturalwaist','naturalwaistWaistBack','shoulderSeam','bustToBust',
                                    'legLengthLowWaist','legLengthHighWaist','jacketLengthShort','jacketLengthLong']
linear_list_loosefit             = ['legLength','inseam','frontTorsoLength','armsLength','rise','backTorsoLength',
                           'waistToKneeLength','waistDepth','upperArmLength','foreArmLength','jacketLength','shirtLength','kneetoAnkleLength',
                            'sleeveLength','frontKneeLength','frontWaistLength','centerBackLength','centerFrontLength','cervicalLength',
                            'sideseam','armhole','armholeLevel','naturalWaistLength','halfSleeveLength','chestDepth',
                                    'stomachDepth','hipDepth','armInseam','sleeveLengthFull','vestBack','vestFront','strappedLinear','strappedSlant','scyeDepth','height','weight']

linear_list_female_loosefit      = ['legLength','inseam','frontTorsoLength','armsLength','rise','backTorsoLength',
                           'waistToKneeLength','waistDepth','upperArmLength','foreArmLength','jacketLength','shirtLength','kneetoAnkleLength',
                            'sleeveLength','frontKneeLength','frontWaistLength','centerBackLength','centerFrontLength','cervicalLength',
                            'sideseam','armhole','armholeLevel','naturalWaistLength','halfSleeveLength','chestDepth',
                            'stomachDepth','hipDepth','armInseam','sleeveLengthFull','vestBack','vestFront','shoulderWaist','shoulderBust'
                                    ,'strappedLinear','strappedSlant','scyeDepth','height', 'weight']

circum_list_male             = ['chest','stomach','waist','thigh','calfMuscle','kneeGirth','neck',
                           'hip','upperwaist','upperNeck','upperHip','naturalWaistGirth','biceps','wrist','thighSlant',
                                'armhole','upperKneeGirth','highWaist','midWaist','lowWaist','foreArmGirth','underChest','ankleGirth','midThighGirth']

circum_list_female      = ['chest','stomach','waist','thigh','calfMuscle','kneeGirth','neck'
                            ,'hip','upperwaist','overBust','underBust','upperNeck','upperHip','naturalWaistGirth'
                           ,'biceps','wrist','thighSlant','armhole','upperKneeGirth','highWaist','midWaist','lowWaist','foreArmGirth','ankleGirth'
                           ,'midThighGirth']
circum_list_male_loosefit            = ['chest','stomach','waist','thigh','calfMuscle','kneeGirth','neck',
                           'hip','upperwaist','upperNeck','upperHip','naturalWaistGirth','biceps','wrist','elbowGirth']

circum_list_female_loosefit     = ['chest','stomach','waist','thigh','calfMuscle','kneeGirth','neck'
                            ,'hip','upperwaist','overBust','underBust','upperNeck','upperHip','naturalWaistGirth',
                                   'biceps','wrist','elbowGirth']


# 3D Body Measurement

maledir     = basedir + '/modelspose/3D_Body/male_data'
femaledir   = basedir + '/modelspose/3D_Body/female_data'
difference = 5

mu_male = {'mu_stature' : 1774.263, 'mu_weight' : 4.350045, 'mu_chest' : 1021.941, 'mu_waist' : 894.4047,
           'mu_hips' : 1029.496, 'mu_inseam' : 796.153, 'mu_fitness' : 4.559389}


mu_female = {'mu_stature' : 1642.32, 'mu_weight' : 3.997508, 'mu_chest' : 928.5578, 'mu_waist' : 756.4329,
           'mu_hips' : 1023.178, 'mu_inseam' : 755.1578, 'mu_fitness' : 4.012371}


## recommend

basedir = os.getcwd()
datadir = basedir+'/data'

brand_link = "https://commonms.s3.ap-south-1.amazonaws.com/mysize_utils/brandlink/link.json"
#Common data on tolerances & thresholds from AWS
female_data = "https://commonms.s3.ap-south-1.amazonaws.com/mysize_utils/RecommendationCommon/female_data.py"
male_data = "https://commonms.s3.ap-south-1.amazonaws.com/mysize_utils/RecommendationCommon/male_data.py"

#Brand_info files
Common = {"India":"https://commonms.s3.ap-south-1.amazonaws.com/brand_info_Files/Existing+Brand_Infos/brand_info-India.json",
          "US":"https://commonms.s3.ap-south-1.amazonaws.com/brand_info_Files/Existing+Brand_Infos/brand_info-US.json",
          "UK":"https://commonms.s3.ap-south-1.amazonaws.com/brand_info_Files/Existing+Brand_Infos/brand_info-UK.json"}

l = ['stomach','chest','waist','overBust','hip','inseam','underBust','legLength','naturalWaistGirth', 'cervicalLength']
straight_apparels = ['shirt','coat','blazer','jacket','suit','suits','coats','jackets','shirts','blazers','kurti', 'kurtis','kurta','kurtas','trench coat','trench coats']

level1 = ["shoulderAcross"]
level2 = {"GIRTHS":["chest","waist","overBust","naturalWaistGirth","stomach","hip","underBust"],
          "LINEARS":["sleeveLength","inseam","halfSleeveLength","armsLength","jacketLength","shirtLength","legLength","upperArmLength","centerBackLength"]}
level3 = ["cervicalLength","height"]
error_acceptance = {"GIRTHS":10,"LINEARS":5}
exception = ['upperNeck','neck','shoulderAcross','halfSleeveLength','armsLength','sleeveLength','jacketLength','shirtLength','thigh','cervicalLength','sleeveLengthFull']
preference_points = ['stomach','chest','overBust','shoulderAcross']


### bmi measurement info
linears_bmi= {"inseam":[8341,9559],"legLength":[8209,9559],"sleeveLength":[7706,5610]}
male_circum_bmi ={"chest":"chest","stomach":"stomach","waist":"waist","naturalWaistGirth":"naturalWaist","hip":"hip","thigh":"thigh","neck":"neck"}
female_circum_bmi ={"overBust":"overBust","stomach":"stomach","waist":"waist","naturalWaistGirth":"naturalWaist","hip":"hip","underBust":"underBust"}

measurements_bmi = {
    "chest": -1,
    "hip": -1,
    "inseam": -1,
    "legLength": -1,
    "naturalWaistGirth": -1,
    "neck": -1,
    "sleeveLength": -1,
    "stomach": -1,
    "thigh": -1,
    "waist": -1
}

# merchant asked for 3D mesh
merchant_3d = ['dashboard@testing.com', 'vigor@novo.com']




