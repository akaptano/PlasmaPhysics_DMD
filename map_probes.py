## @package map_probes
## Contains routines for mapping between surface probe
## names, locations, and indices, accounting for dead probes
from plot_attributes import *
## A list of HIT-SI's dead probes
dead_probes = ['B_L01P045','B_S04P000','B_L07P180',\
    'B_S07T180','B_S05T000']
## Radial location for the surface probes in the order
## L01, L02, L03, L04, L07, L08, L09, L10, S08, S07, S06, S05,
## S04, S03, S02, S01, L05, L06
R = [0.4282, 0.4550, 0.4763, 0.4978, 0.4978, 0.4763,
     0.4549, 0.4282, 0.2061, 0.1760, 0.1384, 0.1007, 0.1007,
     0.1384, 0.1760, 0.2060, 0.5485, 0.5485]
## Axial location for the surface probes in the order
## L01, L02, L03, L04, L07, L08, L09, L10, S08, S07, S06, S05,
## S04, S03, S02, S01, L05, L06
Z = [-0.2220, -0.1737, -0.1292, -0.0847, 0.0847, 0.1292,
    0.1735, 0.2220, 0.2364, 0.2024, 0.1654, 0.1312, -0.1312, \
    -0.1654, -0.2024, -0.2366, -0.0300, 0.0300]
## Toroidal angle locations of the surface probes, not
## including the midplane probes
Phi = np.asarray([0, 45, 180, 225]) * pi / 180.0
## Toroidal angle locations of the midplane probes
midphi = np.linspace(0,2*pi-22.5*pi/180.0,16)
## Poloidal location for the surface probes in the order
## L01, L02, L03, L04, L07, L08, L09, L10, S08, S07, S06, S05,
## S04, S03, S02, S01, L05, L06
Theta = np.asarray([1.164, 1.113, 1.063, 1.012, \
    .774, .723, .673, .622, \
    .330, .279, .228, .177, \
    1.609, 1.558, 1.507, 1.456, \
    .930, .856])*2*pi/1.78568
## Dictionary object for mapping between
## urface probe names and surface probe positions
sp_name_dict = {
    'B_L01P000': [R[0], Z[0], Phi[0], Theta[0]],
    'B_L01T000': [R[0], Z[0], Phi[0], Theta[0]],
    'B_L01P045': [R[0], Z[0], Phi[1], Theta[0]],
    'B_L01T045': [R[0], Z[0], Phi[1], Theta[0]],
    'B_L01P180': [R[0], Z[0], Phi[2], Theta[0]],
    'B_L01T180': [R[0], Z[0], Phi[2], Theta[0]],
    'B_L01P225': [R[0], Z[0], Phi[3], Theta[0]],
    'B_L01T225': [R[0], Z[0], Phi[3], Theta[0]],
    'B_L02P000': [R[1], Z[1], Phi[0], Theta[1]],
    'B_L02T000': [R[1], Z[1], Phi[0], Theta[1]],
    'B_L02P045': [R[1], Z[1], Phi[1], Theta[1]],
    'B_L02T045': [R[1], Z[1], Phi[1], Theta[1]],
    'B_L02P180': [R[1], Z[1], Phi[2], Theta[1]],
    'B_L02T180': [R[1], Z[1], Phi[2], Theta[1]],
    'B_L02P225': [R[1], Z[1], Phi[3], Theta[1]],
    'B_L02T225': [R[1], Z[1], Phi[3], Theta[1]],
    'B_L03P000': [R[2], Z[2], Phi[0], Theta[2]],
    'B_L03T000': [R[2], Z[2], Phi[0], Theta[2]],
    'B_L03P045': [R[2], Z[2], Phi[1], Theta[2]],
    'B_L03T045': [R[2], Z[2], Phi[1], Theta[2]],
    'B_L03P180': [R[2], Z[2], Phi[2], Theta[2]],
    'B_L03T180': [R[2], Z[2], Phi[2], Theta[2]],
    'B_L03P225': [R[2], Z[2], Phi[3], Theta[2]],
    'B_L03T225': [R[2], Z[2], Phi[3], Theta[2]],
    'B_L04P000': [R[3], Z[3], Phi[0], Theta[3]],
    'B_L04T000': [R[3], Z[3], Phi[0], Theta[3]],
    'B_L04P045': [R[3], Z[3], Phi[1], Theta[3]],
    'B_L04T045': [R[3], Z[3], Phi[1], Theta[3]],
    'B_L04P180': [R[3], Z[3], Phi[2], Theta[3]],
    'B_L04T180': [R[3], Z[3], Phi[2], Theta[3]],
    'B_L04P225': [R[3], Z[3], Phi[3], Theta[3]],
    'B_L04T225': [R[3], Z[3], Phi[3], Theta[3]],
    'B_L07P000': [R[4], Z[4], Phi[0], Theta[4]],
    'B_L07T000': [R[4], Z[4], Phi[0], Theta[4]],
    'B_L07P045': [R[4], Z[4], Phi[1], Theta[4]],
    'B_L07T045': [R[4], Z[4], Phi[1], Theta[4]],
    'B_L07P180': [R[4], Z[4], Phi[2], Theta[4]],
    'B_L07T180': [R[4], Z[4], Phi[2], Theta[4]],
    'B_L07P225': [R[4], Z[4], Phi[3], Theta[4]],
    'B_L07T225': [R[4], Z[4], Phi[3], Theta[4]],
    'B_L08P000': [R[5], Z[5], Phi[0], Theta[5]],
    'B_L08T000': [R[5], Z[5], Phi[0], Theta[5]],
    'B_L08P045': [R[5], Z[5], Phi[1], Theta[5]],
    'B_L08T045': [R[5], Z[5], Phi[1], Theta[5]],
    'B_L08P180': [R[5], Z[5], Phi[2], Theta[5]],
    'B_L08T180': [R[5], Z[5], Phi[2], Theta[5]],
    'B_L08P225': [R[5], Z[5], Phi[3], Theta[5]],
    'B_L08T225': [R[5], Z[5], Phi[3], Theta[5]],
    'B_L09P000': [R[6], Z[6], Phi[0], Theta[6]],
    'B_L09T000': [R[6], Z[6], Phi[0], Theta[6]],
    'B_L09P045': [R[6], Z[6], Phi[1], Theta[6]],
    'B_L09T045': [R[6], Z[6], Phi[1], Theta[6]],
    'B_L09T180': [R[6], Z[6], Phi[2], Theta[6]],
    'B_L09P180': [R[6], Z[6], Phi[2], Theta[6]],
    'B_L09P225': [R[6], Z[6], Phi[3], Theta[6]],
    'B_L09T225': [R[6], Z[6], Phi[3], Theta[6]],
    'B_L10P000': [R[7], Z[7], Phi[0], Theta[7]],
    'B_L10T000': [R[7], Z[7], Phi[0], Theta[7]],
    'B_L10P045': [R[7], Z[7], Phi[1], Theta[7]],
    'B_L10T045': [R[7], Z[7], Phi[1], Theta[7]],
    'B_L10P180': [R[7], Z[7], Phi[2], Theta[7]],
    'B_L10T180': [R[7], Z[7], Phi[2], Theta[7]],
    'B_L10P225': [R[7], Z[7], Phi[3], Theta[7]],
    'B_L10T225': [R[7], Z[7], Phi[3], Theta[7]],
    'B_S08P000': [R[8], Z[8], Phi[0], Theta[8]],
    'B_S08T000': [R[8], Z[8], Phi[0], Theta[8]],
    'B_S08P045': [R[8], Z[8], Phi[1], Theta[8]],
    'B_S08T045': [R[8], Z[8], Phi[1], Theta[8]],
    'B_S08P180': [R[8], Z[8], Phi[2], Theta[8]],
    'B_S08T180': [R[8], Z[8], Phi[2], Theta[8]],
    'B_S08P225': [R[8], Z[8], Phi[3], Theta[8]],
    'B_S08T225': [R[8], Z[8], Phi[3], Theta[8]],
    'B_S07T000': [R[9], Z[9], Phi[0], Theta[9]],
    'B_S07P000': [R[9], Z[9], Phi[0], Theta[9]],
    'B_S07P045': [R[9], Z[9], Phi[1], Theta[9]],
    'B_S07T045': [R[9], Z[9], Phi[1], Theta[9]],
    'B_S07P180': [R[9], Z[9], Phi[2], Theta[9]],
    'B_S07T180': [R[9], Z[9], Phi[2], Theta[9]],
    'B_S07P225': [R[9], Z[9], Phi[3], Theta[9]],
    'B_S07T225': [R[9], Z[9], Phi[3], Theta[9]],
    'B_S06P000': [R[10], Z[10], Phi[0], Theta[10]],
    'B_S06T000': [R[10], Z[10], Phi[0], Theta[10]],
    'B_S06P045': [R[10], Z[10], Phi[1], Theta[10]],
    'B_S06T045': [R[10], Z[10], Phi[1], Theta[10]],
    'B_S06P180': [R[10], Z[10], Phi[2], Theta[10]],
    'B_S06T180': [R[10], Z[10], Phi[2], Theta[10]],
    'B_S06P225': [R[10], Z[10], Phi[3], Theta[10]],
    'B_S06T225': [R[10], Z[10], Phi[3], Theta[10]],
    'B_S05P000': [R[11], Z[11], Phi[0], Theta[11]],
    'B_S05T000': [R[11], Z[11], Phi[0], Theta[11]],
    'B_S05P045': [R[11], Z[11], Phi[1], Theta[11]],
    'B_S05T045': [R[11], Z[11], Phi[1], Theta[11]],
    'B_S05P180': [R[11], Z[11], Phi[2], Theta[11]],
    'B_S05T180': [R[11], Z[11], Phi[2], Theta[11]],
    'B_S05P225': [R[11], Z[11], Phi[3], Theta[11]],
    'B_S05T225': [R[11], Z[11], Phi[3], Theta[11]],
    'B_S04P000': [R[12], Z[12], Phi[0], Theta[12]],
    'B_S04T000': [R[12], Z[12], Phi[0], Theta[12]],
    'B_S04T045': [R[12], Z[12], Phi[1], Theta[12]],
    'B_S04P045': [R[12], Z[12], Phi[1], Theta[12]],
    'B_S04P180': [R[12], Z[12], Phi[2], Theta[12]],
    'B_S04T180': [R[12], Z[12], Phi[2], Theta[12]],
    'B_S04P225': [R[12], Z[12], Phi[3], Theta[12]],
    'B_S04T225': [R[12], Z[12], Phi[3], Theta[12]],
    'B_S03T000': [R[13], Z[13], Phi[0], Theta[13]],
    'B_S03P000': [R[13], Z[13], Phi[0], Theta[13]],
    'B_S03P045': [R[13], Z[13], Phi[1], Theta[13]],
    'B_S03T045': [R[13], Z[13], Phi[1], Theta[13]],
    'B_S03P180': [R[13], Z[13], Phi[2], Theta[13]],
    'B_S03T180': [R[13], Z[13], Phi[2], Theta[13]],
    'B_S03P225': [R[13], Z[13], Phi[3], Theta[13]],
    'B_S03T225': [R[13], Z[13], Phi[3], Theta[13]],
    'B_S02P000': [R[14], Z[14], Phi[0], Theta[14]],
    'B_S02T000': [R[14], Z[14], Phi[0], Theta[14]],
    'B_S02P045': [R[14], Z[14], Phi[1], Theta[14]],
    'B_S02T045': [R[14], Z[14], Phi[1], Theta[14]],
    'B_S02P180': [R[14], Z[14], Phi[2], Theta[14]],
    'B_S02T180': [R[14], Z[14], Phi[2], Theta[14]],
    'B_S02P225': [R[14], Z[14], Phi[3], Theta[14]],
    'B_S02T225': [R[14], Z[14], Phi[3], Theta[14]],
    'B_S01P000': [R[15], Z[15], Phi[0], Theta[15]],
    'B_S01T000': [R[15], Z[15], Phi[0], Theta[15]],
    'B_S01P045': [R[15], Z[15], Phi[1], Theta[15]],
    'B_S01T045': [R[15], Z[15], Phi[1], Theta[15]],
    'B_S01P180': [R[15], Z[15], Phi[2], Theta[15]],
    'B_S01T180': [R[15], Z[15], Phi[2], Theta[15]],
    'B_S01P225': [R[15], Z[15], Phi[3], Theta[15]],
    'B_S01T225': [R[15], Z[15], Phi[3], Theta[15]],
    'B_L05P000': [R[16], Z[16], midphi[0], Theta[16]],
    'B_L05T000': [R[16], Z[16], midphi[0], Theta[16]],
    'B_L06P000': [R[17], Z[17], midphi[0], Theta[17]],
    'B_L06T000': [R[17], Z[17], midphi[0], Theta[17]],
    'B_L05P022': [R[16], Z[16], midphi[1], Theta[16]],
    'B_L05T022': [R[16], Z[16], midphi[1], Theta[16]],
    'B_L06P022': [R[17], Z[17], midphi[1], Theta[17]],
    'B_L06T022': [R[17], Z[17], midphi[1], Theta[17]],
    'B_L05P045': [R[16], Z[16], midphi[2], Theta[16]],
    'B_L05T045': [R[16], Z[16], midphi[2], Theta[16]],
    'B_L06P045': [R[17], Z[17], midphi[2], Theta[17]],
    'B_L06T045': [R[17], Z[17], midphi[2], Theta[17]],
    'B_L05P067': [R[16], Z[16], midphi[3], Theta[16]],
    'B_L05T067': [R[16], Z[16], midphi[3], Theta[16]],
    'B_L06P067': [R[17], Z[17], midphi[3], Theta[17]],
    'B_L06T067': [R[17], Z[17], midphi[3], Theta[17]],
    'B_L05P090': [R[16], Z[16], midphi[4], Theta[16]],
    'B_L05T090': [R[16], Z[16], midphi[4], Theta[16]],
    'B_L06P090': [R[17], Z[17], midphi[4], Theta[17]],
    'B_L06T090': [R[17], Z[17], midphi[4], Theta[17]],
    'B_L05P112': [R[16], Z[16], midphi[5], Theta[16]],
    'B_L05T112': [R[16], Z[16], midphi[5], Theta[16]],
    'B_L06P112': [R[17], Z[17], midphi[5], Theta[17]],
    'B_L06T112': [R[17], Z[17], midphi[5], Theta[17]],
    'B_L05P135': [R[16], Z[16], midphi[6], Theta[16]],
    'B_L05T135': [R[16], Z[16], midphi[6], Theta[16]],
    'B_L06P135': [R[17], Z[17], midphi[6], Theta[17]],
    'B_L06T135': [R[17], Z[17], midphi[6], Theta[17]],
    'B_L05T157': [R[16], Z[16], midphi[7], Theta[16]],
    'B_L05P157': [R[16], Z[16], midphi[7], Theta[16]],
    'B_L06P157': [R[17], Z[17], midphi[7], Theta[17]],
    'B_L06T157': [R[17], Z[17], midphi[7], Theta[17]],
    'B_L05P180': [R[16], Z[16], midphi[8], Theta[16]],
    'B_L05T180': [R[16], Z[16], midphi[8], Theta[16]],
    'B_L06P180': [R[17], Z[17], midphi[8], Theta[17]],
    'B_L06T180': [R[17], Z[17], midphi[8], Theta[17]],
    'B_L05P202': [R[16], Z[16], midphi[9], Theta[16]],
    'B_L05T202': [R[16], Z[16], midphi[9], Theta[16]],
    'B_L06P202': [R[17], Z[17], midphi[9], Theta[17]],
    'B_L06T202': [R[17], Z[17], midphi[9], Theta[17]],
    'B_L05P225': [R[16], Z[16], midphi[10], Theta[16]],
    'B_L05T225': [R[16], Z[16], midphi[10], Theta[16]],
    'B_L06P225': [R[17], Z[17], midphi[10], Theta[17]],
    'B_L06T225': [R[17], Z[17], midphi[10], Theta[17]],
    'B_L05P247': [R[16], Z[16], midphi[11], Theta[16]],
    'B_L05T247': [R[16], Z[16], midphi[11], Theta[16]],
    'B_L06P247': [R[17], Z[17], midphi[11], Theta[17]],
    'B_L06T247': [R[17], Z[17], midphi[11], Theta[17]],
    'B_L05P270': [R[16], Z[16], midphi[12], Theta[16]],
    'B_L05T270': [R[16], Z[16], midphi[12], Theta[16]],
    'B_L06P270': [R[17], Z[17], midphi[12], Theta[17]],
    'B_L06T270': [R[17], Z[17], midphi[12], Theta[17]],
    'B_L05P292': [R[16], Z[16], midphi[13], Theta[16]],
    'B_L05T292': [R[16], Z[16], midphi[13], Theta[16]],
    'B_L06P292': [R[17], Z[17], midphi[13], Theta[17]],
    'B_L06T292': [R[17], Z[17], midphi[13], Theta[17]],
    'B_L05P315': [R[16], Z[16], midphi[14], Theta[16]],
    'B_L05T315': [R[16], Z[16], midphi[14], Theta[16]],
    'B_L06P315': [R[17], Z[17], midphi[14], Theta[17]],
    'B_L06T315': [R[17], Z[17], midphi[14], Theta[17]],
    'B_L05P337': [R[16], Z[16], midphi[15], Theta[16]],
    'B_L05T337': [R[16], Z[16], midphi[15], Theta[16]],
    'B_L06P337': [R[17], Z[17], midphi[15], Theta[17]],
    'B_L06T337': [R[17], Z[17], midphi[15], Theta[17]]
}

## Radial positions of the IMP probe
imp_R = [0.3314, 0.3441, 0.3568, 0.3695, 0.3822, 0.3949, 0.4076, \
    0.4203, 0.4330, 0.4457, 0.4584, 0.4711, 0.4838, 0.4965, 0.5092, 0.5219, 0.5346]
## Z positions of the IMP probe
imp_Z = 0.011
## Toroidal position of the experimental IMP probe
imp_phi = 225*pi/180.0
## Toroidal positions of the 8 IMP probes in the BIG-HIT simulation
imp_phis8 = [1.1781, 0.3927, 5.8905, 5.1051, 4.3197, 3.5343, 2.7489, 1.9635]
## Toroidal positions of the 32 IMP probes in the BIG-HIT simulation
imp_phis32 = [1.1781,0.9817,0.7854,0.5890,0.3927,0.1963,0,6.0868,5.8905, \
    5.6941,5.4978,5.3014,5.1051,4.9087,4.7124,4.5160,4.3197,4.1233,3.9270, \
    3.7306,3.5343,3.3379,3.1416,2.9452,2.7489,2.5525,2.3562,2.1598,1.9635, \
    1.7671,1.5708,1.3744]
## Radial positions of the IMP probes in the BIG-HIT simulation
imp_rads = np.loadtxt(out_dir+'radial_points_imp.txt')
## Dictionary object for mapping between
## imp probe names and surface probe positions
imp_name_dict = {
    '01': [imp_R[0], imp_Z, imp_phi],
    '02': [imp_R[1], imp_Z, imp_phi],
    '03': [imp_R[2], imp_Z, imp_phi],
    '04': [imp_R[3], imp_Z, imp_phi],
    '05': [imp_R[4], imp_Z, imp_phi],
    '06': [imp_R[5], imp_Z, imp_phi],
    '07': [imp_R[6], imp_Z, imp_phi],
    '08': [imp_R[7], imp_Z, imp_phi],
    '09': [imp_R[8], imp_Z, imp_phi],
    '10': [imp_R[9], imp_Z, imp_phi],
    '11': [imp_R[10], imp_Z, imp_phi],
    '12': [imp_R[11], imp_Z, imp_phi],
    '13': [imp_R[12], imp_Z, imp_phi],
    '14': [imp_R[13], imp_Z, imp_phi],
    '15': [imp_R[14], imp_Z, imp_phi],
    '16': [imp_R[15], imp_Z, imp_phi],
    '17': [imp_R[16], imp_Z, imp_phi]
}
## Reads in the bowtie geometry coordinates
## and returns them
# @param directory Name of the directory which contains
#   'bowtie_locations.txt'
# @returns R Radial locations of the bowtie boundary
# @returns Z Z locations of the bowtie boundary
def get_bowtie(directory):
    RZ = np.loadtxt(directory + 'bowtie_locations.txt')
    R = np.reshape(RZ[:,0]*100, (242,121))
    Z = np.reshape(RZ[:,1]*100, (242,121))
    return R,Z
