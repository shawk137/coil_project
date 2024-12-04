"""
This is the story of Magnobelix, the fat cousin of Magnetix. The roman empire streches across the european continent. All of Gaul is under roman controll.
 All of Gaul??? No! A small village has kept the romans out, thanks to Magnetix and his magix repelling force, that repells
 both romans and coils.
 Magnetix has one weakpoint however, his enemies must never find out. He can only cast his spells with all coils being in
 the corrext order. Sometimes he forgets and has to recharge his powers with Danix. Only Danix knows how to correct the
 currents in the coils, so that Megnetix' powers can thrive.
"""
import numpy as np

from windingCoordinateGenerator import *
import os

# current script path
script_path = os.path.abspath(__file__)
parent_folder = os.path.dirname(script_path)

mu_0 = 1.25663706127e-6
mm = 0.001
cm = 0.01
m = 1

def wire_vectors(xyzCoord):
    """
    :param xyzCoord: coordinates of the coil
    :return: the vectors between each two consecutive points xyzCoords including last to first, shape like input
    """
    shape = np.shape(xyzCoord)
    wires = np.zeros((shape[0] + 1, shape[1]))
    wires[:-1, :] = xyzCoord
    wires[-1, :] = xyzCoord[0, :]
    wireVectors = np.diff(wires, axis=0)
    return wireVectors


def wire_mid_points(xyzCoord):
    """
    :param xyzCoord: coordinates of the coil
    :return: points in between the original xyzCoords, shape like input
    """
    midpoints = xyzCoord + wire_vectors(xyzCoord) * 0.5
    return midpoints


def get_field(point, xyzCoord, I):
    """
    :param point: point in which the B field is calculated
    :param xyzCoord: coordinates of the coil
    :param I: current of the coil
    :return: B field at point, created by given coil (xyzCoord) with current (I), shape: [xyz] numpy array
    """
    B = np.zeros(3)
    rest_mp = wire_mid_points(xyzCoord)
    rest_vectors = wire_vectors(xyzCoord)
    dist = point - rest_mp
    dist_scalar = np.linalg.norm(dist, axis=1)
    dist_scalar[np.argwhere(dist_scalar == 0)] = 1 # Avoid division by 0! Magnetix is happy :))
    # This works since np.cross gives zero for dist = 0!
    B += mu_0 / (4 * np.pi) * np.sum(I * np.cross(rest_vectors, dist) / dist_scalar[:, np.newaxis] ** 3, axis=0) #Biot-Savart
    return B


def get_force(coil_nr, coilCoordlist, I_list, include_self=True):
    """
    :param coil_nr: The index of the coil, where the forces are calculated
    :param coilCoordlist: list of numpy arrays containing xyz coil coordinates
    :param I_list: list of currents through each coil, !signs must be right!
    :param include_self: boolean, if True points on the coil under test are also considered.
    :return: return force vectors for every point of the coil in a numpy [n, 3] array
    """
    cut = coilCoordlist[coil_nr]  #cut = coil under test
    cut_mp = wire_mid_points(cut)
    cut_vectors = wire_vectors(cut)
    force_cut = np.zeros_like(cut_mp)
    for idx_cut, point in enumerate(cut_mp):
        B = np.zeros((1, 3))
        for idx_rest, xyzCoord in enumerate(coilCoordlist):
            if include_self:
                B += get_field(point, xyzCoord, I_list[idx_rest])
            else:
                if coil_nr != idx_rest:
                    B += get_field(point, xyzCoord, I_list[idx_rest])
        force_cut[idx_cut, :] = I_list[coil_nr] * np.cross(cut_vectors[idx_cut], B)
    return force_cut

def get_magnitude_in_area(coilCoordlist, I_list, testPoints):
    B_field = np.zeros_like(testPoints)
    for idx, point in enumerate(testPoints):
        B = np.zeros((1, 3))
        for i, xyzCoord in enumerate(coilCoordlist):
            B += get_field(point, xyzCoord, I_list[i])
        B_field[idx] = B
    return np.linalg.norm(B_field, axis=1)


def get_moments(coil_nr, coilCoordlist, I_list):
    """
    :param coil_nr: The index of the coil, where the forces are calculated
    :param coilCoordlist: list of numpy arrays containing xyz coil coordinates
    :param I_list: list of currents through each coil, !signs must be right!
    :return: return moment vector for the center of gravity as numpy [3] array
    """
    CG = np.mean(coilCoordlist[coil_nr], axis=0)
    force = get_force(coil_nr, coilCoordlist, I_list, include_self=False)
    mp = wire_mid_points(coilCoordlist[coil_nr])
    dist = mp - CG
    moment = np.sum(np.cross(dist, force), axis=0)
    return moment

def points_in_area(x_nr, y_nr, dist, center, xDir, yDir):
    xDir = np.asarray(xDir)
    xDir =xDir / np.linalg.norm(xDir)
    yDir = np.asarray(yDir)
    yDir = yDir / np.linalg.norm(yDir)
    x_points = np.linspace(-(x_nr - 1) / 2 * dist, (x_nr - 1) / 2 * dist,
                           x_nr)
    y_points = np.linspace(-(y_nr - 1) / 2 * dist, (y_nr - 1) / 2 * dist,
                           y_nr)
    areaCoord = np.zeros((x_nr * y_nr, 2))
    for i in range(x_nr):
        for j in range(y_nr):
            areaCoord[i * y_nr + j, :] = [x_points[i], y_points[j]]
    #points2D = np.reshape(areaCoord,(x_nr,y_nr))
    points3D = np.zeros((x_nr*y_nr, 3))
    for j in range(len(areaCoord)):
        points3D[j,:] = center + areaCoord[j, 0] * xDir + areaCoord[j, 1] * yDir
    return points3D, np.linalg.norm(center)+x_points, np.linalg.norm(center)+y_points #Sis is a great deal of pfusch regarding the center offset.

def thick_coils_coordinates(coilCoordList, nr_x_windings, nr_y_windings, wire_diameter):
    """ 
    :param coilCoordList: list of numpy arrays containing xyz coil coordinates
    :param nr_x_windings: number of windings in x direction
    :param nr_y_windings: number of windings in y direction
    :param wire_diameter: diameter of the wire including insulation
    :return: list of numpy arrays containing xyz coil coordinates of the individual windings
    :return: number of windings
    """
    #--------------generate coordinates of the windings inside the crossection. (0,0) is the midpoint = filament
    x_points = np.linspace(-(nr_x_windings-1)/2*wire_diameter, (nr_x_windings-1)/2*wire_diameter, nr_x_windings)
    y_points = np.linspace(-(nr_y_windings-1)/2*wire_diameter, (nr_y_windings-1)/2*wire_diameter, nr_y_windings)
    crosssectionCoord = np.zeros((nr_x_windings*nr_y_windings, 2))
    for i in range(nr_x_windings):
        for j in range(nr_y_windings):
            crosssectionCoord[i*nr_y_windings+j, :] = [x_points[i], y_points[j]]

    #--------------Prepare coil coordinates and coordinate system along the filament----------------------------------------
    CGlist = coilCG(coilCoordList)
    circVecList=[]
    for i, xyzCoord in enumerate(coilCoordList):
        CG = CGlist[i]
        circVec = np.cross([0,0,1],CG)# vector orthogonal to z axis and connection between z axis and CG
        circVec /= np.linalg.norm(circVec)# normalized
        circVecList.append(circVec)

    X = np.asarray([1,0,0])
    Y = np.asarray([0,1,0])

    steerVecList = circVecList
    steerVecList[0] = Y
    steerVecList[5] = Y
    steerVecList[6] = Y
    steerVecList[11] = Y
    xDirList, yDirList, normalList = customOrientePlanes(coilCoordList, steerVecList)
    windingCoordlist = []   
    for i in range(len(coilCoordList)):
        coil = coilCoordList[i]
        xDir = xDirList[i]
        yDir = yDirList[i]
        for j in range(len(crosssectionCoord)):
            windingCoordlist.append(coil+crosssectionCoord[j,0]*xDir+crosssectionCoord[j,1]*yDir)
    return windingCoordlist, nr_x_windings * nr_y_windings


if __name__ == "__main__":
    print('hi, how are you dooin?')
    indWindings = True
    if indWindings:
        print('calculating Magnetic Field from individualwindings = fat coils')
    else:
        print('calculacting magnetic field from infinitely sthin filaments')
    coilCoordlist = loadAndScale(f'{parent_folder}/coilData/coil_coordinates0.txt', 12, 0.2/100) # [12, 160, 3] = [coils, points, xyz] !!!/100 bc: convert from fusion (cm) units!!!
    windingCoordList, windNr = thick_coils_coordinates(coilCoordlist, nr_x_windings=16, nr_y_windings=10, wire_diameter=2.1*mm)
    I1 = 1064#14.7e+3 #A
    I2 = 520#8.17e+3 #A
    I3 = 703#9.7e+3 #A
    I_list = np.array([I1, I2, I3, -I3, -I2, -I1, I1, I2, I3, -I3, -I2, -I1])
    if indWindings:
        I_list = I_list / windNr
        I_list = np.repeat(I_list, windNr)
    else:
        windingCoordList = coilCoordlist
    areax = 20
    areay = 20
    area3D, X, Y = points_in_area(areax,areay,0.02,[0,0.2,0],[0,0,1],[0,1,0])
    print('area:', area3D)
    mag_in_area = get_magnitude_in_area(windingCoordList, I_list, area3D)
    mag_in_area = np.reshape(mag_in_area, (areax,areay))
    print(mag_in_area)
    plt.pcolormesh(X,Y,mag_in_area, vmin = 0, vmax= 0.015)
    plt.colorbar()
    plt.show()
