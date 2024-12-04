"""
This is the story of Magnobelix, the fat cousin of Magnetix. The roman empire streches across the european continent. All of Gaul is under roman controll.
 All of Gaul??? No! A small village has kept the romans out, thanks to Magnetix and his magix repelling force, that repells
 both romans and coils.
 Magnetix has one weakpoint however, his enemies must never find out. He can only cast his spells with all coils being in
 the corrext order. Sometimes he forgets and has to recharge his powers with Danix. Only Danix knows how to correct the
 currents in the coils, so that Megnetix' powers can thrive.
"""
from windingCoordinateGenerator import *
import os

# current script path
script_path = os.path.abspath(__file__)
parent_folder = os.path.dirname(script_path)

mu_0 = 1.25663706127e-6


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
    coil_nr = 1 #int(input("from which coil do you want to know the force?"))
    coilCoordlist = loadAndScale(f'{parent_folder}/coilData/coil_coordinates0.txt', 12, 0.2/100) # [12, 160, 3] = [coils, points, xyz] !!!/100 bc: convert from fusion (cm) units!!!
    windingCoordList, windNr = thick_coils_coordinates(coilCoordlist, nr_x_windings=16, nr_y_windings=10, wire_diameter=2.1)
    I1 = 1064#14.7e+3 #A
    I2 = 520#8.17e+3 #A
    I3 = 703#9.7e+3 #A
    I_list = np.array([I1, I2, I3, -I3, -I2, -I1, I1, I2, I3, -I3, -I2, -I1])
    I_winding_list = I_list / windNr
    I_winding_list = np.repeat(I_winding_list, windNr)
    force = get_force(coil_nr, coilCoordlist, I_list)
    moment = get_moments(coil_nr, coilCoordlist, I_list)
    total_moment = np.sqrt(np.sum(moment**2))
    print("The total moment for coil {} is {:.3f} Nm\nThe components are: \n{} Nm\n".format(coil_nr, total_moment, np.round(moment, 3)))
    if True: #3D Plot
        ax = plt.figure().add_subplot(projection='3d')
        CGlist = coilCG(coilCoordlist)
        for i in range(len(coilCoordlist)):
            #panCk = pancakeCoordList[i]
            cl = coilCoordlist[i]
            #ax.plot(panCk[:,0], panCk[:,1], panCk[:,2], label='pancake')
            ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], label='coil_{}'.format(i))
            CG = CGlist[i]
            vec = get_field(CG, cl, I_list[i])
            #print('Bfield in CG', str(i), ' ', np.linalg.norm(vec))
            # vecList = CGvectors(CGlist)
            # vec = vecList[i] * 20
            vec *= 1e+0
            #print("len:", np.linalg.norm(vec))
            ax.plot([CG[0], CG[0]+vec[0]], [CG[1], CG[1]+vec[1]], [CG[2], CG[2]+vec[2]])
            ax.plot(CG[0], CG[1], CG[2],'.')
        mp = wire_mid_points(coilCoordlist[coil_nr])
        plot_force = 1e-2*force
        force_tot = np.sum(force, axis=0)
        plot_force_tot = force_tot * 1e-3
        print("total force = {} N".format(np.round(force_tot, 3)))
        print("total force magnitude = {:.3f} N".format(np.linalg.norm(force_tot)))
        ax.plot([CGlist[coil_nr][0], CGlist[coil_nr][0]+plot_force_tot[0]],
                [CGlist[coil_nr][1], CGlist[coil_nr][1]+plot_force_tot[1]],
                [CGlist[coil_nr][2], CGlist[coil_nr][2]+plot_force_tot[2]], color="black", label="total force")
        plot_moment = moment * 1e-2
        ax.plot([CGlist[coil_nr][0], CGlist[coil_nr][0] + plot_moment[0]],
                [CGlist[coil_nr][1], CGlist[coil_nr][1] + plot_moment[1]],
                [CGlist[coil_nr][2], CGlist[coil_nr][2] + plot_moment[2]], color="blue", label="total moment")
        for idx, point in enumerate(mp):
            ax.plot([point[0], point[0] + plot_force[idx, 0]], [point[1], point[1] + plot_force[idx, 1]], [point[2], point[2] + plot_force[idx, 2]])
        ax.legend()
        plt.show()