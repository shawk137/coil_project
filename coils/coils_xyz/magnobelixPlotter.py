import numpy as np
import matplotlib.pyplot as plt
import os
script_path = os.path.abspath(__file__)
parent_folder = os.path.dirname(script_path)

mag_in_area1 = np.loadtxt(f'{parent_folder}/magnitudeWind.txt')
X1 = np.loadtxt(f'{parent_folder}/XWind.txt')
Y1 = np.loadtxt(f'{parent_folder}/YWind.txt')

mag_in_area2 = np.loadtxt(f'{parent_folder}/magnitudeFil.txt')
X2 = np.loadtxt(f'{parent_folder}/XFil.txt')
Y2 = np.loadtxt(f'{parent_folder}/YFil.txt')

if any(X1 != X2) or any(Y1 != Y2):
    raise Exception('Coordinates of mag field 1 and 2 do not match!!?!')

#plot difference in a line plot
fig1 = plt.figure('fig1')
plasmaFil = np.loadtxt(f'{parent_folder}/plasmaFil.txt')
plasmaW = np.loadtxt(f'{parent_folder}/plasmaW.txt')
plt.plot(np.arange(len(plasmaW)), 100*(plasmaW-plasmaFil)/plasmaFil)
plt.xlabel('point on surface')
plt.ylabel('B-Field deviation in %')
plt.show

fig2 = plt.figure()
#extract phi = 0 slice of the plasma boundary
'''ax = plt.figure().add_subplot(projection='3d')

x = np.loadtxt('coilData/surface1.txt')
y = np.loadtxt('coilData/surface2.txt')
z = np.loadtxt('coilData/surface3.txt')
n = np.arange(0,256,8)
n = np.append(n,n[0]) #close the loop

ax.plot(x[n], y[n], z[n], label='parametric curve')
ax.legend()
plt.show()'''

scale = 0.2
offsetY = 0.2
x = np.loadtxt('coilData/surface1.txt') * scale
y = np.loadtxt('coilData/surface2.txt') * scale
z = np.loadtxt('coilData/surface3.txt') * scale + offsetY
n = np.arange(0,256,8)
n = np.append(n,n[0]) #close the loop
plasma0x = x[n]
plasma0y = z[n]

plt.pcolormesh(X1,Y1,mag_in_area1, vmin = 0, vmax= 0.015, shading = 'nearest', cmap = 'Spectral' )
plt.plot(plasma0x,plasma0y,'r')
plt.title('B Field calculated using filaments')
plt.colorbar()
plt.axis('equal')
plt.show()

plt.pcolormesh(X1,Y1,mag_in_area1, vmin = 0, vmax= 0.015, shading = 'nearest', cmap = 'Spectral' )
plt.plot(plasma0x,plasma0y,'r')
plt.title('B Field calculated with individual Windings')
plt.colorbar()
plt.show()

plt.pcolormesh(X1,Y1,mag_in_area1-mag_in_area2, vmin = -0.0005, vmax= 0.0005, shading = 'nearest', cmap = 'Spectral' )
plt.plot(plasma0x,plasma0y,'r')
plt.title('Difference in Tesla')
plt.colorbar()
plt.show()


plt.pcolormesh(X1,Y1,(mag_in_area1-mag_in_area2)/mag_in_area2*100, vmin = -5, vmax= 5, shading = 'nearest', cmap = 'Spectral' )
plt.plot(plasma0x,plasma0y,'r')
plt.title('Difference in percent')
plt.colorbar()
plt.show()