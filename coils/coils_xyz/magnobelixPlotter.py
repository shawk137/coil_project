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

plt.pcolormesh(X1,Y1,mag_in_area1, vmin = 0, vmax= 0.015)
plt.title('B Field calculated using filaments')
plt.plot([0,0.2],[0.1,0.4])
plt.colorbar()
plt.show()

plt.pcolormesh(X1,Y1,mag_in_area1, vmin = 0, vmax= 0.015)
plt.title('B Field calculated with individual Windings')
plt.colorbar()
plt.show()

plt.pcolormesh(X1,Y1,mag_in_area1-mag_in_area2, vmin = -0.0005, vmax= 0.0005)
plt.title('Difference in Tesla')
plt.colorbar()
plt.show()


plt.pcolormesh(X1,Y1,(mag_in_area1-mag_in_area2)/mag_in_area2*100, vmin = -5, vmax= 5)
plt.title('Difference in percent')
plt.colorbar()
plt.show()