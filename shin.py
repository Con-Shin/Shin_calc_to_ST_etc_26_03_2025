import numpy as np
from ase import Atoms
from ase.build import mx2, graphene
from ase.visualize import view
import os
from msspec.iodata import Data
#from iodata import Dataset
from ase  import Atom, Atoms
from msspec.calculator import MSSPEC, XRaySource
from msspec.utils import *
from ase.visualize import view
import shutil
from msspec.looper import Sweep, Looper

def calculate_superlattice_period(a1, a2, theta):
    """
    This is a function to calculate the period of a moire superlattice given lattice paramters and twiste angle.

    Parameters:
    a1 (float): bottom layer
    a2 (float): top layer
    theta (float): twiste angle

    Returns:
    float: period as L
    """
    theta_rad = np.radians(theta)  # conversion degree into radian
    numerator = a1 * a2
    denominator = np.sqrt(a1**2 + a2**2 - 2 * a1 * a2 * np.cos(theta_rad)) # This formula is refered to some paper.
    return numerator / denominator

def rotate_layer(atoms, theta, center=(0, 0, 0)):
    """
    This is to rotate either of layers around an arbitrary center.

    Parameters:
    atoms (Atoms): ASE Atoms
    theta (float): twiste angle
    center (tuple): center of rotation

    Returns:
    Atoms: twisted Atoms
    """
    theta_rad = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad),  np.cos(theta_rad), 0],
        [0, 0, 1]
    ]) # This matrix is to rotate anticlockwise.

    positions = atoms.get_positions() - np.array(center)
    rotated_positions = np.dot(positions, rotation_matrix.T) + np.array(center)
    atoms.set_positions(rotated_positions)
    return atoms

def create_twisted_bilayer(formula1, formula2, a1, a2, thickness1, thickness2, theta, interlayer_distance=6.5, vacuum=15):
    """
    This is to create our Van deer Waals heterostructure given each layers's lattice parameter, thickness, and their interlayer distance.

    Parameters:
    formula1 (str): bottom layer (This case: 'MoS')
    formula2 (str): top layer (This case: 'WSe2')
    a1 (float): bottom layer
    a2 (float): top layer
    thickness1 (float): bottom layer
    thickness2 (float): top layer
    theta (float): twiste angle
    interlayer_distance (float): 
    vacuum (float):

    Returns:
    Atoms:
    """
    # Here, we get the period of the moire superlattice.
    L = calculate_superlattice_period(a1, a2, theta)

    # Creation of the Bottom layer
    bottom_layer = mx2(formula=formula1, kind='2H', a=a1, thickness=thickness1, size=(1, 1, 1), vacuum=None)
    bottom_layer.positions[:, 2] += vacuum / 2 # z position of the bottom layer based on the vacuum
    bottom_layer = bottom_layer.repeat([70,70,1]) #Size of the bottom layer. At some cases, we might need big sized clusters.


    # Creation of the Top layer
    top_layer = mx2(formula=formula2, kind='2H', a=a2, thickness=thickness2, size=(1, 1, 1), vacuum=None)
    top_layer.positions[:, 2] += vacuum / 2 + thickness1 + interlayer_distance
    top_layer = top_layer.repeat([70,70,1])
    top_layer.rotate(60.,'z') # 
    bottom_layer.rotate(60.,'z') #
    top_layer = rotate_layer(top_layer, theta, center = (-17.380, 161.462, 26.285)) #Here, we rotate the top layer.

    Below here, we calculate lattice constants correspoding to our moire superlattice. The deduction of the lattice vectors comes from my pdf note. I will give you it.

    e = a2/(a1) - 1.0

    theta_rad = np.radians(theta)

    F1 = 2*(1+e)*np.cos(theta_rad)-(1+e)**2-1
    L1_x = a1*((1+e)*np.cos(theta_rad)-(1+e)**2)/(F1)
    L1_y = a1*((1+e)*np.sin(theta_rad)) / (F1)
    L1 = np.array([L1_x, L1_y, 0.0])

    F2 = 2*(1/(1+e)-2*np.cos(theta_rad)+(1+e))

    L2_x = a1*((1+e)+(np.sqrt(3)*np.sin(theta_rad)-np.cos(theta_rad)))/(F2)
    L2_y = a1*(np.sqrt(3)*(1+e)-(np.sqrt(3)*np.cos(theta_rad)+np.sin(theta_rad)))/(F2)
    L2 = np.array([L2_x, L2_y, 0.0])

    bilayer = bottom_layer + top_layer
    bilayer.set_cell([[L1[0],L1[1],0.0], [L2[0],L2[1],0.0], [0, 0, vacuum + thickness1 + thickness2 + interlayer_distance]]) # setting the moire superlattice as an unit cell
    bilayer.center(vacuum, axis=2)

    return bilayer, L, L1, L2

def calculate_number(L, a1=3.28, a2=3.16):
    """
    This is to get the number of atoms (more precisely, emitters) in a moire supercell simple way.

    Parameters:
    L (float): period of the moire supercell
    a1 (float): bottom layer
    a2 (float): top layer

    Returns:
    float:
    """

    ratio_bottom = L**2/(a1**2) # ratio of the squares by the lattice periods
    ratio_top = L**2/(a2**2)

    # If one originnal unit cell contain just one emitter, the total number of emitters in a moire supercell is equal to the ration above.
    total_atoms = ratio_bottom
    return total_atoms

BL, L, L1, L2 = create_twisted_bilayer(formula1='WSe2', formula2='MoS2', a1=3.28, a2=3.16, thickness1=3.19, thickness2=3.19, theta=1.08, interlayer_distance=6.50, vacuum=15)
total_atoms= calculate_number(L=L, a1=3.28, a2=3.16)

from ase.io import read

def is_inside_parallelogram(x, y, L1, L2, th_min_u, th_max_u, th_min_v, th_max_v):
    """
   This is to judge if a selected emitter is inside a moire supercell or not.

    :param x: 
    :param y:
    :param L:
    :return: True or False
    """

    v = np.array([x, y])

    A = np.column_stack([L1[:2], L2[:2]])  
    coeffs = np.linalg.solve(A, v)

    u, v = coeffs  

    # If the coefficients are (0 < u < 1) and (0 < v < 1), the emitter is to be insdie. 
    return (th_min_u < u < th_max_u) and (th_min_v < v < th_max_v)

import concurrent.futures
from scipy.spatial import cKDTree
from ase import Atoms

def process_center(args):
    """
   This is to get the coordinates of atoms inside the cylinder of a given radius centered on an emitter.
    """

    center, atoms, tree_2d, positions, radius, z_range = args
    x_c, y_c, z_c = center 
    idxs = tree_2d.query_ball_point([x_c, y_c], radius, p=2)
    selected_atoms = [
        atoms[i] for i in idxs if z_c + z_range[0] <= positions[i, 2] <= z_c + z_range[1]
    ]

    if selected_atoms:
        return Atoms(selected_atoms), center, radius, z_range[1] - z_range[0]
    return None

def extract_cylindrical_clusters_parallel(atoms, element="W", radius=3.0, z_range=(0.0, 10.0), L1=np.array([1, 0]), L2=np.array([1, 0]), th_min_u=-0.1, th_max_u=1, th_min_v=0, th_max_v=1):
    """
   This is to get all the cylinders and coordinates of the centering emitters that are inequivalen in a moire supercell twith a given radius in some z-range.
    """
    clusters = []
    centers = []
    radii = []
    heights = []

    w_positions = np.array([
        atom.position for atom in atoms 
        if atom.symbol == element and is_inside_parallelogram(atom.x, atom.y, L1, L2, th_min_u, th_max_u, th_min_v, th_max_v)
    ])
    if len(w_positions) == 0:
        return clusters, centers, radii, heights

    positions = np.array([atom.position for atom in atoms])

    positions_2d = np.array([[pos[0], pos[1]] for pos in positions]) 
    tree_2d = cKDTree(positions_2d)

    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        results = executor.map(process_center, [
            (center, atoms, tree_2d, positions, radius, z_range) for center in w_positions
        ])

    for result in results:
        if result:
            cluster, center, r, h = result
            clusters.append(cluster)
            centers.append(center)
            radii.append(r)
            heights.append(h)

    return clusters, centers, radii, heights

########### This is just to speed up the translation of all the coordinates of atoms. #########
from numba import njit, prange

@njit(parallel=True)
def translate_positions(positions, displacement):
    for i in prange(positions.shape[0]):
        positions[i, 0] += displacement[0]
        positions[i, 1] += displacement[1]
        positions[i, 2] += displacement[2]

def parallel_translate(atoms: Atoms, displacement):
    displacement = np.asarray(displacement, dtype=np.float64)
    translate_positions(atoms.positions, displacement)
################################################################################################

parallel_translate(BL, [17.380, -161.462, -15.750-0.845])


clusters, centers, radii, heights = extract_cylindrical_clusters_parallel(BL, element='W', radius=20.0, z_range=(-10.0, 15.0), L1=L1, L2=L2, th_min_u=0.0, th_max_u=1, th_min_v=0.0, th_max_v=1)


################ Below here, we do the theta-phi scanning to see a stereographic projection by calculating for all the inequvalent emitters and summing up all. #############
from mpi4py import MPI
from multiprocessing import Pool, cpu_count

import copy

import multiprocessing

import concurrent.futures
import time
from concurrent.futures import ProcessPoolExecutor

radii = (('Mo', 1.39),('S', 1.00),('W', 1.39), ('Se', 1.00))

def compute_cross_section(i):
 
    calc = MSSPEC(spectroscopy='PED', algorithm='expansion', folder=f'calc_{i}')
    calc.calculation_parameters.scattering_order = 1 # For now, I think only the series expansion is possible thougth the Matrix inversion is so heavy and should be patitioned with respect to the path operator matrix somehow.

    calc.tmatrix_parameters.lmax_mode = "imposed"
    calc.tmatrix_parameters.lmaxt = 20
    calc.muffintin_parameters.interstitial_potential = 12.5
    
    for atom in clusters[i]:
        atom.set('mean_square_vibration', 5e-03)
    
    for s, r in radii:
        [atom.set('mt_radius',r) for atom in clusters[i] if atom.symbol == s]

    calc.calculation_parameters.vibrational_damping = 'averaged_tl'
    calc.muffintin_parameters.charge_relaxation = True
    calc.detector_parameters.average_sampling = 'medium'
    calc.detector_parameters.angular_acceptance = 1.5
    calc.tmatrix_parameters.exchange_correlation = "hedin_lundqvist_complex"
    calc.source_parameters.energy = XRaySource.AL_KALPHA
    calc.source_parameters.theta = -45.0
    calc.source_parameters.phi = 0.0
    calc.calculation_parameters.RA_cutoff = 2

#    calc.calculation_parameters.renormalization_mode = 'G_n'
#    calc.calculation_parameters.renormalization_omega = 0.85 + 0.1j

    clusters[i].translate([-1*centers[i][0], -1*centers[i][1], -1*centers[i][2]])    
    clusters[i].absorber = get_atom_index(clusters[i], 0, 0, 0)
    calc.set_atoms(clusters[i])
    
    data = calc.get_theta_phi_scan(level='4s', kinetic_energy=1000)
    dset = data[-1]

    theta_value = dset.theta
    phi_value = dset.phi
    k_value = dset.energy
    d_value = dset.direct_signal
     
    calc.shutdown()
    return dset.cross_section, theta_value, phi_value, k_value, d_value

total_cross_section = 0
theta_values = 0
phi_values = 0
k_values = 0
d_values = 0

# We are doing a parallel processing for the calculation of each emitter with workers set based on our PC's performance.
with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    results = executor.map(compute_cross_section, range(len(clusters)))

for idx, (cross_section, theta, phi, k, d) in enumerate(results):

    if idx == 0:

        theta_values += theta
        phi_values += phi
        k_values += k
    total_cross_section += cross_section
    d_values += d

    mask = theta_values < 65
    filtered_theta = theta_values[mask]
    filtered_phi = phi_values[mask]
    filtered_cross_section = total_cross_section[mask]
    filtered_ke = k_values[mask]
    filtered_d = d_values[mask]
 
############################## Below here, we set a dataset to store all the results to be plotted and viewd. ########################################################################
FOLDER = "output"  
ke = 890
level = '4s'
data = Data('theta_phi scan [0]')
dset = data.add_dset(f"PED")
dset.add_columns(theta=filtered_theta, phi=filtered_phi, energy=filtered_ke, cross_section=filtered_cross_section, direct_signal=filtered_d)
title = ('Stereographic projection of {}({}) at {:.2f} eV''').format('W', level, 890)
xlabel = r'Angle $\phi$($\degree$)' 
ylabel = r'Signal (a. u.)'
view = dset.add_view("E = {:.2f} eV".format(ke), title=title, xlabel=xlabel, ylabel=ylabel,
    projection='stereo', colorbar=True, autoscale=True)

view.select('theta', 'phi', 'cross_section')

#data.view()
data.export(os.path.join(FOLDER, 'result_plot'))
fn = os.path.join(FOLDER, 'plot.hdf5')
data.save(fn)
