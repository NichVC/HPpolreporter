# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:54:00 2021

@author: Nichlas Vous Christensen
@email: nvc@clin.au.dk
@phone: +45 23464522
@organization: Aarhus University, The MR Research Centre
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy
import time
import tkinter.filedialog
import tkinter
from datetime import date,datetime
from scipy.optimize import curve_fit
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import defaultPageSize

def LoadFile(fileName):
    """
    LoadFile reads a binary data file in the Prospa format and extracts the 
    x (float32) and y (complex64) data if the file meets the expected format criteria.

    Parameters:
    -----------
    fileName : str
        The path to the binary file to be loaded.

    Returns:
    --------
    tuple (numpy.ndarray, numpy.ndarray) or (None, None)
        If the file is valid, returns a tuple containing:
            - x : numpy.ndarray of float32, representing the x-axis data.
            - y : numpy.ndarray of complex64, representing the y-axis data.
        If the file is invalid or not found, returns (None, None).

    Raises:
    -------
    Exception:
        If the file format does not match expected Prospa format (owner, data type, version).
        If the file does not represent a complex 1D data file.
        If the data dimensions (height, depth, hyper) are not compatible with 1D data.
    """
    # Check if the file exists
    if os.path.isfile(fileName) is False:
        return None, None

    # Open the file in binary read mode
    f = open(fileName, "rb")

    # Validate file ownership - should be 'SORP'
    owner = f.read(4)
    if owner != b"SORP":
        raise Exception("Not a Prospa file")

    # Validate file format - should be 'ATAD'
    format = f.read(4)
    if format != b"ATAD":
        raise Exception("Not a Prospa data file")

    # Validate file version - should be '1.1V'
    version = f.read(4)
    if version != b"1.1V":
        raise Exception("Not a Prospa data V1.1 file")

    # Validate data type - should be 504 (complex 1D data)
    typeNr = np.fromfile(f, dtype=np.int32, count=1)
    if typeNr != 504:
        raise Exception("Not a complex 1D data file")

    # Read dimensions (width, height, depth, hypercomplexity)
    width = np.fromfile(f, dtype=np.int32, count=1)
    height = np.fromfile(f, dtype=np.int32, count=1)
    depth = np.fromfile(f, dtype=np.int32, count=1)
    hyper = np.fromfile(f, dtype=np.int32, count=1)

    # Ensure data is 1D (height, depth, hyper must all be 1)
    if height > 1 or depth > 1 or hyper > 1:
        raise Exception("Not a 1D data file")

    # Read x-axis (float32) and y-axis (complex64) data
    x = np.fromfile(f, dtype=np.float32, count=width[0])
    y = np.fromfile(f, dtype=np.complex64, count=width[0])

    return x, y


def gaussian(x, A, mu, sigma):
    """
    Computes the value of a Gaussian (normal distribution) function.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The input value(s) where the Gaussian function is evaluated.
    A : float
        The amplitude (peak value) of the Gaussian.
    mu : float
        The mean (center) of the Gaussian distribution.
    sigma : float
        The standard deviation (spread) of the Gaussian distribution.

    Returns:
    --------
    float or numpy.ndarray
        The computed Gaussian function value(s) at x.
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def lorentzian(x, A, x0, gamma):
    """
    Computes the value of a Lorentzian (Cauchy) function.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The input value(s) where the Lorentzian function is evaluated.
    A : float
        The amplitude (peak value) of the Lorentzian.
    x0 : float
        The position of the peak center (mean) of the Lorentzian distribution.
    gamma : float
        The half-width at half-maximum (HWHM), controlling the spread of the distribution.

    Returns:
    --------
    float or numpy.ndarray
        The computed Lorentzian function value(s) at x.
    """
    return A * (1 / np.pi) * (gamma / ((x - x0)**2 + gamma**2))

def gauss_lorentz(x, A_gaussian, mu_gaussian, sigma_gaussian, A_lorentzian, x0_lorentzian, gamma_lorentzian):
    """
    Computes a combined Gaussian-Lorentzian function, which is the sum of 
    a Gaussian and a Lorentzian distribution evaluated at x.

    Parameters:
    -----------
    x : float or numpy.ndarray
        The input value(s) where the combined function is evaluated.
    A_gaussian : float
        The amplitude (peak value) of the Gaussian component.
    mu_gaussian : float
        The mean (center) of the Gaussian distribution.
    sigma_gaussian : float
        The standard deviation (spread) of the Gaussian distribution.
    A_lorentzian : float
        The amplitude (peak value) of the Lorentzian component.
    x0_lorentzian : float
        The position of the peak center (mean) of the Lorentzian distribution.
    gamma_lorentzian : float
        The half-width at half-maximum (HWHM) of the Lorentzian distribution.

    Returns:
    --------
    float or numpy.ndarray
        The computed value(s) of the combined Gaussian-Lorentzian function at x.

    Notes:
    ------
    - This combination can be used to model phenomena where both Gaussian 
      and Lorentzian broadening mechanisms are present.
    - Useful for fitting spectral data where mixed broadening effects are observed.
    """
    return (
        gaussian(x, A_gaussian, mu_gaussian, sigma_gaussian) +
        lorentzian(x, A_lorentzian, x0_lorentzian, gamma_lorentzian)
    )

    
def Spectrum(folder, probe):
    """
    Analyzes a preprocessed and phased spectrum from a specified folder and 
    performs Gaussian-Lorentzian fitting to estimate the peak area.

    Parameters:
    -----------
    folder : str
        The path to the folder containing the 'spectrum_processed.1d' file.

    Returns:
    --------
    tuple (float, numpy.ndarray, numpy.ndarray, numpy.ndarray, list)
        - integral : The area under the fitted Gaussian-Lorentzian curve.
        - y : The real part of the raw spectral data (intensity values).
        - x : The x-axis data (usually frequency or time values).
        - fit_curve : The fitted Gaussian-Lorentzian curve values.
        - indices : The start and end indices of the region of interest (ROI) used for fitting.

    Notes:
    ------
    - The function expects a file named 'spectrum_processed.1d' in the specified folder.
    - The peak detection is performed using the maximum intensity value in the spectrum.
    - The fitting region is fixed at 50 points on each side of the peak.
    - The area under the fitted curve is calculated using the trapezoidal rule.
    """
    # Change to the specified directory
    os.chdir(folder)

    # Load the processed and phased spectrum (x: frequency/time, y: intensity)
    x, y = LoadFile('spectrum_processed.1d')
    y = np.real(y)  # Use the real part of the complex spectrum

    # Fixed window size for fitting
    points_around_peak = 50
    
     # Define window from probe selection for peak search 
    indices_region = np.zeros(2, dtype= int)
    indices_region[0] = np.abs(x - probe['peak_region'][1]).argmin()
    indices_region[1] = np.abs(x - probe['peak_region'][0]).argmin()

    # Find the index of the maximum intensity (peak)
    max_index = indices_region[0] + np.argmax(y[indices_region[0]:indices_region[1]])

    # Identify the peak and determine the region of interest (ROI)
    indices = [max_index - points_around_peak - 1, max_index + points_around_peak]

    # Estimate the integral of the raw data in the ROI
    integral = y[indices[0]:indices[1]].sum()

    # Prepare data for Gaussian-Lorentzian fitting
    ROI = y[indices[0]:indices[1]]
    fit_x = np.linspace(-points_around_peak, points_around_peak, points_around_peak * 2 + 1)

    # Initial parameter guess for the fitting: [A_gaussian, mu_gaussian, sigma_gaussian, 
    # A_lorentzian, x0_lorentzian, gamma_lorentzian]
    initial_guess = [integral / 2, 0.0, 1.0, integral / 2, 0.0, 1.0]

    # Perform curve fitting to optimize the parameters
    fit_params, _ = curve_fit(gauss_lorentz, fit_x, ROI, p0=initial_guess)

    # Generate the fitted curve using the optimized parameters
    fit_curve = gauss_lorentz(fit_x, *fit_params)

    # Calculate the area under the fitted curve using the trapezoidal rule
    integral = np.trapz(fit_curve, fit_x)

    return integral, y, x, fit_curve, indices

def IntegralPeak(folder, probe):
    """
    Computes the integral (area under the curve) of a peak from a preprocessed 
    and phased spectrum file ('spectrum_processed.1d').

    Parameters:
    -----------
    folder : str
        The path to the folder containing the 'spectrum_processed.1d' file.

    Returns:
    --------
    float
        The computed integral (sum of intensities) of the peak region.

    Notes:
    ------
    - The function assumes that the 'spectrum_processed.1d' file is located 
      in the specified folder.
    - The integration window is fixed at 50 points on each side of the peak.
    - The integration is performed on the real part of the spectral data.
    """
    # Change to the specified directory
    os.chdir(folder)

    # Load the processed and phased spectrum (x: frequency/time, y: intensity)
    x, y = LoadFile('spectrum_processed.1d')
    y = np.real(y)  # Use the real part of the complex spectrum

    # Fixed window size for integration
    points_around_peak = 50

    # Define window from probe selection for peak search
    indices_region = np.zeros(2, dtype= int)
    indices_region[0] = np.abs(x - probe['peak_region'][1]).argmin()
    indices_region[1] = np.abs(x - probe['peak_region'][0]).argmin()

    # Find the index of the maximum intensity (peak)
    max_index = indices_region[0] + np.argmax(y[indices_region[0]:indices_region[1]])

    # Determine the start and end indices of the region of interest (ROI)
    indices = [max_index - points_around_peak - 1, max_index + points_around_peak]

    # Compute the integral (sum of intensities) over the ROI
    integral = np.sum(y[indices[0]:indices[1]])

    return integral


def main_calculations(bioprobe, base_dir, thermal_dir, hyperpol_dir, t, discard):
    """
    Main function to calculate polarization and perform fitting on hyperpolarized (HP) 
    data collected from NMR experiments. The function:
    - Loads thermal and hyperpolarized data.
    - Applies receiver gain correction.
    - Corrects flip angle differences.
    - Computes temperature-dependent Boltzmann distribution.
    - Normalizes the data and performs curve fitting.
    - Generates plots for results.
    
    Parameters:
    - bioprobe: Index of the specific bioprobe (e.g., Pyruvate, Urea, etc.)
    - base_dir: Base directory for experiment data
    - thermal_dir: Directory for thermal data
    - hyperpol_dir: Directory for hyperpolarized data
    - t: Time in seconds from dissolution to experiment start
    - discard: Index to discard initial data points in the analysis
    
    Returns:
    - polarization_solid: estimated solid state polarization
    - polarization_liquid: estimated liquid state polarization 
    - enhancement_factor: difference between thermal and hyperpolarized integral 
    - bioprobe_name: the name of the chosen bioprobe
    - T1: estimated T1 relaxation inside the magnet
    - popt: fitting parameters from the exponential decay fit
    - t_list_delay: array of timestamps from each of the spectra
    """

    # Helper function to read parameter file
    def read_par(path):
        path += '\\acqu.par'
        with open(path) as f:
            acqu = f.read()
        Sample = re.search('Sample.*?"(.*?)"', acqu).group(1)
        nrScans = re.search('nrScans.*?= (.*?)\s', acqu).group(1)
        rxGain = re.search('rxGain.*?= (.*?)\s', acqu).group(1)
        pulseAngle = re.search('pulseAngle.*?= (.*?)\s', acqu).group(1)
        B0_MHz = re.search('b1Freq.*?= (.*?)\s', acqu).group(1)
        RoomT = re.search('CurrentTemperatureRoom.*?= (.*?)\s', acqu).group(1)
        BoxT = re.search('CurrentTemperatureBox.*?= (.*?)\s', acqu).group(1)
        dwellTime = re.search('dwellTime.*?= (.*?)\s', acqu).group(1)
        nrPnts = re.search('nrPnts.*?= (.*?)\s', acqu).group(1)
        repTime = re.search('repTime.*?= (.*?)\s', acqu).group(1)
        return Sample, nrScans, rxGain, pulseAngle, B0_MHz, RoomT, BoxT, dwellTime, nrPnts, repTime

    # Helper function to generate a list of time points from hyperpolarized data
    def make_t_list(hyperpol_list):
        timestamps = []
        t_list = []
        for i in range(1, len(hyperpol_list)):
            path = hyperpol_list[i] + '\\acqu.par'
            with open(path) as f:
                acqu = f.read()
            timestamps.append(re.search('startTime.*?".*?T(.*?)"', acqu).group(1))
            if i == 1:
                t_list.append(0)
            else:
                FMT = '%H:%M:%S.%f'
                dif = datetime.strptime(timestamps[-1], FMT) - datetime.strptime(timestamps[-2], FMT)
                t_list.append(float(dif.total_seconds() + t_list[-1]))
        return np.array(t_list)

    # Helper function for exponential decay fitting
    def func(x, A, T1):
        return A * np.exp(-x / T1)

    # Setup bioprobe information
    # Note: peak_region is not yet in use, but could be implemented for more peaks etc.
    bioprobe_info = {
        0: {'name': 'Urea', 'peak_region': [170, 160], 'T1': 60},
        1: {'name': 'Pyruvate', 'peak_region': [174, 166], 'T1': 64.4},
        2: {'name': 'C2-Pyruvate', 'peak_region': [207, 203], 'T1': 38.8},
        3: {'name': 'Fumarate', 'peak_region': [178, 169], 'T1': 50},
        4: {'name': 'HP001', 'peak_region': [26, 17], 'T1': 80},
        5: {'name': 'Alanine', 'peak_region': [178, 174], 'T1': 30},
        6: {'name': 'KIC', 'peak_region': [172.5, 170], 'T1': 60},
        7: {'name': 'C2-Pyruvate (D2O)', 'peak_region': [207, 203], 'T1': 72.9},
        8: {'name': 'Pyruvate (D2O)', 'peak_region': [174, 166], 'T1': 109.3},
        9: {'name': 'Z-OMPD (C1)', 'peak_region': [175, 165], 'T1': 68.8},
        10: {'name': 'Z-OMPD (C5)', 'peak_region': [185, 175], 'T1': 68.8}
    }

    # Get bioprobe-specific details
    bioprobe_data = bioprobe_info.get(bioprobe, None)
    if not bioprobe_data:
        raise ValueError("Invalid bioprobe value")
    
    bioprobe_name = bioprobe_data['name']
    T1 = bioprobe_data['T1']
    
    # Path setup for thermal and hyperpolarized data
    hyperpol_dir_temp = hyperpol_dir + ''
    thermal_dir += '\\00000'
    hyperpol_dir += f'\\0000{discard}' if discard < 10 else f'\\000{discard}'

    # Read parameter files for thermal and hyperpolarized data
    # Assigned globally as they are used in the PDF generator
    global thermal_par,hyperpol_par
    thermal_par = read_par(thermal_dir)
    hyperpol_par = read_par(hyperpol_dir)

    # Load spectra data for both thermal and hyperpolarized samples
    I_them, thermal_y, thermal_x, them_curve, them_curve_x = Spectrum(thermal_dir, bioprobe_data)
    I_hyp, hyp_y, hyp_x, hyp_curve, hyp_curve_x = Spectrum(hyperpol_dir, bioprobe_data)

    # Receiver gain and flip angle corrections
    RG_hyp = int(hyperpol_par[2])
    RG_them = int(thermal_par[2])
    flip_hyp = int(hyperpol_par[3])
    flip_them = int(thermal_par[3])

    # Constants and magnetic field setup
    B0_Hz = float(thermal_par[4]) * 10**6  # in Hz
    I_them *= 2.038**((RG_hyp - RG_them) / 6)  # Apply receiver gain correction
    angle_correction = np.sin(np.pi * flip_hyp / 180) / np.sin(np.pi * flip_them / 180)
    dilution = 1 # If any dilution between hyperpol and thermal

    # Boltzmann distribution calculations
    T = float(hyperpol_par[5])  # Temperature in Celsius
    k = scipy.constants.Boltzmann  # Boltzmann constant
    h = scipy.constants.h  # Planck's constant
    gyro_C = 10.7084 * 10**6  # Hz*T^-1 (gamma for 13C)
    B0 = B0_Hz / gyro_C  # Magnetic field strength in Tesla

    # Calculate Boltzmann distribution for hyperpolarization
    # Boltzmann dist calculation: https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Spectroscopy/Magnetic_Resonance_Spectroscopies/Nuclear_Magnetic_Resonance/NMR_-_Theory
    E_low = -1 / 2 * h * gyro_C * B0
    E_high = 1 / 2 * h * gyro_C * B0
    deltaE = E_high - E_low
    boltzmann_dist = 0.5 * (1 - np.exp(-deltaE / (k * (T + 273.15))))

    # Process hyperpolarized data
    hyperpol_list = [x[0] for x in os.walk(hyperpol_dir_temp)]
    I_hyp_ALL = []
    percentage_complete = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    j = 0
    for i in range(1, len(hyperpol_list)):
        I_hyp_ALL.append(IntegralPeak(hyperpol_list[i], bioprobe_data))
        if i % int((len(hyperpol_list) - 1) / 10) == 0:
            try:
                print(f'{percentage_complete[j]} % complete')
                j += 1
            except IndexError:
                continue

    # Generate time list based on acquisition timestamps
    t_list = make_t_list(hyperpol_list)
    t_list_delay = t_list[discard]

    # Normalize the hyperpolarized data
    I_hyp_ALL /= I_hyp_ALL[discard]

    # Perform exponential decay fitting for T1 estimation
    popt, _ = curve_fit(func, t_list[discard:], I_hyp_ALL[discard:], p0=[2, 60])
    
    # Go to script dir for temporary figure saving
    os.chdir(base_dir)

    # Plot the results (fitted curve and raw data)
    figure = plt.figure(figsize=[10, 5])
    I_hyp_at_t0 = I_hyp / np.exp(-t_list_delay / popt[1]) if discard != 0 else I_hyp
    new_normalize = I_hyp / I_hyp_at_t0
    I_hyp_ALL *= new_normalize
    plt.plot(t_list, func(t_list, popt[0], popt[1]) * new_normalize, 'k')
    plt.plot(t_list[discard:], I_hyp_ALL[discard:], 'b.', markersize=5)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized integral')
    plt.legend([f'Exponential fit: $I(t) = $${{e^{{-t/{popt[1]:.1f}}}}}$',f'Data points ($t_0 = {(t+t_list_delay):.1f}$ s)'])
    figure.savefig('temp.svg', format='svg')

    # Plot the thermal data (normalized)
    thermal_y /= max(thermal_y)
    them_curve /= max(them_curve)
    figure_thermal = plt.figure(figsize=[8, 5])
    plt.plot(thermal_x, thermal_y, 'C1-')
    plt.plot(thermal_x[them_curve_x[0]:them_curve_x[1]], them_curve, 'C0-')
    plt.axis([thermal_x[them_curve_x[0]]+3,thermal_x[them_curve_x[1]]-3,-0.1,1.1])
    plt.xlabel('$^{13}$C Chemical Shift (ppm)')
    plt.legend(['Thermal','Gaussian-Lorentzian fit'])
    figure_thermal.savefig('temp_thermal.svg', format='svg')
    
    # Plot the hyperpolarized data (normalized)
    hyp_y /= max(hyp_y)
    hyp_curve /= max(hyp_curve)
    figure_hyp = plt.figure(figsize=[8,5])
    plt.plot(hyp_x,hyp_y,'C1-')
    plt.plot(hyp_x[hyp_curve_x[0]:hyp_curve_x[1]],hyp_curve,'C0-')
    plt.axis([hyp_x[hyp_curve_x[0]]+3,hyp_x[hyp_curve_x[1]]-3,-0.1,1.1])
    plt.xlabel('$^{13}$C Chemical Shift (ppm)')
    plt.legend([f'Hyperpol ({hyperpol_dir[-5:]})','Gaussian-Lorentzian fit'])
    figure_hyp.savefig('temp_hyp.svg', format='svg')

    # Get the final polarization estimates
    polarization_liquid = I_hyp_at_t0 / I_them / angle_correction * boltzmann_dist / dilution
    polarization_solid = polarization_liquid / np.exp(-t/T1)
    enhancement_factor = I_hyp_at_t0 / I_them / angle_correction
    
    return polarization_solid, polarization_liquid, enhancement_factor, bioprobe_name, T1, popt, t_list_delay

def generate_pdf(polarization_solid, polarization_liquid, enhancement_factor, bioprobe_name, T1, popt, t, base_dir, thermal_dir, hyperpol_dir, discard, t_list_delay):
    """
    Generates a PDF report for hyperpolarization experiments. The function:
    - Creates a formatted PDF report detailing the polarization values, T1, and experiment parameters.
    - Adds metadata such as experiment start time, polarization values, and temperatures during thermal and hyperpolarization experiments.
    - Includes images representing thermal and hyperpolarization data.
    - Allows for scaling and positioning of images and text for better report presentation.

    Parameters:
    - polarization_solid: Solid polarization value (as a decimal)
    - polarization_liquid: Liquid polarization value (as a decimal)
    - enhancement_factor: Enhancement factor of the experiment
    - bioprobe_name: Name of the bioprobe used in the experiment
    - T1: T1 relaxation time of the bioprobe
    - popt: Optimized fitting parameters for the experiment
    - t: Time of experiment in seconds
    - base_dir: Base directory where the report will be saved
    - thermal_dir: Directory containing thermal data
    - hyperpol_dir: Directory containing hyperpolarization data
    - discard: Number of data points to discard from the start of the dataset
    - t_list_delay: Delay time for the hyperpolarization experiment

    Returns:
    - A PDF report saved in the specified base directory containing experiment details and plots.
    """    
    # Helper function to scale images
    def scale(drawing, scaling_factor):
        scaling_x = scaling_factor
        scaling_y = scaling_factor
        
        drawing.width = drawing.minWidth() * scaling_x
        drawing.height = drawing.height * scaling_y
        drawing.scale(scaling_x, scaling_y)
        return drawing

    # Helper function to add images to the PDF
    def add_image(my_canvas, image_path, scaling_factor, cords):
        drawing = svg2rlg(image_path)
        scaled_drawing = scale(drawing, scaling_factor=scaling_factor)
        renderPDF.draw(scaled_drawing, my_canvas, cords[0], cords[1])
    
    # Helper function to add text to the PDF
    def add_text(my_canvas, text, color, fontsize, y):
        PAGE_WIDTH  = defaultPageSize[0]
        my_canvas.setFillColorRGB(color[0]/255,color[1]/255,color[2]/255)
        my_canvas.setFont("Helvetica",fontsize)
        text_width = stringWidth(text,"Helvetica",fontsize)
        my_canvas.drawString((PAGE_WIDTH - text_width) / 2.0, y, text)
    
    # Main function starts here
    os.chdir(base_dir)

    thermal_dir_basename = os.path.basename(thermal_dir)
    hyperpol_dir_basename = os.path.basename(hyperpol_dir)
    
    my_canvas = canvas.Canvas(f'{base_dir}/Reports/{hyperpol_dir_basename}_HPpolreport_1p.pdf')
    
    # Title and metadata
    add_text(my_canvas, f'Hyperpolarization Report for Data ({bioprobe_name}):', [220, 50, 50], 20, 750)
    add_text(my_canvas, f"THERMAL:  {thermal_dir_basename}", [0, 0, 0], 10, 730)
    add_text(my_canvas, f"HYPERPOL:  {hyperpol_dir_basename}", [0, 0, 0], 10, 715)
    add_text(my_canvas, date.today().strftime("%B %d, %Y"), [0, 0, 0], 10, 695)
    
    # Polarization, T1, and other parameters
    add_text(my_canvas, f'Polarization Solid: {100*polarization_solid:.1f} %', [0, 0, 0], 15, 665)
    add_text(my_canvas, f'Assuming T1 = {T1:.1f} s when in ambient magnetic field', [0, 0, 0], 10, 650)
    add_text(my_canvas, f'Polarization Liquid: {100*polarization_liquid:.1f} %', [0, 0, 0], 15, 625)
    add_text(my_canvas, f'Enhancement factor: {enhancement_factor:.1e}', [0, 0, 0], 10, 610)
    add_text(my_canvas, f"T1 in magnet (60 MHz Spinsolve): {popt[1]:.1f} s", [0, 0, 0], 10, 585)
    
    # Time information
    if discard == 0:
        add_text(my_canvas, f"Time start: {t:.1f} s", [0, 0, 0], 10, 570)
    else:
        add_text(my_canvas, f"Time start: {(t + t_list_delay):.1f} s ({t:.1f} s before magnet + {t_list_delay:.1f} s in magnet)", [0, 0, 0], 10, 570)
    
    # Temperature during experiments
    add_text(my_canvas, f"Room (box) temperature during THERMAL experiment: {float(thermal_par[5]):.1f} \u00b0C ({float(thermal_par[6]):.1f} \u00b0C)", [0, 0, 0], 10, 555)
    add_text(my_canvas, f"Room (box) temperature during HYPERPOL experiment: {float(hyperpol_par[5]):.1f} \u00b0C ({float(hyperpol_par[6]):.1f} \u00b0C)", [0, 0, 0], 10, 540)
    
    # Add images
    add_image(my_canvas, 'temp.svg', 0.68, [0, 230])
    if discard > 0:
        add_text(my_canvas, f"NOTE: Analysis started from point {discard+1}.", [0, 0, 0], 8, 225)
        
    add_image(my_canvas, 'temp_thermal.svg', 0.4, [30, 40])
    add_image(my_canvas, 'temp_hyp.svg', 0.4, [300, 40])
    
    # Save PDF
    my_canvas.save()
    
    # Clean up temporary files
    os.remove('temp.svg')
    os.remove('temp_thermal.svg')
    os.remove('temp_hyp.svg')


def main():
    # The time from dissolution to start of acqusition.
    t = int(input('Time from measurement (in seconds): '))
    # The choice of bioprobe only affects the T1 at ambient field and the name used in the report.
    # More customization could be used if desired, e.g. more peak picking and so on.
    bioprobe = int(input('\nBioprobe?\nC1-Urea [0], C1-Pyruvate [1], C2-Pyruvate [2], Fumarate [3], HP001 [4], Alanine [5], KIC [6], C2-Pyruvate (deuterium) [7], Pyruvate (deuterium) [8], Z-OMPD (C1) [9], Z-OMPD (C5) [10]: '))
    # If accidentally the first few timepoints were clipped due to too high RG, these can be omitted from the calculation.
    discard = int(input('\nWhich point would you like to start with? First [1], Second [2] ... : '))-1
    
    # Get script dir for future reference and saving purposes.
    base_dir = os.getcwd()

    # Get the thermal data
    root = tkinter.Tk()
    root.withdraw()
    thermal_dir = tkinter.filedialog.askdirectory(title='Select thermal data folder', initialdir="Data/Thermal")
    
    # Get the hyperpolarized data from a 1-point test (1 sample)
    root = tkinter.Tk()
    root.withdraw()
    hyperpol_dir = tkinter.filedialog.askdirectory(title='Select hyper data folder (1-point)', initialdir="Data/HP_1p")

    print('\nData loaded, generating report...')
    
    # Run main function to do all processing and computation.
    polarization_solid, polarization_liquid, enhancement_factor, bioprobe_name, T1, popt, t_list_delay = main_calculations(bioprobe, base_dir, thermal_dir, hyperpol_dir, t, discard)
    # Run function to generate PDF with the computed results.
    generate_pdf(polarization_solid, polarization_liquid, enhancement_factor, bioprobe_name, T1, popt, t, base_dir, thermal_dir, hyperpol_dir, discard, t_list_delay)
    
    print('\nReport has been generated, program will exit in 5 seconds.')
    time.sleep(5)

if __name__ == "__main__":
    main()

