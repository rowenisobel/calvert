import cv2
import sep
from astroquery.astrometry_net import AstrometryNet
import numpy as np
import math
from astropy.io import fits
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astropy.table import Table, hstack
from astropy.wcs import WCS

__all__ = ['plate_pipeline']

class plate_pipeline():
    
    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.tif = cv2.imread(data_filename)
        self.tif_dat = self.tif.astype(np.float32)
        self.b, self.g, self.r = cv2.split(self.tif_dat)
        self.comb_tif = self.b + self.g + self.r
        self.invdat = -self.comb_tif+np.min(self.comb_tif)+np.max(self.comb_tif)
        self.fits_file = fits.writeto("{}.fits".format(data_filename[:-4]), self.invdat, overwrite = True)
        
    def astrometry(self, API, ra_guess, dec_guess, radius):
        ast = AstrometryNet()
        ast.api_key = API
        self.solve1 = ast.solve_from_image("{}.fits".format(self.data_filename[:-4]), force_image_upload=True, center_ra=ra_guess, center_dec=dec_guess, radius = radius, ra_dec_units=('degree', 'degree'))
        fits.writeto("{}.fits".format(self.data_filename[:-4]), self.invdat, self.solve1, overwrite = True)
        self.solve2 = ast.solve_from_image("{}.fits".format(self.data_filename[:-4]), force_image_upload=True, center_ra=ra_guess, center_dec=dec_guess, radius = radius, ra_dec_units=('degree', 'degree'))
        fits.writeto("{}.fits".format(self.data_filename[:-4]), self.invdat, self.solve2, overwrite = True)
        self.fits_solve2 = fits.open("{}.fits".format(self.data_filename[:-4]))
    
    def transform(self, tfunc, use_inv = False):
        if use_inv == False:
            data_copy = self.comb_tif.copy()
            self.tdata = tfunc(data_copy)
            self.tran_image = fits.writeto("{}_tform.fits".format(self.data_filename[:-4]), self.tdata, self.solve2, overwrite=True)   
        elif use_inv == True:
            data_copy = self.invdat.copy()
            self.tdata = tfunc(data_copy)
            self.tran_image = fits.writeto("{}_tform.fits".format(self.data_filename[:-4]), self.tdata, self.solve2, overwrite=True)
        
    def septrac(self, sigma = 3.0, gain = 1.0):
        self.bkg = sep.Background(self.tdata)
        self.wcssolve = WCS(self.solve2)
        self.bkg_sub = self.tdata - self.bkg
        self.objects = sep.extract(self.bkg_sub, sigma, err=self.bkg.globalrms)
        self.objtable = Table(self.objects)
        self.objtable['ra_p'], self.objtable['dec_p'] = self.wcssolve.wcs_pix2world(self.objtable['x'], self.objtable['y'])
        self.objtable['pet_mag'] = 25 - 2.5*np.log10(self.objtable['flux'])
        
    def gaia_call(self):
        mid_coord = self.wcssolve.pixel_to_world(round(np.shape(self.tdata)[0]/2), round(np.shape(self.tdata)[1]/2))
        bottom_coord = self.wcssolve.pixel_to_world(0, 0)
        top_coord = self.wcssolve.pixel_to_world(np.shape(self.tdata)[0], np.shape(self.tdata)[1])
        full_coord_ra = np.abs(top_coord.ra.deg-bottom_coord.ra.deg)
        full_coord_dec = np.abs(top_coord.dec.deg-bottom_coord.dec.deg)
        job = Gaia.launch_job("SELECT TOP 500000 "
                        "source_id,ra,dec,parallax,parallax_error,pm,pmra,pmra_error,pmdec,pmdec_error,phot_g_mean_mag,"
                        "phot_bp_mean_mag,phot_bp_mean_flux_over_error,phot_rp_mean_mag,phot_rp_mean_flux_over_error,bp_rp,phot_variable_flag, classprob_dsc_combmod_galaxy"
                        " from gaiadr3.gaia_source"
                        " WHERE CONTAINS(POINT('ICRS',ra,dec),BOX('ICRS',{0},{1},{2},{3}))=1  AND  ((phot_bp_mean_mag + 1.1*bp_rp) <= 19.0)".format(mid_coord.ra.deg, mid_coord.dec.deg, full_coord_ra+0.3, full_coord_dec+0.3))
        self.gaia_res = job.get_results()
        
    def match_cat(self):
        coords_gaia = SkyCoord(self.gaia_res['ra'], self.gaia_res['dec'], frame = 'icrs', unit = 'deg')
        coords_plate = SkyCoord(self.objtable['ra_p'], self.objtable['dec_p'], frame = 'icrs', unit = 'deg')
        index, diff2, diff3 = coords_plate.match_to_catalog_sky(coords_gaia)
        self.match_cat = hstack([self.gaia_res[index], self.objtable])
        self.match_cat['ang_dist'] = np.abs(np.sqrt(self.match_cat['ra']**2+self.match_cat['dec']**2)-np.sqrt(self.match_cat['ra_p']**2+self.match_cat['dec_p']**2)) #i dont this makes any sense, distance formula
        self.match_cat['pg'] = self.match_cat['phot_bp_mean_mag']+0.9*self.match_cat['bp_rp']
        self.match_cat['dec_res'] = (self.match_cat['dec'] -  self.match_cat['dec_p'])*3600
        self.match_cat['ra_res'] = (self.match_cat['ra'] -  self.match_cat['ra_p'])*3600*0.9114