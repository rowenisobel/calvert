import cv2
import sep
from astroquery.astrometry_net import AstrometryNet
import numpy as np
import math
from astropy.io import fits
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy import coordinates as coords
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astropy.table import Table, hstack
from astropy.wcs import WCS
import random
from scipy.optimize import curve_fit
from astroquery.sdss import SDSS

#example run:
#rr = plate_pipeline('/Users/irescapa/Downloads/R3170_det_pp.tif')
#rr.astrometry(API_key, 257.5000, 43.7500, 10)
#rr.septrac(sigma = 3)
#rr.gaia_call()
#rr.match_cat()
#rr.transform()
#rr.galc_sep(sigma = 3)
#rr.galc_mat()

__all__ = ['plate_pipeline']

class plate_pipeline():
    
    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.tif = cv2.imread(data_filename)
        self.tif_dat = self.tif.astype(np.float32)
        self.b, self.g, sself.r = cv2.split(self.tif_dat)
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
        
    def septrac(self, sigma = 3.0, gain = 1.0):
        self.bkg = sep.Background(self.invdat)
        self.wcssolve = WCS(self.solve2)
        self.bkg_sub = self.invdat - self.bkg
        self.objects = sep.extract(self.bkg_sub, sigma, err=self.bkg.globalrms)
        self.objtable = Table(self.objects)
        coord_p = self.wcssolve.pixel_to_world(self.objtable['x'], self.objtable['y'])
        self.objtable['ra_p'], self.objtable['dec_p'] = coord_p.ra.deg, coord_p.dec.deg
        self.objtable['pet_mag'] = 25 - 2.5*np.log10(self.objtable['flux'])
        
    def gaia_call(self):
        mid_coord = self.wcssolve.pixel_to_world(round(np.shape(self.invdat)[0]/2), round(np.shape(self.invdat)[1]/2))
        bottom_coord = self.wcssolve.pixel_to_world(0, 0)
        top_coord = self.wcssolve.pixel_to_world(np.shape(self.invdat)[0], np.shape(self.invdat)[1])
        full_coord_ra = np.abs(top_coord.ra.deg-bottom_coord.ra.deg)
        full_coord_dec = np.abs(top_coord.dec.deg-bottom_coord.dec.deg)
        job = Gaia.launch_job("SELECT TOP 500000 "
                        "source_id,ra,dec,parallax,parallax_error,pm,pmra,pmra_error,pmdec,pmdec_error,phot_g_mean_mag,"
                        "phot_bp_mean_mag,phot_bp_mean_flux_over_error,phot_rp_mean_mag,phot_rp_mean_flux_over_error,bp_rp,phot_variable_flag, classprob_dsc_combmod_galaxy"
                        " from gaiadr3.gaia_source"
                        " WHERE CONTAINS(POINT('ICRS',ra,dec),BOX('ICRS',{0},{1},{2},{3}))=1  AND  ((phot_bp_mean_mag + 1.1*bp_rp) <= 19.0)".format(mid_coord.ra.deg, mid_coord.dec.deg, full_coord_ra+0.3, full_coord_dec+0.3))
        self.gaia_res = job.get_results()
        #gmasel = astropy.coordinates.match_coordinates_sky(gaia_res, gaia_res, nthneighbor=1)
        
    def match_cat(self):
        self.coords_gaia = SkyCoord(self.gaia_res['ra'], self.gaia_res['dec'], frame = 'icrs', unit = 'deg')
        self.coords_plate = SkyCoord(self.objtable['ra_p'], self.objtable['dec_p'], frame = 'icrs', unit = 'deg')
        index, diff2, diff3 = self.coords_plate.match_to_catalog_sky(self.coords_gaia)
        self.match_cat = hstack([self.gaia_res[index], self.objtable])
        self.match_cat = self.match_cat[np.unique(self.match_cat['source_id'], return_index = True)[1]]   #idk how this chooses the values, like how do we know which one is the right match?
        self.match_cat['ang_dist'] = np.abs(np.sqrt(self.match_cat['ra']**2+self.match_cat['dec']**2)-np.sqrt(self.match_cat['ra_p']**2+self.match_cat['dec_p']**2)) #i dont know if this makes any sense, distance formula
        self.match_cat['pg'] = self.match_cat['phot_bp_mean_mag']+0.9*self.match_cat['bp_rp']
        self.match_cat['dec_res'] = (self.match_cat['dec'] -  self.match_cat['dec_p'])*3600
        self.match_cat['ra_res'] = (self.match_cat['ra'] -  self.match_cat['ra_p'])*3600*0.9114
        
    def transform(self):
        tabu = self.match_cat
        grama = 15
        smalm = 15
        gal_choi = []
        while len(gal_choi) == 0:
            grama += 1
            smalm -= 1
            mag_up = tabu[tabu['pg'] < grama]
            mag_mid = mag_up[mag_up['pg'] > smalm]
            gal_choi = mag_mid[mag_mid['classprob_dsc_combmod_galaxy'] == 1]

        hdu1 = (self.fits_solve2)
        imdata = self.comb_tif
        im_raa = np.arange(np.shape(self.invdat)[0])
        im_deca = np.arange(np.shape(self.invdat)[1])
        im_grra, im_grdec = np.meshgrid(im_raa, im_deca)
        im_vals = self.wcssolve.pixel_to_world(im_grra, im_grdec)
        
        def log_x(x, a,b,c,d):
            x = (a/np.log10(c*x-b)+d)**2
            return x

        galies = []
        for galy in range(0,len(gal_choi)):
            pos = coords.SkyCoord(gal_choi['ra'][galy], gal_choi['dec'][galy], frame='icrs', unit=u.deg)
            image = SDSS.get_images(matches=Table(SDSS.query_region(pos)), band='g')
            sdat = image[0][0].data
            ra_pix = np.arange(image[0][0].header['NAXIS1'])
            dec_pix = np.arange(image[0][0].header['NAXIS2'])
            gridra, griddec = np.meshgrid(ra_pix, dec_pix)
            sdss_val = WCS(image[0][0].header).pixel_to_world(gridra, griddec)

            inn, di2, di3 = sdss_val.flatten().match_to_catalog_sky(im_vals.flatten())

            dat1 = imdata.flatten()[inn]
            gg1 = sdat.flatten()

            popts = []
            sortd = np.sort([dat1,gg1], 1)
            split = 2
            while True:
                try:
                    wanted = []
                    spl_so = np.array_split(sortd, split, 1)
                    for arr in range(0,split):
                        migval = np.mean(spl_so[arr][0])
                        s_vau = np.mean(spl_so[arr][1])
                        wanted.append([migval+0.0000001*arr, s_vau])
                    wanted.append([np.min(dat1), np.max(gg1)+50])
                    wanted.append([np.max(dat1), np.min(gg1)-10])
                    wanted = np.array(wanted)
                    shap = np.shape(wanted)
                    wanted = np.sort(np.reshape(wanted, (shap)), axis = 0)
                    popt, pcov = curve_fit(log_x, (wanted[:,0]), (wanted[:,1][::-1]), maxfev = 5000)
                    popts.append([split, popt, np.nansum(np.nansum(np.abs(imdata-log_x(imdata, *popt))**2))])
                except:
                    popts = np.array(popts, dtype = object)
                    break
                else:
                    split += 25
            popts = np.array(popts, dtype = object)
            if len(popts) == 0:
                pass
            else:
                rig = np.where(np.min((popts[:,2])) == (popts[:,2]))
                galies.append(popts[rig])
        galies = np.array(galies)
        finr = np.where(np.min(galies[:,0,2]) == (galies[:,0,2]))
        self.gcdata = log_x(imdata, *galies[finr[0][0]][0][1])
        fits.writeto("{}_galcal.fits".format(self.data_filename[:-4]), self.gcdata, self.solve2, overwrite = True)
        self.fits_galcal = fits.open("{}_galcal.fits".format(self.data_filename[:-4]))

    def galc_sep(self, sigma = 3.0, gain = 1.0):
        self.bkg = sep.Background(self.gcdata)
        self.bkg_sub = self.gcdata - self.bkg
        self.objects = sep.extract(self.bkg_sub, sigma, err=self.bkg.globalrms)
        self.objtable = Table(self.objects)
        coord_p = self.wcssolve.pixel_to_world(self.objtable['x'], self.objtable['y'])
        self.objtable['ra_p'], self.objtable['dec_p'] = coord_p.ra.deg, coord_p.dec.deg
        self.objtable['pet_mag'] = 25 - 2.5*np.log10(self.objtable['flux'])
        
    def galc_mat(self):
        self.coords_plate = SkyCoord(self.objtable['ra_p'], self.objtable['dec_p'], frame = 'icrs', unit = 'deg')
        index, diff2, diff3 = self.coords_plate.match_to_catalog_sky(self.coords_gaia)
        self.gc_match = hstack([self.gaia_res[index], self.objtable])
        self.gc_match = self.gc_match[np.unique(self.gc_match['source_id'], return_index = True)[1]]   #idk how this chooses, or if all the unique values are the same???
        self.gc_match['ang_dist'] = np.abs(np.sqrt(self.gc_match['ra']**2+self.gc_match['dec']**2)-np.sqrt(self.gc_match['ra_p']**2+self.gc_match['dec_p']**2)) #i dont know if this makes any sense, distance formula
        self.gc_match['pg'] = self.gc_match['phot_bp_mean_mag']+0.9*self.gc_match['bp_rp']
        self.gc_match['dec_res'] = (self.gc_match['dec'] -  self.gc_match['dec_p'])*3600
        self.gc_match['ra_res'] = (self.gc_match['ra'] -  self.gc_match['ra_p'])*3600*0.9114
        
