from astropy.io import fits
import numpy as np
import math

__all__ = ['File']

class File:
    
    def __init__(self, dirtofits):
        '''
        stores the data and header of the FITS file at 
        the specified directory, along with matrix and 
        polynomial parameters from header
        '''
        
        # FITS data/header
        self.dirtofits = dirtofits
        self.fitsfile = fits.open(dirtofits)
        self.fitsdata = self.fitsfile[0].data
        self.fitsheader = self.fitsfile[0].header
        
        # second run reference pixel values
        # x and y of reference pixel
        self.pixx = float(self.fitsheader['CRPIX1'])
        self.pixy = float(self.fitsheader['CRPIX2'])
        # ra and dec of reference pixel
        self.ra = float(self.fitsheader['CRVAL1'])
        self.dec = float(self.fitsheader['CRVAL2'])
        # ref poles
        self.lonpole = float(self.fitsheader['LONPOLE'])
        self.latpole = float(self.fitsheader['LATPOLE'])
        
        # first run reference pixel values
        # x and y of reference pixel
        self.r1pixx = float(self.fitsheader['_RPIX1'])
        self.r1pixy = float(self.fitsheader['_RPIX2'])
        # ra and dec of reference pixel
        self.r1ra = float(self.fitsheader['_RVAL1'])
        self.r1dec = float(self.fitsheader['_RVAL2'])
        # ref poles
        self.r1lonpole = float(self.fitsheader['_ONPOLE'])
        self.r1latpole = float(self.fitsheader['_ATPOLE'])
        
        # second run matrix/polynomial parameters
        # CD matrix
        self.CD1_1 = float(self.fitsheader['CD1_1'])
        self.CD1_2 = float(self.fitsheader['CD1_2'])
        self.CD2_1 = float(self.fitsheader['CD2_1'])
        self.CD2_2 = float(self.fitsheader['CD2_2'])
        
        # first run matrix/polynomial parameters
        # CD matrix
        self.r1CD1_1 = float(self.fitsheader['_D1_1'])
        self.r1CD1_2 = float(self.fitsheader['_D1_2'])
        self.r1CD2_1 = float(self.fitsheader['_D2_1'])
        self.r1CD2_2 = float(self.fitsheader['_D2_2'])



    def pixtocoord(self,imagex,imagey):
        # A polynomial
        A_0_0 = float(self.fitsheader['A_0_0'])
        A_0_1 = float(self.fitsheader['A_0_1'])
        A_0_2 = float(self.fitsheader['A_0_2'])
        A_1_0 = float(self.fitsheader['A_1_0'])
        A_1_1 = float(self.fitsheader['A_1_1'])
        A_2_0 = float(self.fitsheader['A_2_0'])
        # B polynomial
        B_0_0 = float(self.fitsheader['B_0_0'])
        B_0_1 = float(self.fitsheader['B_0_1'])
        B_0_2 = float(self.fitsheader['B_0_2'])
        B_1_0 = float(self.fitsheader['B_1_0'])
        B_1_1 = float(self.fitsheader['B_1_1'])
        B_2_0 = float(self.fitsheader['B_2_0'])
        
        # offsetx and offsety will be used in the relative pixel coordinates -> intermediate coordinates calculation
        print("FOR FILE AT %s\npixel on image you are interested in: %s,%s" %(self.dirtofits,imagex,imagey))
        print("where u,v are relative pixel coordinates to FITS reference point %s,%s" %(self.pixx,self.pixy))
        offsetx = imagex-self.pixx
        offsety = imagey-self.pixy
        print("(u,v) = (%s,%s)" %(offsetx,offsety))
        
        # prints CD matrix, A polynomial, B polynomial directly from FITS header (where A and B polynomial must be 2nd order, but relatively easy to implement other orders)
        print("\nFITS HEADER MATRICES/POLYNOMIALS\nCD matrix:\t|%s\t%s|\n\t\t|%s\t%s|" %(self.CD1_1,self.CD1_2,self.CD2_1,self.CD2_2))
        print("A polynomial:\t%s + %s*u^2 + %s*v^2 + %s*uv + %s*u + %s*v" %(A_0_0,A_2_0,A_0_2,A_1_1,A_1_0,A_0_1))
        print("B polynomial:\t%s + %s*u^2 + %s*v^2 + %s*uv + %s*u + %s*v\n" %(B_0_0,B_2_0,B_0_2,B_1_1,B_1_0,B_0_1))
        
        # calculates the intermediate world coordinates using method described in paper III of FITS headers
        # CD matrix accounts for any skew, rotation, and scaling
        # A and B polynomials correct for distortion using a simple image polynomial
        x_nt = self.CD1_1*offsetx+self.CD1_2*offsety
        y_nt = self.CD2_1*offsetx+self.CD2_2*offsety
        
        x = (self.CD1_1 * (offsetx + (A_0_0) + (A_2_0*offsetx*offsetx) + (A_0_2*offsety*offsety) + (A_1_1*offsetx*offsety) + (A_1_0*offsetx) + (A_0_1*offsety))) + (self.CD1_2 * (offsety + (B_0_0) + (B_2_0*offsetx*offsetx) + (B_0_2*offsety*offsety) + (B_1_1*offsetx*offsety) + (B_1_0*offsetx) + (B_0_1*offsety)))
        y = (self.CD2_1 * (offsetx + (A_0_0) + (A_2_0*offsetx*offsetx) + (A_0_2*offsety*offsety) + (A_1_1*offsetx*offsety) + (A_1_0*offsetx) + (A_0_1*offsety))) + (self.CD2_2 * (offsety + (B_0_0) + (B_2_0*offsetx*offsetx) + (B_0_2*offsety*offsety) + (B_1_1*offsetx*offsety) + (B_1_0*offsetx) + (B_0_1*offsety)))
        
        print("INTERMEDIATE WORLD COORDINATES\nno tweak:        (x,y) = (%s,%s)" %(x_nt,y_nt))
        print("2nd order tweak: (x,y) = (%s,%s)" %(x,y))
        
        # calculates phi and theta, which are the native spherical coordinates (latitute and longitude), using a spherical projection of the intermediate coordinates
        if (np.angle(complex(-y,x),True)<0):
            phi = np.angle(complex(-y,x),True)+360.
        else:
            phi = np.angle(complex(-y,x),True)
                
        theta = math.degrees(math.atan(180./(math.pi*(math.sqrt(x*x+y*y)))))
        
        if (np.angle(complex(-y_nt,x_nt),True)<0):
            phi_nt = np.angle(complex(-y_nt,x_nt),True)+360.
        else:
            phi_nt = np.angle(complex(-y_nt,x_nt),True)
                
        theta_nt = math.degrees(math.atan(180./(math.pi*(math.sqrt(x_nt*x_nt+y_nt*y_nt)))))
        
        print("\nNATIVE SPHERICAL COORDINATES\nno tweak:        (phi,theta) = (%s,%s)" %(phi_nt,theta_nt))
        print("2nd order tweak: (phi,theta) = (%s,%s)" %(phi,theta))
        
        
        
        # calculates ra and dec, which are the celestial spherical coordinates, using a spherical coordinate rotation of the native spherical coordinates
        ang1forra = math.sin(math.radians(theta))*math.cos(math.radians(self.dec))-math.cos(math.radians(theta))*math.sin(math.radians(self.dec))*math.cos(math.radians(phi-self.lonpole))
        ang2forra = -1.*math.cos(math.radians(theta))*math.sin(math.radians(phi-self.lonpole))
        if ((np.angle(complex(ang1forra,ang2forra),True)+self.ra)<0):
            calcra = self.ra+np.angle(complex(ang1forra,ang2forra),True)+360.
        else:
            calcra = self.ra+np.angle(complex(ang1forra,ang2forra),True)
        
        calcdec = math.degrees(math.asin(math.sin(math.radians(theta))*math.sin(math.radians(self.dec))+math.cos(math.radians(theta))*math.cos(math.radians(self.dec))*math.cos(math.radians(phi-self.lonpole))))

        ang1forra_nt = math.sin(math.radians(theta_nt))*math.cos(math.radians(self.dec))-math.cos(math.radians(theta_nt))*math.sin(math.radians(self.dec))*math.cos(math.radians(phi_nt-self.lonpole))
        ang2forra_nt = -1.*math.cos(math.radians(theta_nt))*math.sin(math.radians(phi_nt-self.lonpole))
        if ((np.angle(complex(ang1forra_nt,ang2forra_nt),True)+self.ra)<0):
            calcra_nt = self.ra+np.angle(complex(ang1forra_nt,ang2forra_nt),True)+360.
        else:
            calcra_nt = self.ra+np.angle(complex(ang1forra_nt,ang2forra_nt),True)
        
        calcdec_nt = math.degrees(math.asin(math.sin(math.radians(theta_nt))*math.sin(math.radians(self.dec))+math.cos(math.radians(theta_nt))*math.cos(math.radians(self.dec))*math.cos(math.radians(phi_nt-self.lonpole))))

        
        print("\nCELESTIAL SPHERICAL COORDINATES\n(ra,dec) = (%s,%s) at reference point (%s,%s)" %(self.ra,self.dec,self.pixx,self.pixy))
        print("no tweak:        (ra,dec) = (%s,%s) at pixel (%s,%s)" %(calcra_nt,calcdec_nt,imagex,imagey))
        print("2nd order tweak: (ra,dec) = (%s,%s) at pixel (%s,%s)" %(calcra,calcdec,imagex,imagey))
        
        return calcra,calcdec

    
    
    def r1pixtocoord(self,imagex,imagey):
        if (self.fitsheader['comment'][33]!="Original key: \"B_0_0\""):
            print("first run FITS information stored in unusual location! exiting function")
            return
        # A polynomial
        r1A_0_0 = float(self.fitsheader['__0_0'])
        r1A_0_1 = float(self.fitsheader['__0_1'])
        r1A_0_2 = float(self.fitsheader['__0_2'])
        r1A_1_0 = float(self.fitsheader['__1_0'])
        r1A_1_1 = float(self.fitsheader['__1_1'])
        r1A_2_0 = float(self.fitsheader['__2_0'])
        # B polynomial
        r1B_0_0 = float(self.fitsheader['comment'][34][3:])
        r1B_0_1 = float(self.fitsheader['comment'][37][3:])
        r1B_0_2 = float(self.fitsheader['comment'][40][3:])
        r1B_1_0 = float(self.fitsheader['comment'][43][3:])
        r1B_1_1 = float(self.fitsheader['comment'][46][3:])
        r1B_2_0 = float(self.fitsheader['comment'][49][3:])
        
        # offsetx and offsety will be used in the relative pixel coordinates -> intermediate coordinates calculation
        print("FOR FILE AT %s\npixel on image you are interested in: %s,%s" %(self.dirtofits,imagex,imagey))
        print("where u,v are relative pixel coordinates to FITS reference point %s,%s" %(self.r1pixx,self.r1pixy))
        offsetx = imagex-self.r1pixx
        offsety = imagey-self.r1pixy
        print("(u,v) = (%s,%s)" %(offsetx,offsety))
        
        # prints CD matrix, A polynomial, B polynomial directly from FITS header (where A and B polynomial must be 2nd order, but relatively easy to implement other orders)
        print("\nFITS HEADER MATRICES/POLYNOMIALS\nCD matrix:\t|%s\t%s|\n\t\t|%s\t%s|" %(self.r1CD1_1,self.r1CD1_2,self.r1CD2_1,self.r1CD2_2))
        print("A polynomial:\t%s + %s*u^2 + %s*v^2 + %s*uv + %s*u + %s*v" %(r1A_0_0,r1A_2_0,r1A_0_2,r1A_1_1,r1A_1_0,r1A_0_1))
        print("B polynomial:\t%s + %s*u^2 + %s*v^2 + %s*uv + %s*u + %s*v\n" %(r1B_0_0,r1B_2_0,r1B_0_2,r1B_1_1,r1B_1_0,r1B_0_1))
        
        # calculates the intermediate world coordinates using method described in paper III of FITS headers
        # CD matrix accounts for any skew, rotation, and scaling
        # A and B polynomials correct for distortion using a simple image polynomial
        x_nt = self.r1CD1_1*offsetx+self.r1CD1_2*offsety
        y_nt = self.r1CD2_1*offsetx+self.r1CD2_2*offsety
        
        x = (self.r1CD1_1 * (offsetx + (r1A_0_0) + (r1A_2_0*offsetx*offsetx) + (r1A_0_2*offsety*offsety) + (r1A_1_1*offsetx*offsety) + (r1A_1_0*offsetx) + (r1A_0_1*offsety))) + (self.r1CD1_2 * (offsety + (r1B_0_0) + (r1B_2_0*offsetx*offsetx) + (r1B_0_2*offsety*offsety) + (r1B_1_1*offsetx*offsety) + (r1B_1_0*offsetx) + (r1B_0_1*offsety)))
        y = (self.r1CD2_1 * (offsetx + (r1A_0_0) + (r1A_2_0*offsetx*offsetx) + (r1A_0_2*offsety*offsety) + (r1A_1_1*offsetx*offsety) + (r1A_1_0*offsetx) + (r1A_0_1*offsety))) + (self.r1CD2_2 * (offsety + (r1B_0_0) + (r1B_2_0*offsetx*offsetx) + (r1B_0_2*offsety*offsety) + (r1B_1_1*offsetx*offsety) + (r1B_1_0*offsetx) + (r1B_0_1*offsety)))
        
        print("INTERMEDIATE WORLD COORDINATES\nno tweak:        (x,y) = (%s,%s)" %(x_nt,y_nt))
        print("2nd order tweak: (x,y) = (%s,%s)" %(x,y))
        
        # calculates phi and theta, which are the native spherical coordinates (latitute and longitude), using a spherical projection of the intermediate coordinates
        if (np.angle(complex(-y,x),True)<0):
            phi = np.angle(complex(-y,x),True)+360.
        else:
            phi = np.angle(complex(-y,x),True)
                
        theta = math.degrees(math.atan(180./(math.pi*(math.sqrt(x*x+y*y)))))
        
        if (np.angle(complex(-y_nt,x_nt),True)<0):
            phi_nt = np.angle(complex(-y_nt,x_nt),True)+360.
        else:
            phi_nt = np.angle(complex(-y_nt,x_nt),True)
                
        theta_nt = math.degrees(math.atan(180./(math.pi*(math.sqrt(x_nt*x_nt+y_nt*y_nt)))))
        
        print("\nNATIVE SPHERICAL COORDINATES\nno tweak:        (phi,theta) = (%s,%s)" %(phi_nt,theta_nt))
        print("2nd order tweak: (phi,theta) = (%s,%s)" %(phi,theta))
        
        
        
        # calculates ra and dec, which are the celestial spherical coordinates, using a spherical coordinate rotation of the native spherical coordinates
        ang1forra = math.sin(math.radians(theta))*math.cos(math.radians(self.r1dec))-math.cos(math.radians(theta))*math.sin(math.radians(self.r1dec))*math.cos(math.radians(phi-self.r1lonpole))
        ang2forra = -1.*math.cos(math.radians(theta))*math.sin(math.radians(phi-self.r1lonpole))
        if ((np.angle(complex(ang1forra,ang2forra),True)+self.r1ra)<0):
            calcra = self.r1ra+np.angle(complex(ang1forra,ang2forra),True)+360.
        else:
            calcra = self.r1ra+np.angle(complex(ang1forra,ang2forra),True)
        
        calcdec = math.degrees(math.asin(math.sin(math.radians(theta))*math.sin(math.radians(self.r1dec))+math.cos(math.radians(theta))*math.cos(math.radians(self.r1dec))*math.cos(math.radians(phi-self.r1lonpole))))

        ang1forra_nt = math.sin(math.radians(theta_nt))*math.cos(math.radians(self.r1dec))-math.cos(math.radians(theta_nt))*math.sin(math.radians(self.r1dec))*math.cos(math.radians(phi_nt-self.r1lonpole))
        ang2forra_nt = -1.*math.cos(math.radians(theta_nt))*math.sin(math.radians(phi_nt-self.r1lonpole))
        if ((np.angle(complex(ang1forra_nt,ang2forra_nt),True)+self.r1ra)<0):
            calcra_nt = self.r1ra+np.angle(complex(ang1forra_nt,ang2forra_nt),True)+360.
        else:
            calcra_nt = self.r1ra+np.angle(complex(ang1forra_nt,ang2forra_nt),True)
        
        calcdec_nt = math.degrees(math.asin(math.sin(math.radians(theta_nt))*math.sin(math.radians(self.r1dec))+math.cos(math.radians(theta_nt))*math.cos(math.radians(self.r1dec))*math.cos(math.radians(phi_nt-self.r1lonpole))))

        
        print("\nCELESTIAL SPHERICAL COORDINATES\n(ra,dec) = (%s,%s) at reference point (%s,%s)" %(self.r1ra,self.r1dec,self.r1pixx,self.r1pixy))
        print("no tweak:        (ra,dec) = (%s,%s) at pixel (%s,%s)" %(calcra_nt,calcdec_nt,imagex,imagey))
        print("2nd order tweak: (ra,dec) = (%s,%s) at pixel (%s,%s)" %(calcra,calcdec,imagex,imagey))
        
        return calcra,calcdec
    


    def coordtopix(self,ira,idec):
        # AP polynomial
        AP_0_0 = float(self.fitsheader['AP_0_0'])
        AP_0_1 = float(self.fitsheader['AP_0_1'])
        AP_0_2 = float(self.fitsheader['AP_0_2'])
        AP_1_0 = float(self.fitsheader['AP_1_0'])
        AP_1_1 = float(self.fitsheader['AP_1_1'])
        AP_2_0 = float(self.fitsheader['AP_2_0'])
        # BP polynomial
        BP_0_0 = float(self.fitsheader['BP_0_0'])
        BP_0_1 = float(self.fitsheader['BP_0_1'])
        BP_0_2 = float(self.fitsheader['BP_0_2'])
        BP_1_0 = float(self.fitsheader['BP_1_0'])
        BP_1_1 = float(self.fitsheader['BP_1_1'])
        BP_2_0 = float(self.fitsheader['BP_2_0'])
        
        # celestial coordinates (ra,dec) to native spherical coordinates (phi,theta)
        print("celestial coordinates (ra,dec) = (%s,%s)" %(ira,idec))
        ang1forphi = math.sin(math.radians(idec))*math.cos(math.radians(self.dec))-math.cos(math.radians(idec))*math.sin(math.radians(self.dec))*math.cos(math.radians(ira-self.ra))
        ang2forphi = -1.*math.cos(math.radians(idec))*math.sin(math.radians(ira-self.ra))
        
        if ((np.angle(complex(ang1forphi,ang2forphi),True)+self.lonpole)<0):
            phi = self.lonpole+np.angle(complex(ang1forphi,ang2forphi),True)+360.
        else:
            phi = self.lonpole+np.angle(complex(ang1forphi,ang2forphi),True)
        
        theta = math.degrees(math.asin(math.sin(math.radians(idec))*math.sin(math.radians(self.dec))+math.cos(math.radians(idec))*math.cos(math.radians(self.dec))*math.cos(math.radians(ira-self.ra))))
        print("native spherical coordinates (phi,theta) = (%s,%s)" %(phi,theta))
        
        # native spherical coordinates (theta,phi) to intermediate world coordinates (x,y)
        rtheta = (180./math.pi)*(1/(math.tan(math.radians(theta))))
        x = rtheta*math.sin(math.radians(phi))
        y = -rtheta*math.cos(math.radians(phi))
        print("intermediate world coordinates (x,y) = (%s,%s)" %(x,y))
        
        # intermediate world coordinates (x,y) to corrected pixel coordinates (U,V)
        inv_CD1_1 = 1./(self.CD1_1*self.CD2_2-self.CD1_2*self.CD2_1)*self.CD2_2
        inv_CD1_2 = -1./(self.CD1_1*self.CD2_2-self.CD1_2*self.CD2_1)*self.CD1_2
        inv_CD2_1 = -1./(self.CD1_1*self.CD2_2-self.CD1_2*self.CD2_1)*self.CD2_1
        inv_CD2_2 = 1./(self.CD1_1*self.CD2_2-self.CD1_2*self.CD2_1)*self.CD1_1

        U = inv_CD1_1*x+inv_CD1_2*y
        V = inv_CD2_1*x+inv_CD2_2*y
        print("corrected pixel coordinates (U,V) = (%s,%s)" %(U,V))
        
        # corrected pixel coordinates (U,V) to realtive pixel coordinates (u,v)
        u = U + (AP_0_0) + (AP_2_0*U*U) + (AP_0_2*V*V) + (AP_1_1*U*V) + (AP_1_0*U) + (AP_0_1*V)
        v = V + (BP_0_0) + (BP_2_0*U*U) + (BP_0_2*V*V) + (BP_1_1*U*V) + (BP_1_0*U) + (BP_0_1*V)
        print("relative pixel coordinates (u,v) = (%s,%s)" %(u,v))
        
        # relative pixel coordinates (u,v) to actual pixel coordinates
        imagex = u+self.pixx
        imagey = v+self.pixy
        print("pixel on original image: (%s,%s)" %(imagex,imagey))
        
        return imagex,imagey
        
        
        
    def r1coordtopix(self,ira,idec):
        if (self.fitsheader['comment'][33]!="Original key: \"B_0_0\""):
            print("first run FITS information stored in unusual location! exiting function")
            return
        # AP polynomial
        r1AP_0_0 = float(self.fitsheader['_P_0_0'])
        r1AP_0_1 = float(self.fitsheader['_P_0_1'])
        r1AP_0_2 = float(self.fitsheader['_P_0_2'])
        r1AP_1_0 = float(self.fitsheader['_P_1_0'])
        r1AP_1_1 = float(self.fitsheader['_P_1_1'])
        r1AP_2_0 = float(self.fitsheader['_P_2_0'])
        # BP polynomial
        r1BP_0_0 = float(self.fitsheader['comment'][62][3:])
        r1BP_0_1 = float(self.fitsheader['comment'][65][3:])
        r1BP_0_2 = float(self.fitsheader['comment'][68][3:])
        r1BP_1_0 = float(self.fitsheader['comment'][71][3:])
        r1BP_1_1 = float(self.fitsheader['comment'][74][3:])
        r1BP_2_0 = float(self.fitsheader['comment'][77][3:])
        
        # celestial coordinates (ra,dec) to native spherical coordinates (phi,theta)
        print("celestial coordinates (ra,dec) = (%s,%s)" %(ira,idec))
        ang1forphi = math.sin(math.radians(idec))*math.cos(math.radians(self.r1dec))-math.cos(math.radians(idec))*math.sin(math.radians(self.r1dec))*math.cos(math.radians(ira-self.r1ra))
        ang2forphi = -1.*math.cos(math.radians(idec))*math.sin(math.radians(ira-self.r1ra))
        
        if ((np.angle(complex(ang1forphi,ang2forphi),True)+self.r1lonpole)<0):
            phi = self.r1lonpole+np.angle(complex(ang1forphi,ang2forphi),True)+360.
        else:
            phi = self.r1lonpole+np.angle(complex(ang1forphi,ang2forphi),True)
        
        theta = math.degrees(math.asin(math.sin(math.radians(idec))*math.sin(math.radians(self.r1dec))+math.cos(math.radians(idec))*math.cos(math.radians(self.r1dec))*math.cos(math.radians(ira-self.r1ra))))
        print("native spherical coordinates (phi,theta) = (%s,%s)" %(phi,theta))
        
        # native spherical coordinates (theta,phi) to intermediate world coordinates (x,y)
        rtheta = (180./math.pi)*(1/(math.tan(math.radians(theta))))
        x = rtheta*math.sin(math.radians(phi))
        y = -rtheta*math.cos(math.radians(phi))
        print("intermediate world coordinates (x,y) = (%s,%s)" %(x,y))
        
        # intermediate world coordinates (x,y) to corrected pixel coordinates (U,V)
        inv_CD1_1 = 1./(self.r1CD1_1*self.r1CD2_2-self.r1CD1_2*self.r1CD2_1)*self.r1CD2_2
        inv_CD1_2 = -1./(self.r1CD1_1*self.r1CD2_2-self.r1CD1_2*self.r1CD2_1)*self.r1CD1_2
        inv_CD2_1 = -1./(self.r1CD1_1*self.r1CD2_2-self.r1CD1_2*self.r1CD2_1)*self.r1CD2_1
        inv_CD2_2 = 1./(self.r1CD1_1*self.r1CD2_2-self.r1CD1_2*self.r1CD2_1)*self.r1CD1_1

        U = inv_CD1_1*x+inv_CD1_2*y
        V = inv_CD2_1*x+inv_CD2_2*y
        print("corrected pixel coordinates (U,V) = (%s,%s)" %(U,V))
        
        # corrected pixel coordinates (U,V) to realtive pixel coordinates (u,v)
        u = U + (r1AP_0_0) + (r1AP_2_0*U*U) + (r1AP_0_2*V*V) + (r1AP_1_1*U*V) + (r1AP_1_0*U) + (r1AP_0_1*V)
        v = V + (r1BP_0_0) + (r1BP_2_0*U*U) + (r1BP_0_2*V*V) + (r1BP_1_1*U*V) + (r1BP_1_0*U) + (r1BP_0_1*V)
        print("relative pixel coordinates (u,v) = (%s,%s)" %(u,v))
        
        # relative pixel coordinates (u,v) to actual pixel coordinates
        imagex = u+self.r1pixx
        imagey = v+self.r1pixy
        print("pixel on original image: (%s,%s)" %(imagex,imagey))
        
        return imagex,imagey

