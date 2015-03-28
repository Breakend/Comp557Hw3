# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:53:31 2015

@author: Fahim
"""

'''
Test the different components of the raytracer

IMPORTANT NOTE: To test whether a ray going through the center of the image reaches
the camera's pointTo, we need to make sure that the camera is close enough to
that point. Since the ray directions are discretized, there's some error in
the actual intersection point. If the camera is too far away this error will
get larger.
'''

import unittest
import numpy as np
import numpy.testing as nptest
from Scene import Scene
from HelperClasses import Camera, Render
import GeomTransform as GT

#---- Base Test Class
class RayCreationFromPixelBaseTests(object):
    '''
    Base class for all test cases. The derived classes override the default
    setUp method.
    
    Tests performed for create_ray(pixel) are:

    Test if the ray origin in correct
    
    Test if the ray is going towards the general direction of lookat 
    
    Test if the ray going through the center of the image passes through the
    lookat point in the scene.
    
    Test if the extreme rays satisfy the fov 
    
    Derived classes tests if all of the above hold for different camera configuration
    '''
 
    def test_ray_through_center_eyePoint(self):
        ''' Test if the ray origin is set correctly  '''
        render = self.scene.render
        camera = render.camera
        np.random.seed(3421)
        R = np.random.permutation(range(camera.imageHeight))
        C = np.random.permutation(range(camera.imageWidth))
        NPixels = 100
        Pixels = np.empty((NPixels, 2))
        Pixels[:,0] = C[:NPixels]
        Pixels[:,1] = R[:NPixels]
        
        for pixel in Pixels:
            ray = self.scene.create_ray(pixel[1], pixel[0])
            #print(ray)
            nptest.assert_array_equal(ray.eyePoint, camera.pointFrom)

    def test_ray_through_center_towards_neg_z_direction(self):
        ''' ray direction should be towards the lookat direction '''
        render = self.scene.render
        camera = render.camera
        np.random.seed(3421)
        R = np.random.permutation(range(camera.imageHeight))
        C = np.random.permutation(range(camera.imageWidth))
        NPixels = 100
        Pixels = np.empty((NPixels, 2))
        Pixels[:,0] = C[:NPixels]
        Pixels[:,1] = R[:NPixels]
        for pixel in Pixels:
            ray = self.scene.create_ray(pixel[1], pixel[0])
            dot_prod = np.dot(ray.viewDirection, render.camera.lookat)
            # test if rays in the lookat direction
            self.assertEqual(np.sign(dot_prod), 1.)                                      

    def test_ray_through_center_dist_to_camPointTo(self):
        ''' 
        Test if the point the camera is looking at is at the right distance
        from the ray origin. The ray going through the center of the image
        will pass through the point that the camera is looking at (i.e. Camera.pointTo).
        '''
        camera = self.scene.render.camera
        # find the center of the image
        halfW = int((camera.imageWidth - 1) / 2.)
        halfH = int((camera.imageHeight - 1) / 2.)
        #print(halfW, halfH)
        # compute the ray going through the center of the image
        ray = self.scene.create_ray(halfH, halfW)
        
        # compute the distance between camera's position and the scene point
        # it's looking at
        dist = np.linalg.norm(camera.pointTo - camera.pointFrom)
        #print(dist)
        
        # get the point that is at the same distance from the ray's origin
        point_at_dist = ray.getPoint(dist)
        #print(point_at_dist)        
        
        # the two points should be the same
        nptest.assert_array_almost_equal(point_at_dist, camera.pointTo, decimal=3)

    def test_angle_between_left_right_extreme_rays(self):
        '''
        Test if the angle between left-most and right-most rays are consistent 
        with the fov
        '''
        camera = self.scene.render.camera
        # find the middle row
        halfH = int((camera.imageHeight - 1) / 2.)
        
        # compute the ray going through the left edge of the image
        rayLeft = self.scene.create_ray(halfH, 0)
        
        # compute the ray going through the right edge of the image
        rayRight = self.scene.create_ray(halfH, camera.imageWidth)
        
        cos_theta = np.dot(GT.normalize(rayLeft.viewDirection), GT.normalize(rayRight.viewDirection))
        est_angle_deg = np.rad2deg(np.arccos(cos_theta))
        #print(est_angle_deg)
        # check fov
        nptest.assert_approx_equal(est_angle_deg, camera.aspect * camera.fov)
        
    def test_angle_between_top_bottom_extreme_rays(self):
        '''
        Test if the angle between top-most and bottom-most rays are consistent 
        with the fov
        '''
        camera = self.scene.render.camera
                
        # find the middle col
        halfW = int((camera.imageWidth - 1) / 2.)
        
        # compute the ray going through the top edge of the image
        rayTop = self.scene.create_ray(0, halfW)
        
        # compute the ray going through the bottom edge of the image
        rayBottom = self.scene.create_ray(camera.imageHeight, halfW)
        
        cos_theta = np.dot(GT.normalize(rayTop.viewDirection), GT.normalize(rayBottom.viewDirection))
        
        # check fov
        nptest.assert_approx_equal(cos_theta, np.cos(np.deg2rad(camera.fov)))
        
################# Derived classes with different camera configuration
class TestRayCreationFromPixelCamOnPosZ(RayCreationFromPixelBaseTests, unittest.TestCase):
    
    def setUp(self):
        '''
        Create a scene with a camera placed at [0, 0, 4] and looking at the
        world origin. 
        '''
        scene = Scene()
        self.scene = scene
  
        #Camera
        camera = Camera({'from':np.array([0.,0.,4.]), 
                         'to':np.array([0.,0.,0.]), 
                         'up':np.array([0.,1.,0.]), 
                         'fov':45, 
                         'width':3001, 'height':3001}) # choose very high dimension to avoid rounding error
        render = Render({'camera':camera})
        scene.render = render
 
        
class TestRayCreationFromPixelCamOnNegZ(RayCreationFromPixelBaseTests, unittest.TestCase):
    
    def setUp(self):
        '''
        Create a scene with a camera placed at [0, 0, 4] and looking at the
        world origin. 
        '''
        scene = Scene()
        self.scene = scene
  
        #Camera
        camera = Camera({'from':np.array([0.,0.,-4.]), 
                         'to':np.array([0.,0.,0.]), 
                         'up':np.array([0.,1.,0.]), 
                         'fov':45, 
                         'width':3001, 'height':3001}) # choose very high dimension to avoid rounding error
        render = Render({'camera':camera})
        scene.render = render
 
 
class TestRayCreationFromPixelCamOnPosZUpNegX(RayCreationFromPixelBaseTests, unittest.TestCase):
   
    def setUp(self):
        '''
        Create a scene with a camera placed at [0, 0, 4] and looking at the
        world origin. 
        '''
        scene = Scene()
        self.scene = scene
  
        #Camera
        camera = Camera({'from':np.array([0.,0.,4.]), 
                         'to':np.array([0.,0.,0.]), 
                         'up':np.array([-1.,0.,0.]), 
                         'fov':45, 
                         'width':3001, 'height':3001}) # choose very high dimension to avoid rounding error
        render = Render({'camera':camera})
        scene.render = render
 

class TestRayCreationFromPixelArbitraryCamPosOrientation(RayCreationFromPixelBaseTests, unittest.TestCase):
    
    def setUp(self):
        '''
        Create a scene with a camera placed at [0, 0, 4] and looking at the
        world origin. 
        '''
        scene = Scene()
        self.scene = scene
  
        #Camera
        camera = Camera({'from':np.array([1.,4.,-2.]), 
                         'to':np.array([1.,-2.,4]), 
                         'up':GT.normalize(np.array([-1.,1.,0.])), 
                         'fov':45, 
                         'width':10001, 'height':10001}) # choose very high dimension to avoid rounding error
        render = Render({'camera':camera})
        scene.render = render
 

                                  
def main(): # to make it easier to import this file and run the tests
    unittest.main()
    
if __name__ == '__main__':
    main()
    