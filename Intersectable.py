# consider all division as floating point division
from __future__ import division
from HelperClasses import Material
from Ray import Ray, IntersectionResult
import GeomTransform as GT
import numpy as np
import math

# use this for testing if a variable is close to 0,
# a variable x is close to 0
# if |x| < EPS_DISTANCE or in the code:
# np.fabs(x) < EPS_DISTANCE
EPS_DISTANCE = 1e-9


class Sphere:
  """

  Sphere class

  Properties that a sphere has are center, radius and material.
  The intersection method returns the result of ray-sphere intersection.

  """

  def __init__(self, params={}):
      # radius = 1.0, center = [0.0, 0.0, 0.0], material = Material()):
      self.material = params.get('material', Material())
      self.radius = float(params.get('radius', 1.0))
      self.center = np.array(params.get('center', [0., 0., 0.]))

  def intersect(self, ray):
      '''
      input: ray in the same coordinate system as the sphere
      output: IntersectionResult, contains the intersection point, normal,
              distance from the eye position and material (see Ray.py)
      Let, the ray be P = P0 + tv, where P0 = eye, v = ray direction
      We want to find t.
      sphere with center Pc and radius r:
      (P - Pc)^T (P - Pc) = r^2
      We need to consider the different situations that can arise with ray-sphere
      intersection.
      NOTE 1: If the eye position is inside the sphere it SHOULD return the
      ray-sphere intersection along the view direction. Because the point *is
      visible* to the eye as far as ray-sphere intersection is concerned.
      The color used for rendering that point will be considered during the
      lighting and shading stage. In general there's no reason why there can't
      be a viewer and light sources inside a sphere.
      NOTE 2: If the ray origin is on the surface of the sphere then the nearest
      intersection should be ignored otherwise we'll have problems where
      the surface point cannot 'see' the light because of self intersection.
      '''
      '''
      Implement intersection between the ray and the current object and
      return IntersectionResult variable (isect) which will store the
      intersection point, the normal at the intersection and material of the
      object at the intersection point.
      '''
      isect = IntersectionResult()  # by default isect corresponds to no intersection

      global EPS_DISTANCE  # use this for testing if a variable is close to 0
      # TODO ===== BEGIN SOLUTION HERE =====

      # IF THE RAY IS INSIDE THE SPHERE (i.e. eye < center + radius)
      # do stuff
      # ELSE IF The ray is on the sphere
      # OTHERWISE it's a normal ray intersection from the outside

      # Pn = (P0 - Pc) + tv
      # D = P0 - Pc
      # Pn^T Pn = r^2
      # (D + tv)^T (D+tv) = r^2
      # (D^2 - r^2) + 2Dv*t + t^2 * v^2 = r^2
      # E = D^2 - r^2
      # t = (- 2D*v +/- sqrt(2Dv^2 - 4v^2(E)))/2(v^2)

      dir_vec = ray.viewDirection
      p = ray.eyePoint
      c2 = np.dot(dir_vec, dir_vec)
      c1 = 2 * np.dot(dir_vec, p-self.center)
      c0 = np.dot(p-self.center, p-self.center) - (self.radius**2)
      delta = c1*c1 - 4*c2*c0
      if delta < -EPS_DISTANCE:
        return isect

      delta = np.fabs(delta)
      x1 = (-c1 - math.sqrt(delta)) / (2*c2)
      x2 = (-c1 + math.sqrt(delta)) / (2*c2)
      x = min(x1, x2)
      if x < -EPS_DISTANCE:
        if x1 > 0:
          x = x1
        if x2 > 0:
          x = x2

      if np.fabs(x) < EPS_DISTANCE:
        if x1 > EPS_DISTANCE:
          x = x1
        if x2 > EPS_DISTANCE:
          x = x2

      if (x < -EPS_DISTANCE) or np.fabs(x) < EPS_DISTANCE:
        # if (x > 0):
        return isect
      else:
        isect.t = x
        isect.material = self.material
        isect.p = ray.eyePoint + x * ray.viewDirection
        isect.n = GT.normalize(isect.p - self.center)

      # While at first was doing full quadratic solving. This was more computationally heavy
      # And thus am using http://www.lighthouse3d.com/tutorials/maths/ray-sphere-intersection/
      # as a base for this

      # vec_pc = self.center - ray.eyePoint
      # norm_vecpc = np.linalg.norm(vec_pc)
      # # Projection of center of sphere onto the rays
      # u = self.center - ray.eyePoint
      # puv = GT.normalize(ray.viewDirection) * np.dot(ray.viewDirection, u)
      # projected_point = ray.eyePoint + puv

      # if np.dot(vec_pc, ray.viewDirection) < 0:
      #   # Sphere behind origin
      #   if norm_vecpc > self.radius:
      #     # No intersection
      #     return isect
      #   elif np.fabs(norm_vecpc - self.radius) < EPS_DISTANCE:
      #     # CASE 2: sits on surface
      #     # technically intersection = ray.eyePoint but said to ignore in spec
      #     return isect
      #   else:
      #     # Inside the sphere for real
      #     dist = np.sqrt(self.radius**2 - np.linalg.norm(projected_point - self.center)**2)
      #     dil = dist - np.linalg.norm(projected_point - ray.eyePoint)
      #     isect.p = ray.eyePoint + ray.viewDirection * dil
      #     isect.t = dil
      #     isect.material = self.material
      #     isect.n = GT.normalize(isect.p - self.center)
      #     return isect
      # else:
      #   # Outside of sphere
      #   if np.linalg.norm(self.center - projected_point) > self.radius:
      #     # no intersection
      #     return isect
      #   else:
      #     dist = np.sqrt(self.radius**2 - np.linalg.norm(projected_point - self.center)**2)
      #     if norm_vecpc > self.radius:
      #       # origin is outside sphere
      #       di1 = np.linalg.norm(projected_point - ray.eyePoint) - dist
      #     else:
      #       # origin is inside sphere
      #       di1 = np.linalg.norm(projected_point - ray.eyePoint) + dist

      #     isect.p = ray.eyePoint + ray.viewDirection * di1
      #     isect.material = self.material
      #     isect.t = di1
      #     isect.n = GT.normalize(isect.p - self.center)
      #     return isect
      # ===== END SOLUTION HERE =====
      return isect


class Plane:
  """

  Plane class

  Plane passing through origin with a given normal. If the second material is
  defined, it has a checkerboard pattern.

  A plane can be used as a floor, wall or ceiling. E.g. see cornell.xml

  """

  def __init__(self, params={}):
      # normal=[0.0,1.0,0.0], material = Material(), material2 = 0 ):
      self.normal = GT.normalize(np.array(params.get('normal', [0.0, 1.0, 0.0])))
      material_list = params.get('material', [Material(), None])
      if type(material_list) is not list:
          self.material = material_list
          self.material2 = None
      else:
          self.material = material_list[0]
          self.material2 = material_list[1]
      # print(params)
      # print(self.normal, self.material, self.material2)

  def intersect(self, ray):
    '''
    Find the intersection of the ray with the plane. Consider the ray and the
    plane to be in the same coordinate system.
    Return the result of intersection in a variable of type IntersectionResult.

    Note:
    1. For checkerboard planes there are two materials. You need to consider
    what the material is at the intersection point. If the plane has only 1
    material then self.material2 is set to None. To determine whether the plane
    has checkerboard pattern, you should have code like:
    if self.material2 is not None:
        # the plane has checkerboard pattern
    2. If a ray originates on the plane and goes away from the plane then that
    is not considered as an intersection. Otherwise we'll have problem with
    shadow rays.
    3. If the ray lies entirely on the plane we don't consider that to be an
    intersection (i.e. we won't see the plane in the rendered scene.)
    see TestXZPlaneThroughOriginNoIntersectionWithRay.test_ray_on_plane in
    TestPlaneIntersection.py for corresponding test case.
    '''

    '''
    Implement intersection between the ray and the current object and
    return IntersectionResult variable (isect) which will store the
    intersection point, the normal at the intersection and material of the
    object at the intersection point. For checkerboard planes you need to
    decide which of the two materials to use at the intersection point.
    '''
    isect = IntersectionResult()
    # TODO check about antialiasing because stupid dots on the horizon for plane scene

    global EPS_DISTANCE  # use this for testing if a variable is close to 0
    # TODO ===== BEGIN SOLUTION HERE =====
    ln = np.dot(self.normal, ray.viewDirection)
    pl = np.dot(-self.normal, ray.eyePoint)

    if np.fabs(ln) < EPS_DISTANCE:
      # in both cases, either the plane contains all the points
      # ad we don't consider it an intersection or runs parallel and never intersects
      return isect

    t = pl / ln

    if t < -EPS_DISTANCE or (np.fabs(pl) < EPS_DISTANCE):
      return isect

    isect.p = ray.eyePoint + t*ray.viewDirection
    isect.t = t
    isect.n = self.normal

    if self.material2 is not None:
      # checkerboard pattern
      x = isect.p[0]
      z = isect.p[2]
      if x < 0:
        x -= 10001.0
      if z < 0:
        z -= 10001.0
      if (int(x) ^ int(z)) & 1:
        isect.material = self.material2
      else:
        isect.material = self.material
    else:
      # 1 material
      isect.material = self.material

    # ===== END SOLUTION HERE =====
    return isect


class Box:
  """

  Box class

  Axis-aligned box defined by setting a pair of opposing points.

  """

  def __init__(self, params={}):
      # minPoint = [-1, -1, -1], maxPoint = [1, 1, 1], material = Material()):
      self.minPoint = np.array(params.get('min', [-1., -1., -1.]))
      self.maxPoint = np.array(params.get('max', [1., 1., 1.]))
      self.material = params.get('material', Material())
      assert(np.all(self.minPoint <= self.maxPoint))
      # print(self.minPoint, self.maxPoint, self.material)

  def intersect(self, ray):
    """
      The box can be viewed as the intersection of 6 planes. The following code
      checks the intersection to all planes and the order.  Depending on the
      order we detect the intersection.

      Note:
      1. At the box corners you can return any one of the three normals.
      2. You can assume that all rays originate outside the box
      3. A ray can originate on one of the plane or corners of the box and go
         outside in which case we do not consider that to be an intersection
         with the box.
    """
    '''
    Implement intersection between the ray and the current object and
    return IntersectionResult variable (isect) which will store the
    intersection point, the normal at the intersection and material of the
    object at the intersection point.
    '''
    isect = IntersectionResult()

    global EPS_DISTANCE  # use this for testing if a variable is close to 0
    # tmin and tmax are temporary variables to keep track of the order of the
    # plane intersections. The ray will pass through at least a set of parallel
    # planes. tmin is the last intersection of the first planes of each set,and
    # tmax is the first intersection of the last planes of each set.
    t_max = np.inf
    t_min = -np.inf

    # parameters = [self.minPoint, self.maxPoint]
    # TODO ===== BEGIN SOLUTION HERE =====
    t_min_index = 0
    t_min_using_reverse = temp_r = False
    normals = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    for i in range(0, 3):
      if ray.viewDirection[i] == 0.0:
        continue
      temp_r = False
      t2t = (self.minPoint[i] - ray.eyePoint[i])/ray.viewDirection[i]
      t1t = (self.maxPoint[i] - ray.eyePoint[i])/ray.viewDirection[i]
      if(t2t > t1t):
        temp_r = True
      t1 = min(t1t, t2t)
      t2 = max(t1t, t2t)
      if t1 > t_min:
        t_min_index = i
        t_min_using_reverse = not temp_r
        t_min = t1
      if t2 < t_max:
        # t_max_using_reverse = temp_r
        t_max = t2

    # min_with_error = self.minPoint + EPS_DISTANCE
    # max_with_error = self.maxPoint - EPS_DISTANCE

    if t_min > EPS_DISTANCE and t_min < t_max:
      isect.p = ray.eyePoint + ray.viewDirection * t_min
      isect.t = t_min
      isect.n = normals[t_min_index]
      if t_min_using_reverse:
        isect.n *= -1
      isect.material = self.material

    # ===== END SOLUTION HERE =====
    return isect


class SceneNode:
  """
  SceneNode class

  This intersectable object is used as a transformation in the scene creation.
  It allows the scene to be build in a hierarchical fashion. It allows
  rotations and translations. The intersection ray will be transformed to
  find the intersection in the transformed space, and the intersection
  result is transformed back to the original coordinate space.
  It performs a test for all its children.

  """
  def __init__(self, M=np.eye(4), params=None):
    self.children = []
    self.M = M
    if params is not None:
        rot_angles = np.array(params.get('rotation', [0., 0., 0.]))
        translate_amount = np.array(params.get('translation', [0., 0., 0.]))
        scale_amount = np.array(params.get('scale', [1., 1., 1.]))
        # compute the transformation matrix that
        # gets applied to all children of this node
        Tform = GT.translate(translate_amount) * GT.rotateX(rot_angles[0]) * \
            GT.rotateY(rot_angles[1]) * GT.rotateZ(rot_angles[2]) * \
            GT.scale(scale_amount)
        self.M = Tform.getA()

    self.Minv = np.linalg.inv(self.M)
    # print(self.M, self.Minv)

  def intersect(self, ray):
    '''
    Implement intersection between the ray and the current object and
    return IntersectionResult variable (isect) which will store the
    intersection point, the normal at the intersection and material of the
    object at the intersection point. The variable isect should contain the
    nearest intersection point and all its properties.
    '''
    isect = IntersectionResult()
    transformedRay = Ray()
    ray_tip = ray.eyePoint + ray.viewDirection
    # import pdb;pdb.set_trace()
    # transformedRay.viewDirection = GT.normalize(np.dot(np.transpose(self.M), np.append(ray.viewDirection, [0]))[:3])
    transformedRay.eyePoint = np.dot(self.Minv, np.append(ray.eyePoint, [1]))
    transformedRay.eyePoint = transformedRay.eyePoint[:3] / transformedRay.eyePoint[3]
    trans_ray_tip = np.dot(self.Minv, np.append(ray_tip, [1]))
    trans_ray_tip = trans_ray_tip[:3] / trans_ray_tip[3]
    transformedRay.viewDirection = trans_ray_tip - transformedRay.eyePoint
    # Check
    # https://github.com/jianhe25/Tiny-ray-tracer/blob/master/hw3-RayTracer/RayTracer.cpp
    global EPS_DISTANCE  # use this for testing if a variable is close to 0
    # TODO ===== BEGIN SOLUTION HERE =====
    intersections = []
    for child in self.children:
      intersection = child.intersect(transformedRay)
      if intersection.t == np.inf:
        continue
      # import pdb;pdb.set_trace()
      # intersection.p = ray.eyePoint + ray.viewDirection * intersection.t
      intersection.p = np.dot(self.M, np.append(intersection.p, [1]))
      intersection.p = intersection.p[:3]/intersection.p[3]
      intersection.t = np.linalg.norm(intersection.p - ray.eyePoint)
      intersection.n = np.dot(np.transpose(self.Minv), np.append(intersection.n, [0]))[:3]
      # intersection.n = intersection.n[:3]/intersection.n[3]
      # intersection.n = GT.normalize(intersection.n)

      intersections.append(intersection)

    min_isect = isect

    for iss in intersections:
      # if iss.material is not None and iss.material.name == "green":
        # import pdb;pdb.set_trace()
      if (iss.t < min_isect.t):
        min_isect = iss

    isect = min_isect
    # ===== END SOLUTION HERE =====
    return isect
