#!/usr/bin/python3
# encoding: utf-8

import sys
import os
import re
import copy
import numpy
import types
from typing import Iterable
from math import sqrt, cos, sin, tan, atan, atan2, pow, fmod, pi, acos, asin, inf, copysign

from xml import dom
from xml.dom import minidom
from xml.dom.minidom import Document
from xml.dom.minidom import Element
from xml.dom.minidom import Attr
from xml.dom.minidom import Node

svg_ns = 'http://www.w3.org/2000/svg'

class Precision:
    def __init__(self, precision):
        self.precision = precision
        
    def p_round(self, v):
        rv = round(v, self.precision)
        return rv
    
    def a_round(self, v):
        rv = round(v, self.precision * 2)
        return rv

    def h_round(self, v):
        return round(v * pow(10, self.precision))
    
class Coord:
    def __init__(self, p, x, y, w = 1.0):
        self.p = p
        self.x = p.p_round(x)
        self.y = p.p_round(y)
        self.w = p.p_round(w)
        
    def dist(self, o) -> float:
        (x,y) = (0,0)
        if isinstance(o, Coord):
            c: Coord = o
            (x,y) = (c.x, c.y)
        else:
            raise RuntimeError('can only compute distance to another coordinate')
        dx = self.x - x
        dy = self.y - y
        return sqrt(dx*dx+dy*dy) 

    def __eq__(self, o):
        p = self.p
        return p.h_round(self.x) == p.h_round(o.x) and p.h_round(self.y) == p.h_round(o.y)
        
    def __hash__(self):
        p = self.p
        return p.h_round(self.x) * 3 + p.h_round(self.y) * 7

    def __repr__(self):
        return f'Coord({self.x},{self.y})'
    
    def __str__(self):
        return f'({self.x},{self.y})'
    
    def __add__(self, o):
        if isinstance(o, Coord):
            p = self.p
            x = self.x + o.x
            y = self.y + o.y
            return Coord(p,x,y)
        if isinstance(o, tuple):
            p = self.p
            (x,y) = o
            x += self.x
            y += self.y
            return Coord(p,x,y)
        
    def __iter__(self):
        return iter((self.x, self.y))
    
class Bounds:
    def __init__(self, p, cmin = None, cmax = None):
        self.p = p
        if cmin and cmax:
            self.empty = False
            self.cmin = cmin
            self.cmax = cmax
        else:
            self.empty = True
            self.cmin = Coord(p, 0.0, 0.0)
            self.cmax = Coord(p, 0.0, 0.0)
        
    def width(self):
        return 0 if self.empty else self.cmax.x - self.cmin.x 
    
    def height(self):
        return 0 if self.empty else self.cmax.y - self.cmin.y
    
    def _expand_coord(self, coord: Coord):
        if self.empty:
            self.cmin.x = self.cmax.x = coord.x
            self.cmin.y = self.cmax.y = coord.y
            self.empty = False
        else:
            self.cmin.x = min(self.cmin.x, coord.x)
            self.cmin.y = min(self.cmin.y, coord.y)
            self.cmax.x = max(self.cmax.x, coord.x)
            self.cmax.y = max(self.cmax.y, coord.y)
            
    def pad(self, v):
        cminpad = self.cmin + (-v,-v)
        cmaxpad = self.cmax + (v,v)
        self.expand(cminpad)
        self.expand(cmaxpad)

    def expand(self, obj):     
        if isinstance(obj, Coord):
            c: Coord = obj
            self._expand_coord(c)
        if isinstance(obj, Bounds):
            b: Bounds = obj
            if not b.empty:
                self._expand_coord(b.cmin)
                self._expand_coord(b.cmax)
        if isinstance(obj, Geom):
            g: Geom = obj
            self.expand(g.bounds())
        if isinstance(obj, Path):
            g: Geom = obj
            self.expand(g.bounds())
        if isinstance(obj, Vert):
            v: Vert = obj
            self._expand_coord(v.coord)
        if isinstance(obj, dict):
            for o in obj.values():
                self.expand(o)
        if isinstance(obj, Iterable):
            for o in obj:
                self.expand(o)
        return self
        
    def union(self, b):
        b: Bounds = Bounds()
        b.expand(self)
        b.expand(b)
        return b
    
    def __str__(self):
        return f'Bounds {str(self.cmin)} [] {str(self.cmax)}'
    
    def to_view_box(self):
        view_box = []
        view_box.extend([self.cmin.x, self.cmin.y])
        view_box.extend([self.width(), self.height()])
        return view_box
        
    def to_path_data(self):
        svg_data = []
        svg_data.append('M')
        cmin = self.cmin
        cmax = self.cmax
        c1 = cmin
        c2 = Coord(p, cmax.x, cmin.y)
        c3 = cmax
        c4 = Coord(p, cmin.x, cmax.y)
        svg_data.extend(c1)
        svg_data.append('L')
        svg_data.extend(c2)
        svg_data.append('L')
        svg_data.extend(c3)
        svg_data.append('L')
        svg_data.extend(c4)
        svg_data.append('Z')
        return svg_data

class Vert:
    def __init__(self, vid, coord):
        self.vid = vid
        self.coord = coord
        self.color = self.vid
        
    def dist(self, o) -> float:
        if isinstance(o,Coord):
            c: Coord = o
            return self.coord.dist(c)
        elif isinstance(o,Vert):
            v: Vert = o
            return self.dist(v.coord)
        else:
            raise RuntimeError('can only compute distance to a coordinate or vertex')

    def __str__(self):
        return 'V' + str(self.vid) + '@' + str(self.coord) + '-c' + str(self.color)
    
    def to_path_data(self):
        svg_data = []
        svg_data.extend(self.coord)
        return svg_data
        
        
class Geom:
    def __init__(self, p, gid, verts):
        self.p = p
        self.gid = gid
        self.verts = verts
        self.color = verts[0].color
        
    def compute_color(self):
        mods = 0
        self.color = min(v.color for v in self.verts)
        for v in self.verts:
            if v.color != self.color:
                v.color = self.color
                mods += 1
        return mods
    
    def compute(self):
        pass
    
    def vfirst(self) -> Vert:
        return self.verts[0]
    
    def vlast(self) -> Vert:
        return self.verts[-1]
    
    def bounds(self) -> Bounds:
        b = Bounds(p)
        b.expand(self.verts)
        return b
    
    def reverse(self):
        self.verts.reverse()
        self.compute()
        
class Line(Geom):
    def __init__(self, p, gid, v1, v2):
        super().__init__(p, gid, [v1, v2])
        
    def __str__(self):
        return f'line-{self.gid} {self.vfirst()} -> {self.vlast()}'
    
    def length(self) -> float:
        return self.vfirst().dist(self.vlast())
    
    def to_path_data(self):
        svg_data = []
        svg_data.append('L')
        c2 = self.vlast().coord
        svg_data.extend(c2)
        return svg_data
    
class Arc(Geom):
    def __init__(self, p, gid, v1, v2, rx, ry, xrot, fa, fs):
        super().__init__(p, gid, [v1, v2])
        self.rx = p.p_round(rx)
        self.ry = p.p_round(ry)
        self.xrot = p.a_round(xrot)
        self.rx = rx
        self.ry = ry
        self.xrot = xrot
        self.fa = fa
        self.fs = fs
        
    def reverse(self):
        Geom.reverse(self)
        self.fs = not self.fs
        self.compute()
        
    def __str__(self):
        return f'arc-{self.gid} {self.vfirst()} -> {self.vlast()}'
    
    def ellipse_pt(self, cx, cy, rx, ry, phi, theta):
        (x,y) = (cx + rx*cos(theta)*cos(phi) - ry*sin(theta)*sin(phi), 
                 cy + rx*cos(theta)*sin(phi) + ry*sin(theta)*cos(phi))
        return Coord(p, x, y)
    
    def compute(self):
        p = self.p
        v1 = self.vfirst().coord
        v2 = self.vlast().coord
        rx = self.rx
        ry = self.ry
        fa = self.fa
        fs = self.fs
        phi = self.xrot
        (x1,y1) = (v1.x, v1.y)
        (x2,y2) = (v2.x, v2.y)
        rx = abs(rx)
        ry = abs(ry)
        [[x1p], [y1p]] = numpy.matmul([[cos(phi), sin(phi)],[-sin(phi), cos(phi)]], [[(x1-x2)/2.0],[(y1-y2)/2.0]])
        L = ((x1p*x1p)/(rx*rx)) + (y1p*y1p)/(ry*ry)
        print('GID:', self.gid)
        print('v1:', v1, 'v2:', v2)
        print('L',L)
        if L > 1.0:
            rx = sqrt(L) * rx
            ry = sqrt(L) * ry
        print('rx,ry',rx,ry)
        print('fa:',fa)
        print('fs:',fs)
        print('xrot:', phi)
        radicant = ((rx*rx*ry*ry - rx*rx*y1p*y1p - ry*ry*x1p*x1p) /  
                    (rx*rx*y1p*y1p + ry*ry*x1p*x1p))
        print('radicant:', radicant);
        radicant = -sqrt(abs(radicant)) if fs == fa else sqrt(abs(radicant))
        [[cxp], [cyp]] = numpy.multiply([[rx*y1p/ry],[-ry*x1p/rx]], radicant)
        [[cx],[cy]] = numpy.matmul([[cos(phi), -sin(phi)],[sin(phi), cos(phi)]],[[cxp],[cyp]]) + [[(x1+x2)/2.0],[(y1+y2)/2.0]]
        (cx,cy)=(p.p_round(cx), p.p_round(cy))
        theta0 = fmod(atan2((y1p-cyp)/ry,(x1p-cxp)/rx) + 2*pi,2*pi)
        theta1 = fmod(atan2((-y1p-cyp)/ry,(-x1p-cxp)/rx) + 2*pi,2*pi)
        thetaD = theta1 - theta0
        thetaD = copysign(thetaD, 1.0 if fs else -1.0)
        if (fa and abs(thetaD) < pi) or (not fa and abs(thetaD) > pi):
            thetaD = copysign(2.0*pi - abs(thetaD),thetaD)
            
        thetaXT = [ fmod(atan2(ry*tan(phi), rx) + 2*pi,2*pi), fmod(atan2(ry*tan(phi), rx) + 3*pi,2*pi),
                    fmod(atan2(ry, tan(phi)*rx) + 2*pi,2*pi), fmod(atan2(ry, tan(phi)*rx) + 3*pi,2*pi)]
        thetaXTX = {}
        for t in thetaXT:
            thetaXTX[t] = self.ellipse_pt(cx, cy, rx, ry, phi, t)
       
        print('txtreme:', thetaXTX)
        print('center:', cx, cy)
        print('angles:', theta0, theta1, thetaD)
        print('halfangle:',fmod(theta0 + (thetaD / 2.0), 2*pi))
        
        if (theta0 < theta1):
            (ctheta0,ctheta1) = (theta0,theta1)
            cfs = fs
        else:
            (ctheta0,ctheta1) = (theta1,theta0)
            cfs = not fs
        print('ctheta0:', ctheta0, 'ctheta1:', ctheta1)
        print('cfs:', cfs)
        
        
        b = Bounds(p);
        b.expand(self.verts)
        print(str(b))
        for (t,ep) in thetaXTX.items():
            inside = (ctheta0 <= t) and (t <= ctheta1)
            print('inside: ', inside, 'theta:', t, 'pt:', ep)
            if inside == cfs:
                b.expand(ep)

        
        self.theta0 = theta0
        self.theta1 = theta1
        self.thetaD = thetaD
        self.center = Coord(self.p, cx, cy)
        self.bbox = b
    
    def bounds(self):
        return self.bbox
    
    # def test_path(self):
    #     svg_data = []
    #     svg_data.append('M')
    #     svg_data.extend(self.center.to_svg_data())
    #     svg_data.append('L')
    #     (x,y) = self.ellipse_pt(self.center.x, self.center.y, self.rx, self.ry, self.xrot, self.theta0)
    #     svg_data.extend((x,y))
    #
    #     svg_data.append('M')
    #     svg_data.extend(self.center.to_svg_data())
    #     svg_data.append('L')
    #     (x,y) = self.ellipse_pt(self.center.x, self.center.y, self.rx, self.ry, self.xrot, self.theta1)
    #     svg_data.extend((x,y))
    #
    #     svg_data.append('M')
    #     svg_data.extend(self.center.to_svg_data())
    #     svg_data.append('L')
    #     (x,y) = self.ellipse_pt(self.center.x, self.center.y, self.rx, self.ry, self.xrot, fmod(self.theta0 + (self.thetaD / 2.0), 2*pi))
    #     svg_data.extend((x,y))
    #     return svg_data

    
    def to_path_data(self):
        svg_data = []
        svg_data.extend(['A', self.rx, self.ry, self.xrot, 1 if self.fa else 0, 1 if self.fs else 0])
        c2 = self.vlast().coord
        svg_data.extend(c2)
        return svg_data
    
class Circle(Geom):
    def __init__(self, p, gid, v1, r):
        super().__init__(p, gid, [v1])
        self.r = self.p.p_round(r)
    
    def __str__(self):
        return f'circle-{self.gid} {self.vfirst()} r{self.r}'
    
    def bounds(self):
        b = super().bounds()
        c = self.vfirst().coord
        r = self.r
        bcp = c + (r,r)
        bcm = c + (-r,-r)
        b.expand([bcp, bcm])
        return b
    
class Rect(Geom):
    def __init__(self, p, gid, v1, v2, rx=0, ry=0):
        super().__init__(p, gid, [v1, v2])
        self.width = abs(v1.coord.x - v2.coord.x)
        self.height = abs(v1.coord.y - v2.coord.y)
        self.rx = rx
        self.ry = ry
        
    def __str__(self):
        return f'rect-{self.gid} {self.vfirst()} [=] {self.width}x{self.height}'
        
    def to_path_data(self):
        svg_data = []
        svg_data.append('M')
        c = self.vfirst().coord
        c2 = c + (width, 0)
        c3 = c + (width, height)
        c4 = c + (0, height)
        svg_data.extend(c)
        svg_data.append('L')
        svg_data.extend(c2)
        svg_data.append('L')
        svg_data.extend(c3)
        svg_data.append('L')
        svg_data.extend(c4)
        svg_data.append('Z')
        return svg_data

class Path:
    def __init__(self, p, pid):
        self.pid = pid
        self.geoms = {}
        self.path = []
        
    def compute(self):
        geoms = self.geoms.copy()
        path = []
        (gid,geom) = geoms.popitem()
        path.append(geom)
        mods = 1
        while mods > 0 and len(geoms) > 0:
            mods = 0
            to_remove = set(())
            for (gid, geom) in geoms.items():
                if path[0].vfirst().vid == geom.vlast().vid:
                    to_remove.add(gid)
                    path.insert(0,geom)
                    mods = mods + 1
                elif path[0].vfirst().vid == geom.vfirst().vid:
                    to_remove.add(gid)
                    geom.reverse()
                    path.insert(0,geom)
                    mods = mods + 1
                elif path[-1].vlast().vid == geom.vfirst().vid:
                    to_remove.add(gid)
                    path.append(geom)
                    mods = mods + 1
                elif path[-1].vlast().vid == geom.vlast().vid:
                    to_remove.add(gid)
                    geom.reverse()
                    path.append(geom)
                    mods = mods + 1
            for gid in to_remove: 
                geoms.pop(gid)
            #print(str(self.pid) + ' mods: ' + str(mods));
            
        if len(geoms) > 0:
            print('Error in: ', self.pid, len(geoms))
            for (gid, geom) in geoms.items():
                print('    ', self.pid, geom)
            raise RuntimeError('Unable to incorporate geometry into path ' + str(self.pid))
        
        self.path = path
        
    def bounds(self):
        b = Bounds(p)
        b.expand(self.geoms.values())
        return b
                
    def to_path_data(self):
        svg_data = []
        svg_data.append('M')
        svg_data.extend(self.path[0].vfirst().to_path_data())
        for geom in self.path:
            svg_data.extend(geom.to_path_data())
        return svg_data
    
    def __str__(self):
        s = f'path-{self.gid}\n'
        for geom in self.path:
            s += '    ' + str(geom) + '\n'
        return s


                
class Transform:
    def identity():
        identity_mat = [[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]]
        return identity_mat
    
    def __init__(self, p):
        self.p = p
        self.matrix = Transform.identity()
         
    def scale(self, xscale, yscale):
        scale_mat = [[xscale,    0.0,    0.0],
                     [   0.0, yscale,    0.0],
                     [   0.0,    0.0,    1.0]]
        self.matrix = numpy.matmul(scale_mat, self.matrix)
    
    def xform(self, obj):
        if isinstance(obj, Coord):
            coord = obj
            coord_mat = [coord.x, coord.y, coord.w]
            coord_mat = numpy.matmul(coord_mat, self.matrix)
            coord = Coord(p, coord_mat[0], coord_mat[1], coord_mat[2])
            return coord
        elif isinstance(obj, Bounds):
            bounds = obj
            if not bounds.empty:
                cmin = self.xform(bounds.cmin)
                cmax = self.xform(bounds.cmax)
                bounds = Bounds(p)
                bounds.expand(cmin)
                bounds.expand(cmax)
                return bounds
            return Bounds(p)
        else:
            raise NotImplementedError(str(type(obj)) + 'not supported')
        
class Scene:
    def __init__(self, p: Precision):
        self.p = p
        self.next_vid = 0
        self.next_gid = 0
        self.verts = {}
        self.geoms = []
        self.circles = []
        self.paths = {}
        self.bounds = Bounds(p)
        
    def add_vert(self, coord):
        if coord in self.verts.keys():
            return self.verts[coord]
        else:
            vid = self.next_vid
            vert = Vert(vid, coord)
            self.next_vid += 1
            self.verts[vert.coord] = vert
            self.bounds.expand(coord)
            return vert
        
    def add_line(self, v1, v2):
        self.geoms.append(Line(self.p, self.next_gid, v1, v2))
        self.next_gid += 1
 
    def add_arc(self, v1, v2, rx, ry, xrot, fa, fs):
        self.geoms.append(Arc(self.p, self.next_gid, v1, v2, rx, ry, xrot, fa, fs))
        self.next_gid += 1
        
    def add_circle(self, v1, r):
        r = p.p_round(r)
        circle = Circle(self.p, self.next_gid, v1, r)
        self.next_gid += 1
        self.circles.append(circle)

    def compute_geom(self):
        for geom in self.geoms:
            geom.compute()

    def compute_colors(self):
        mods = 1
        while mods > 0:
            mods = 0
            for geom in self.geoms:
                mods += geom.compute_color()
            print('mods: ' + str(mods))
            
    def compute_paths(self):
        self.paths.clear()
        for geom in self.geoms:
            color = geom.color
            if color not in self.paths.keys():
                self.paths[color] = Path(self.p, color)
            path = self.paths[color]
            path.geoms[geom.gid] = geom
        print('pathcount: ', len(self.paths))
        for path in self.paths.values():
            path.compute()
        
    
p = Precision(3)
scene = Scene(p)


xml_ns_uri = 'http://www.w3.org/XML/1998/namespace'

def elements_of(elem: Element) -> Iterable[Element]:
    return filter(lambda node: isinstance(node, Element), elem.childNodes)

def attributes_of(elem: Element) -> Iterable[Attr]:
    return filter(lambda node: isinstance(node, Attr), elem.childNodes)


class PathReader:
    path_ops = 'MmLlHhVvCcSsQqTtAaZz'
    path_re = re.compile(f'([{path_ops}])([^{path_ops}]+)')
    
    def __init__(self, scene):
        self.scene = scene
        self.p = scene.p
        
    def read(self, path_data):
        match = self.path_re.search(path_data)
        cur_vert = None
        prev_vert = None
        while (match):
            (op, data) = match.group(1,2)
            op: str = op.strip()
            data: str = data.strip()
            cur_vert = None
            if op == 'M':
                (x,y) = data.split()
                coord = Coord(p, float(x), float(y))
                cur_vert = scene.add_vert(coord)
            elif op == 'm':
                (x,y) = data.split()
                coord = Coord(p, float(x), float(y))
                if prev_vert:
                    coord = coord + prev_vert.coord
                cur_vert = scene.add_vert(coord)
            elif op == 'L':
                (x,y) = data.split()
                coord = Coord(p, float(x), float(y))
                cur_vert = scene.add_vert(coord)
                scene.add_line(prev_vert, cur_vert)
            elif op == 'A':
                (rx, ry, xrot, fa, fs, x, y) = data.split()
                coord = Coord(p, float(x),float(y))
                cur_vert = scene.add_vert(coord)
                scene.add_arc(prev_vert, cur_vert, float(rx), float(ry), float(xrot), int(fa) == 1, int(fs) == 1)
            else:
                print('unhandled op:' + op)
            prev_vert = cur_vert
            match = self.path_re.search(path_data, match.end())



class SvgReader:
    svg_ns = 'http://www.w3.org/2000/svg'
    
    def __init__(self, doc, scene):
        self.doc = doc
        self.scene = scene
        self.path_reader = PathReader(scene)
        
    def read(self):
        de = self.doc.documentElement
        self.read_element(de);
        
    def read_children(self, e):
        for child in elements_of(e):
            self.read_element(child)
            
    def read_element(self, e):
        name = e.nodeName
        handler_name = 'read_' + name
        handler = self.read_default
        if hasattr(self, handler_name):
            handler_attr = getattr(self, handler_name)
            if callable(handler_attr):
                handler = handler_attr
        handler(e)
        
    def read_default(self, e: Element):
        name = e.nodeName
        print(f'Unhandled element <{name}>')
            
    def read_svg(self, svg: Element):
        print('Handle: <svg>')
        self.read_children(svg)
        
    def read_g(self, g: Element):
        print('Handle: <g>')
        self.read_children(g)
        
    def read_path(self, path: Element):
        path_id = path.getAttribute('id')
        print(path_id)
        path_data = path.getAttribute('d')
        self.path_reader.read(path_data)
        
    def read_circle(self, circle: Element):
        circle_id = circle.getAttribute('id')
        print(circle_id)
        cx_str = circle.getAttribute('cx')
        cx = float(cx_str)
        cy_str = circle.getAttribute('cy')
        cy = float(cy_str)
        cr_str = circle.getAttribute('r')
        cr = float(cr_str)
        coord = Coord(self.scene.p, cx, cy);
        vert = self.scene.add_vert(coord)
        self.scene.add_circle(vert, cr)






infile_name = sys.argv[1]
print(infile_name)

indoc = minidom.parse(infile_name)
reader = SvgReader(indoc, scene)

reader.read()
    

# path_re = re.compile('([MLHVCSQTAZ]) *([^MLHVCSQTAZ]+) *')


scene.compute_geom()
scene.compute_colors()
scene.compute_paths()

def replacement_writexml(self, writer, indent="", addindent="", newl=""):
    # indent = current indentation
    # addindent = indentation to add to higher levels
    # newl = newline string
    writer.write(indent+"<" + self.tagName)

    attrs = self._get_attributes()
    a_names = sorted(attrs.keys())
    
    move_front = ['id', 'class']
    for mf in reversed(move_front):
        if mf in a_names:
            a_names.remove(mf)
            a_names.insert(0,mf)

    for a_name in a_names:
        writer.write(" %s=\"" % a_name)
        minidom._write_data(writer, attrs[a_name].value)
        writer.write("\"")
    if self.childNodes:
        writer.write(">")
        if (len(self.childNodes) == 1 and
            self.childNodes[0].nodeType == Node.TEXT_NODE):
            self.childNodes[0].writexml(writer, '', '', '')
        else:
            writer.write(newl)
            for node in self.childNodes:
                node.writexml(writer, indent+addindent, addindent, newl)
            writer.write(indent)
        writer.write("</%s>%s" % (self.tagName, newl))
    else:
        writer.write("/>%s"%(newl))

def svg_str(obj):
    svg_data = obj.to_path_data()
    svg_data = [ str(d) for d in svg_data ]
    return ' '.join(svg_data)

class SvgWriter:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.show_bounds = False
        self.show_labels = False
        self.use_css = False
        self.use_attribs = True
        self.styles = {
            'cut': {
                'fill': 'none',
                'stroke':'#FF0000',
                'stroke-width':'0.25px',
                },
            'bounds': {
                'fill':'none',
                'stroke':'#0000FF',
                'stroke-width':'0.25px',
                },
            'engrave': {
                'fill':'#000000',
                'stroke':'none',
                }
            }
        self.impl = minidom.getDOMImplementation()
        
    def create_stylesheet(self, doc: Document):
        style_elem = doc.createElement('style')
        style_elem .setAttribute('type', 'text/css')
        styles = self.styles
        css_data = f'\n'
        for style_name in sorted(styles.keys()):
            css_data += f'.{style_name} {{\n'
            style_attrs = styles[style_name]
            for attr_name in sorted(style_attrs.keys()):
                css_data += f'\t{attr_name}: {style_attrs[attr_name]};\n'
            css_data += f'}}\n'
        css_data += f'\n'
        css_cdata = doc.createCDATASection(css_data)
        style_elem.appendChild(css_cdata)
        return style_elem

        
        
    def add_style(self, name, elem: Element):
        if self.use_css:
            elem.setAttribute('class', name)
        if self.use_attribs:
            style_attrs = self.styles[name]
            for attr_name in sorted(style_attrs.keys()):
                elem.setAttribute(attr_name, style_attrs[attr_name])
            # path_elem.setAttribute('fill', 'none')
            # path_elem.setAttribute('stroke-width', '.25px')
            # path_elem.setAttribute('stroke', '#FF0000')
            
    def createShapeElement(self, doc: Document, name):
        elem = doc.createElement(name)
        elem.writexml = types.MethodType(replacement_writexml, elem)
        return elem

    def to_xml_doc(self):
        svg_ns_uri = 'http://www.w3.org/2000/svg'
        xml_ns_uri = 'http://www.w3.org/XML/1998/namespace'

        impl = self.impl
        doctype = impl.createDocumentType('svg', '-//W3C//DTD SVG 1.1//EN', 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd')
        doc = impl.createDocument(svg_ns_uri, 'svg', doctype)
        
        svg: Element = doc.documentElement
        svg.setAttributeNS('xml_ns_uri', 'xmlns', svg_ns_uri)
        svg.setAttributeNS(svg_ns_uri, 'version', '1.1')
        
        if self.use_css:
            stylesheet_elem = self.create_stylesheet(doc)
            svg.appendChild(stylesheet_elem)
        
        g = doc.createElement('g')
        g.setAttribute('transform', 'scale(1,-1)')
        svg.appendChild(g)
        
        xmlbounds = Bounds(p)
        
        def lengthof(g: Geom) -> float:
            if isinstance(g, Line):
                l: Line = g
                return l.length()
            else:
                return 0.0
        
        for (pid, path) in scene.paths.items():
            path_elem = self.createShapeElement(doc, 'path')
            path_elem.setAttribute('id', f'path-{path.pid}')
            path_elem.setAttribute('d', svg_str(path))
            self.add_style('cut', path_elem)
            xmlbounds.expand(path)
            
            if self.show_bounds:
                b = path.bounds()
                bounds_elem = self.createShapeElement(doc, 'path')
                bounds_elem.setAttribute('id', f'pathbounds-{path.pid}')
                self.add_style('bounds', bounds_elem)
                bounds_elem.setAttribute('d', svg_str(b))
                g.appendChild(bounds_elem)
                
            g.appendChild(path_elem)
                
            if self.show_labels:
                longest: Geom = max(path.path, key=lengthof)
                c1 = longest.vfirst().coord
                c2 = longest.vlast().coord
                if c1.x == c2.x:
                    if c1.y > c2.y:
                        (c1,c2) = (c2,c1)
                if c1.x > c2.x:
                    (c1,c2) = (c2,c1)
                    
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                theta = atan2(-dy,dx)
                deg = theta / pi * 180.0
                
                text = f'path-{path.pid}'
                font_size = (lengthof(longest) / (2 * len(text))) * 2.82
                font_size = min(font_size, 10.0) 
                text_elem = self.createShapeElement(doc, 'text')
                text_elem.setAttribute('x', str(c1.x))
                text_elem.setAttribute('y', str(-c1.y))
                text_elem.setAttribute('font-size', f'{str(font_size)}pt')
                text_elem.setAttribute('transform', f'scale(1,-1) rotate({str(deg)}, {str(c1.x)}, {str(-c1.y)})')
                text_node = doc.createTextNode(text)
                text_elem.appendChild(text_node)
                g.appendChild(text_elem)
        
        for circle in scene.circles:
            circle_elem = self.createShapeElement(doc, 'circle') 
            center = circle.vfirst().coord
            radius = circle.r
            circle_elem.setAttributeNS(svg_ns, 'cx', str(center.x))
            circle_elem.setAttributeNS(svg_ns, 'cy', str(center.y))
            circle_elem.setAttributeNS(svg_ns, 'r', str(radius))
            circle_elem.setAttribute('id', f'circ-{circle.gid}')
            self.add_style('cut', circle_elem)
            xmlbounds.expand(circle)
        
            if self.show_bounds:
                b = circle.bounds()
                bounds_elem = self.createShapeElement(doc, 'path')
                bounds_elem.setAttribute('id', f'circlebounds-{circle.gid}')
                bounds_elem.setAttribute('d', svg_str(b))
                self.add_style('bounds', bounds_elem)
                g.appendChild(bounds_elem)
                
            g.appendChild(circle_elem)
            
            if self.show_labels:
                c1 = circle.vfirst().coord
                text = f'circ-{circle.gid}'
                font_size = ((circle.r * 2.0) / len(text)) * 2.82
                font_size = min(font_size, 10.0) 
                text_elem = self.createShapeElement(doc, 'text')
                text_elem.setAttribute('x', str(c1.x))
                text_elem.setAttribute('y', str(-c1.y))
                text_elem.setAttribute('font-size', f'{str(font_size)}pt')
                text_elem.setAttribute('text-anchor', 'middle')
                text_elem.setAttribute('transform', f'scale(1,-1)')
                text_node = doc.createTextNode(text)
                text_elem.appendChild(text_node)
                g.appendChild(text_elem)
        
                 
        xmlbounds.pad(3)
                
        #xmlbounds = scene.bounds
        attr_width = str(xmlbounds.width()) + 'mm'
        svg.setAttributeNS(svg_ns_uri, 'width', attr_width)
        
        attr_height = str(xmlbounds.height()) + 'mm'
        svg.setAttributeNS(svg_ns_uri, 'height', attr_height)
        
        tx = Transform(p)
        tx.scale(1.0, -1.0)
        view_box = tx.xform(xmlbounds)
        
        attr_viewbox = ' '.join([ str(d) for d in view_box.to_view_box()])
        svg.setAttributeNS(svg_ns_uri, 'viewBox', attr_viewbox)
    
        return doc



def gen_outfile_name(infile_name, postfix):        
    infile_split = infile_name.split('.')
    outfile_split = []
    outfile_split.extend(infile_split[0:-1])
    outfile_split[-1] = outfile_split[-1] + postfix
    outfile_split.extend(infile_split[-1:])
    outfile_name = '.'.join(outfile_split)
    return outfile_name

def write_xml_doc(doc, outfile_name):
    print(outfile_name)
    f = open(outfile_name, 'w')
    f.write(doc.toprettyxml())
    f.close()

svg_writer = SvgWriter(scene)
svg_writer.use_css = True
svg_writer.use_attribs = False

outdoc = svg_writer.to_xml_doc()
write_xml_doc(outdoc, gen_outfile_name(infile_name, '-tidy'))

svg_writer.show_labels = True

outdoc = svg_writer.to_xml_doc()
write_xml_doc(outdoc, gen_outfile_name(infile_name, '-tidy-labeled')) 
 


            
