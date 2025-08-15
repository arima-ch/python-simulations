import sys
import math
import time
import random
from dataclasses import dataclass

import numpy as np
import pyglet
import moderngl
from pyglet.window import key, mouse

# ---------- Simulation parameters ----------
MAX_PARTICLES = 1200
INIT_PARTICLES = 300
SOFTENING = 1e-1
THETA = 0.6  # Barnes-Hut opening angle
DT = 0.016
G = 200.0  # gravitational constant (scaled for visuals)

WINDOW_SIZE = (1280, 800)


# ---------- Utility dataclasses ----------
@dataclass
class Body:
    pos: np.ndarray  # shape (2,)
    vel: np.ndarray  # shape (2,)
    mass: float
    color: tuple


# ---------- Barnes-Hut Quadtree (2D) ----------
class Quad:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w  # half width

    def contains(self, pos):
        return (pos[0] >= self.x - self.w and pos[0] <= self.x + self.w and
                pos[1] >= self.y - self.w and pos[1] <= self.y + self.w)

    def NW(self):
        return Quad(self.x - self.w / 2, self.y + self.w / 2, self.w / 2)

    def NE(self):
        return Quad(self.x + self.w / 2, self.y + self.w / 2, self.w / 2)

    def SW(self):
        return Quad(self.x - self.w / 2, self.y - self.w / 2, self.w / 2)

    def SE(self):
        return Quad(self.x + self.w / 2, self.y - self.w / 2, self.w / 2)


class BHTree:
    def __init__(self, quad: Quad):
        self.quad = quad
        self.body = None
        self.center_of_mass = np.zeros(2, dtype=float)
        self.total_mass = 0.0
        self.NW = None
        self.NE = None
        self.SW = None
        self.SE = None

    def insert(self, body: Body):
        if not self.quad.contains(body.pos):
            return False

        if self.body is None and self.NW is None and self.total_mass == 0.0:
            # empty external node
            self.body = body
            self.center_of_mass = body.pos.copy()
            self.total_mass = body.mass
            return True

        if self.NW is None:
            # subdivide and move existing body into children (if exists)
            self._subdivide()
            if self.body is not None:
                existing = self.body
                self.body = None
                self._put_into_child(existing)

        # put new body into a child
        success = self._put_into_child(body)
        # update center of mass properly
        new_total = self.total_mass + body.mass
        if new_total > 0.0:
            self.center_of_mass = (self.center_of_mass * self.total_mass + body.pos * body.mass) / new_total
        self.total_mass = new_total
        return success

    def _subdivide(self):
        q = self.quad
        self.NW = BHTree(q.NW())
        self.NE = BHTree(q.NE())
        self.SW = BHTree(q.SW())
        self.SE = BHTree(q.SE())

    def _put_into_child(self, body: Body):
        if self.NW.quad.contains(body.pos):
            return self.NW.insert(body)
        if self.NE.quad.contains(body.pos):
            return self.NE.insert(body)
        if self.SW.quad.contains(body.pos):
            return self.SW.insert(body)
        if self.SE.quad.contains(body.pos):
            return self.SE.insert(body)
        return False

    def is_external(self):
        return self.NW is None and self.NE is None and self.SW is None and self.SE is None

    def compute_force(self, body: Body, theta=THETA):
        force = np.zeros(2, dtype=float)
        if self.total_mass == 0.0:
            return force
        # avoid self-interaction
        if self.is_external() and self.body is body:
            return force

        s = self.quad.w * 2.0  # width of region
        dvec = self.center_of_mass - body.pos
        dist = np.linalg.norm(dvec) + SOFTENING

        if self.is_external() or (s / dist) < theta:
            # treat as single body (center of mass)
            if dist > 0.0:
                force_dir = dvec / dist
                force_mag = (G * body.mass * self.total_mass) / (dist * dist)
                force = force_dir * force_mag
            return force
        else:
            # recurse
            if self.NW: force += self.NW.compute_force(body, theta)
            if self.NE: force += self.NE.compute_force(body, theta)
            if self.SW: force += self.SW.compute_force(body, theta)
            if self.SE: force += self.SE.compute_force(body, theta)
            return force


# ---------- Renderer & App ----------
class NBodyApp(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(width=kwargs.get('width', WINDOW_SIZE[0]),
                         height=kwargs.get('height', WINDOW_SIZE[1]),
                         caption='N-Body Gravity Cinematic', resizable=True)
        # create moderngl context from pyglet
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        # additive blending for glow by default when drawing particles
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

        # particle program (vertex + fragment)
        self.prog_particle = self.ctx.program(
            vertex_shader='''#version 330
            in vec2 in_pos;
            in float in_size;
            in vec3 in_color;
            uniform mat3 u_matrix;
            out vec3 v_color;
            out float v_size;
            void main() {
                vec3 p = u_matrix * vec3(in_pos, 1.0);
                gl_Position = vec4(p.xy, 0.0, 1.0);
                v_color = in_color;
                v_size = in_size;
                gl_PointSize = in_size;
            }
            ''',
            fragment_shader='''#version 330
            in vec3 v_color;
            in float v_size;
            out vec4 f_color;
            void main() {
                // circular soft point
                vec2 c = gl_PointCoord - vec2(0.5);
                float r = length(c);
                // alpha falls off toward edge; 0 at r>=0.5
                float alpha = 1.0 - smoothstep(0.0, 0.5, r);
                alpha = pow(alpha, 0.7);
                f_color = vec4(v_color, alpha);
            }
            '''
        )

        # fullscreen quad vertex shader (shared)
        quad_vs_src = '''#version 330
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            v_uv = in_uv;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        '''

        # blit program (just samples texture)
        self.prog_blit = self.ctx.program(
            vertex_shader=quad_vs_src,
            fragment_shader='''#version 330
            uniform sampler2D Texture;
            in vec2 v_uv;
            out vec4 f_color;
            void main(){
                vec4 t = texture(Texture, v_uv);
                f_color = vec4(t.rgb, 1.0);
            }
            '''
        )

        # blur program (separable-ish)
        self.prog_blur = self.ctx.program(
            vertex_shader=quad_vs_src,
            fragment_shader='''#version 330
            uniform sampler2D Texture;
            uniform vec2 texel;
            in vec2 v_uv;
            out vec4 f_color;
            void main(){
                vec3 sum = vec3(0.0);
                float w0 = 0.40;
                float w1 = 0.25;
                float w2 = 0.10;
                sum += texture(Texture, v_uv).rgb * w0;
                sum += texture(Texture, v_uv + vec2(texel.x, 0.0)).rgb * w1;
                sum += texture(Texture, v_uv - vec2(texel.x, 0.0)).rgb * w1;
                sum += texture(Texture, v_uv + vec2(0.0, texel.y)).rgb * w1;
                sum += texture(Texture, v_uv - vec2(0.0, texel.y)).rgb * w1;
                sum += texture(Texture, v_uv + vec2(texel.x*2.0, 0.0)).rgb * w2;
                sum += texture(Texture, v_uv - vec2(texel.x*2.0, 0.0)).rgb * w2;
                sum += texture(Texture, v_uv + vec2(0.0, texel.y*2.0)).rgb * w2;
                sum += texture(Texture, v_uv - vec2(0.0, texel.y*2.0)).rgb * w2;
                f_color = vec4(sum, 1.0);
            }
            '''
        )

        # small fade shader to decay trails (outputs a constant dark color with given alpha)
        self.prog_fade = self.ctx.program(
            vertex_shader=quad_vs_src,
            fragment_shader='''#version 330
            uniform float u_alpha;
            in vec2 v_uv;
            out vec4 f_color;
            void main(){
                // render a translucent black quad to slightly darken (fade) the framebuffer
                f_color = vec4(0.0, 0.0, 0.0, u_alpha);
            }
            '''
        )

        # fullscreen quad vbo (pos.xy, uv.xy)
        quad = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        # create a VAO for blit/quad-based passes
        self.quad_vao_blit = self.ctx.vertex_array(self.prog_blit, [(self.quad_vbo, '2f 2f', 'in_pos', 'in_uv')])
        # For fade program we only need positions; skip the UV floats in the buffer layout
        self.quad_vao_fade = self.ctx.vertex_array(self.prog_fade, [(self.quad_vbo, '2f 2x4', 'in_pos')])

        # framebuffers for trails + blur (created/resized in ensure_framebuffers)
        self.trail_tex = None
        self.trail_fbo = None
        self.blur_tex = None
        self.blur_fbo = None
        self.ensure_framebuffers()

        # particle buffer placeholders
        self.max_particles = MAX_PARTICLES
        self.pos = np.zeros((self.max_particles, 2), dtype='f4')
        self.vel = np.zeros((self.max_particles, 2), dtype='f4')
        self.mass = np.zeros((self.max_particles,), dtype='f4')
        self.color = np.zeros((self.max_particles, 3), dtype='f4')
        self.particle_sizes = np.zeros((self.max_particles,), dtype='f4')
        self.count = 0

        self.spawn_initial(INIT_PARTICLES)

        # create GL buffer for particles (interleaved buffers are an option; keep simple)
        self.vbo_pos = self.ctx.buffer(reserve=self.max_particles * (2 * 4))
        self.vbo_size = self.ctx.buffer(reserve=self.max_particles * 4)
        self.vbo_color = self.ctx.buffer(reserve=self.max_particles * (3 * 4))
        self.vao = self.ctx.vertex_array(self.prog_particle, [
            (self.vbo_pos, '2f', 'in_pos'),
            (self.vbo_size, '1f', 'in_size'),
            (self.vbo_color, '3f', 'in_color'),
        ])

        # camera transform (simple 2D)
        self.cam_pos = np.array([0.0, 0.0], dtype=float)
        self.cam_zoom = 1.0

        # interaction
        self.mouse_down = False
        self.right_down = False
        self.last_mouse = (0, 0)
        self.slow_motion = False
        self.trails = True

        # performance
        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # schedule update
        pyglet.clock.schedule_interval(self.update, DT)

    def ensure_framebuffers(self):
        sx, sy = self.get_framebuffer_size()
        # create or recreate textures/framebuffers if missing or size changed
        if self.trail_tex is None or self.trail_tex.size != (sx, sy):
            if self.trail_tex is not None:
                try:
                    self.trail_tex.release()
                except Exception:
                    pass
            if self.trail_fbo is not None:
                try:
                    self.trail_fbo.release()
                except Exception:
                    pass
            if self.blur_tex is not None:
                try:
                    self.blur_tex.release()
                except Exception:
                    pass
            if self.blur_fbo is not None:
                try:
                    self.blur_fbo.release()
                except Exception:
                    pass

            # floating point textures for nicer bloom
            self.trail_tex = self.ctx.texture((sx, sy), 3, dtype='f1')
            self.trail_fbo = self.ctx.framebuffer(color_attachments=[self.trail_tex])

            self.blur_tex = self.ctx.texture((sx, sy), 3, dtype='f1')
            self.blur_fbo = self.ctx.framebuffer(color_attachments=[self.blur_tex])

    def spawn_initial(self, n):
        for _ in range(n):
            self.spawn_random()

    def spawn_random(self, pos=None, vel=None, mass=None, color=None):
        if self.count >= self.max_particles:
            return
        i = self.count
        if pos is None:
            pos = np.array([random.uniform(-700, 700), random.uniform(-400, 400)], dtype='f4')
        if vel is None:
            vel = np.array([random.uniform(-20, 20), random.uniform(-20, 20)], dtype='f4')
        if mass is None:
            mass = random.uniform(0.8, 6.0)
        if color is None:
            # color biased by mass
            h = random.random()
            s = 0.7
            v = 0.9 - 0.2 * (mass / 6.0)
            color = self.hsv_to_rgb(h, s, v)

        self.pos[i] = pos
        self.vel[i] = vel
        self.mass[i] = mass
        self.color[i] = color
        self.particle_sizes[i] = 2.0 + mass * 1.8
        self.count += 1

    def hsv_to_rgb(self, h, s, v):
        # simple hsv to rgb
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return (r, g, b)

    def get_framebuffer_size(self):
        w, h = self.get_size()
        return int(max(1, w)), int(max(1, h))

    def update(self, dt):
        # physics update using Barnes-Hut
        if self.count == 0:
            return

        # compute bounding box
        minx = np.min(self.pos[:self.count, 0])
        maxx = np.max(self.pos[:self.count, 0])
        miny = np.min(self.pos[:self.count, 1])
        maxy = np.max(self.pos[:self.count, 1])
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        span = max(maxx - minx, maxy - miny) * 1.1 + 1.0
        root = BHTree(Quad(cx, cy, span / 2.0))

        # insert bodies into BH tree
        for i in range(self.count):
            b = Body(self.pos[i].copy(), self.vel[i].copy(), float(self.mass[i]), tuple(self.color[i]))
            root.insert(b)

        # compute forces & integrate
        actual_dt = dt * (0.2 if self.slow_motion else 1.0)
        forces = np.zeros((self.count, 2), dtype='f4')
        for i in range(self.count):
            b = Body(self.pos[i].copy(), self.vel[i].copy(), float(self.mass[i]), tuple(self.color[i]))
            f = root.compute_force(b)
            # acceleration = F / m
            forces[i] = f / b.mass if b.mass != 0.0 else 0.0

        # symplectic Euler (semi-implicit)
        self.vel[:self.count] += forces * actual_dt
        self.pos[:self.count] += self.vel[:self.count] * actual_dt

        # bounds wrapping (for visuals)
        R = 3000.0
        for i in range(self.count):
            x, y = self.pos[i]
            if x < -R:
                self.pos[i, 0] = R
            if x > R:
                self.pos[i, 0] = -R
            if y < -R:
                self.pos[i, 1] = R
            if y > R:
                self.pos[i, 1] = -R

        # mouse influence
        if self.mouse_down or self.right_down:
            mx, my = self.last_mouse
            world = self.screen_to_world(mx, my)
            for i in range(self.count):
                d = world - self.pos[i]
                r2 = d[0] * d[0] + d[1] * d[1] + 1e-6
                dist = math.sqrt(r2)
                if dist == 0.0:
                    continue
                fdir = d / dist
                strength = 2000.0 / r2
                if self.mouse_down:
                    self.vel[i] += fdir * strength * actual_dt
                if self.right_down:
                    self.vel[i] -= fdir * strength * actual_dt

        # update GL buffers
        self.vbo_pos.write(self.pos[:self.count].astype('f4').tobytes())
        self.vbo_size.write(self.particle_sizes[:self.count].astype('f4').tobytes())
        self.vbo_color.write(self.color[:self.count].astype('f4').tobytes())

        # fps
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 0.5:
            self.fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now

    def on_draw(self):
        self.ensure_framebuffers()
        sx, sy = self.get_framebuffer_size()

        # 1) fade previous trails slightly (draw translucent black over trail fbo)
        self.trail_fbo.use()
        # enable alpha blending for fade
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        # draw translucent black to fade old trails (small alpha)
        self.prog_fade['u_alpha'].value = 0.06 if self.trails else 1.0  # if trails off, clear fully
        self.quad_vao_fade.render(mode=moderngl.TRIANGLE_STRIP)

        # 2) render particles additively onto trail_fbo
        self.trail_fbo.use()
        # additive blending for glowing particles
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        # compute camera matrix
        mat = self.compute_matrix()
        # write 3x3 matrix as 9 floats (flatten column-major is fine as GL expects row-major; modernGL handles)
        self.prog_particle['u_matrix'].write(mat.astype('f4').tobytes())
        # draw points
        # use point primitives
        self.vao.render(mode=moderngl.POINTS, vertices=self.count)

        # 3) blur pass: trail_tex -> blur_fbo (horizontal-ish)
        self.trail_tex.use(0)
        self.blur_fbo.use()
        texel = (1.0 / sx, 1.0 / sy)
        self.prog_blur['texel'].value = texel
        self.prog_blur['Texture'].value = 0
        self.quad_vao_blit.render(mode=moderngl.TRIANGLE_STRIP)

        # 4) blur back: blur_tex -> trail_fbo (vertical-ish)
        self.blur_tex.use(0)
        self.trail_fbo.use()
        self.prog_blur['texel'].value = texel
        self.prog_blur['Texture'].value = 0
        self.quad_vao_blit.render(mode=moderngl.TRIANGLE_STRIP)

        # 5) final composite to screen: draw trail texture to screen
        self.ctx.screen.use()
        self.ctx.clear(0.02, 0.02, 0.03)
        self.trail_tex.use(0)
        self.prog_blit['Texture'].value = 0
        # use standard alpha blending
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.quad_vao_blit.render(mode=moderngl.TRIANGLE_STRIP)

        # hud overlay via pyglet
        self.draw_hud()

    def compute_matrix(self):
        # build simple 3x3 matrix for 2D world -> clip space
        w, h = self.get_size()
        sx = 2.0 / (w * self.cam_zoom)
        sy = 2.0 / (h * self.cam_zoom)
        tx = - (self.cam_pos[0]) * sx
        ty = - (self.cam_pos[1]) * sy
        # matrix maps world (x,y) to clip coords
        mat = np.array([
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0]
        ], dtype='f4')
        return mat

    def draw_hud(self):
        fps_text = f"FPS: {self.fps:.1f}  Bodies: {self.count}  G: {G:.1f}  Trails: {self.trails}"
        label = pyglet.text.Label(fps_text,
                                  font_name='Consolas', font_size=12,
                                  x=10, y=self.height - 20,
                                  color=(255, 255, 255, 255))
        label.draw()

    # ---------- input handlers ----------
    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.mouse_down = True
        if button == mouse.RIGHT:
            self.right_down = True
        self.last_mouse = (x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.mouse_down = False
        if button == mouse.RIGHT:
            self.right_down = False
        self.last_mouse = (x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.last_mouse = (x, y)
        # spawn particles while dragging with middle button or ctrl
        if buttons & mouse.MIDDLE:
            for _ in range(3):
                world = self.screen_to_world(x + random.uniform(-5, 5), y + random.uniform(-5, 5))
                vel = np.array([dx * 5.0, dy * 5.0], dtype='f4')
                self.spawn_random(pos=world, vel=vel)

    def on_key_press(self, symbol, modifiers):
        global G
        if symbol == key.SPACE:
            self.slow_motion = not self.slow_motion
        if symbol == key.UP:
            self.spawn_random()
        if symbol == key.DOWN:
            if self.count > 0:
                self.count -= 1
        if symbol == key.PLUS or symbol == key.EQUAL:
            G *= 1.1
        if symbol == key.MINUS:
            G *= 0.9
        if symbol == key.T:
            self.trails = not self.trails
        if symbol == key.F:
            self.set_fullscreen(not self.fullscreen)
        if symbol == key.ESCAPE:
            self.close()
        # camera
        if symbol == key.W:
            self.cam_pos[1] += 50.0 / self.cam_zoom
        if symbol == key.S:
            self.cam_pos[1] -= 50.0 / self.cam_zoom
        if symbol == key.A:
            self.cam_pos[0] -= 50.0 / self.cam_zoom
        if symbol == key.D:
            self.cam_pos[0] += 50.0 / self.cam_zoom
        if symbol == key.Q:
            self.cam_zoom *= 1.1
        if symbol == key.E:
            self.cam_zoom *= 0.9

    def screen_to_world(self, sx, sy):
        # invert compute_matrix mapping
        w, h = self.get_size()
        # normalized device coordinates
        nx = (sx / w) * 2.0 - 1.0
        ny = (sy / h) * 2.0 - 1.0
        sx_scale = 2.0 / (w * self.cam_zoom)
        sy_scale = 2.0 / (h * self.cam_zoom)
        tx = - (self.cam_pos[0]) * sx_scale
        ty = - (self.cam_pos[1]) * sy_scale
        # clip -> world
        world_x = (nx - tx) / sx_scale
        world_y = (ny - ty) / sy_scale
        return np.array([world_x, world_y], dtype='f4')


# ---------- run ----------
if __name__ == '__main__':
    win = NBodyApp(width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
    pyglet.app.run()