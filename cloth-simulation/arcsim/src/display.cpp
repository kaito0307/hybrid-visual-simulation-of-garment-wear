/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "display.hpp"
#include <GLFW/glfw3.h>

#ifndef NO_OPENGL

#include "bvh.hpp"
#include "geometry.hpp"
#include "io.hpp"
#include "timer.hpp"
#include "opengl.hpp"
#include "physics.hpp"
#include "timer.hpp"
#include "dynamicremesh.hpp"
#include "util.hpp"
#include "sepstrength.hpp"
#include "wear.hpp"
#include <sstream>
#include <json/json.h>

using namespace std;
string obj2png_filename;
extern string outprefix;

extern int frame;
extern Timer fps;

bool stepDebug;

vector<Annotation> Annotation::list;
Pane Pane::panes[3] = { false, false, true };

extern string inprefix;

template <Space s>
void draw_mesh (const Mesh &mesh, bool set_color=false);
void select_lasso_region(const std::vector<std::pair<int, int>>& lasso_path);

int display_mode = 0;
std::vector<DisplayMode> display_modes = {
    {"sigma", 1e4, true},
    {"sigma_bend", 1e4, true},
    {"sep strength", 1, true},
    {"Sp_str", 1, true},
    {"Sp_bend", 1, true},
    {"Cur Strain Energy,", 0.6, true},
    {"Acc Strain Energy", 1e-2, true},
    {"Fric. Work ClothBody", 400, true},
    {"Fric. Work ClothCloth", 30, true},
    {"Fric. Pow. ClothBody", 100, true},
    {"Fric. Pow. ClothCloth", 10, true},
    {"Normal load", 1000, true},
    {"-sentinel-", 0, false}
};

bool HideClothes = false;
bool HideGrid = false;

Pane* Pane::current(int x) {
    if (x < 0) return nullptr;
    int width = 1280;
    int npanes = 0;
    for (int i = 0; i < 3; i++)
        if (panes[i].enabled)
            npanes++;
    if (GLFWwindow* win = glfwGetCurrentContext()) {
        glfwGetWindowSize(win, &width, nullptr);
    }
    // Find which enabled pane x falls into
    int j = 0;
    for (int i = 0; i < 3; i++) {
        if (!panes[i].enabled) continue;
        int x0 = width * j / npanes;
        int x1 = width * (j + 1) / npanes;
        if (x >= x0 && x < x1)
            return &panes[i];
        j++;
    }
    return nullptr;
}

Vec3 Pane::pos(Vert* v) {
	if (this == &world()) return v->node->x;
	if (this == &plastic()) return v->node->y;
	return v->u;
}

void vertex (const Vec2 &x) {
    glVertex2d(x[0], x[1]);
}

void vertex (const Vec3 &x) {
    glVertex3d(x[0], x[1], x[2]);
}

void normal (const Vec3 &n) {
    glNormal3d(n[0], n[1], n[2]);
}

void color (const Vec3 &x) {
    glColor3d(x[0], x[1], x[2]);
}

Vec3 strain_color (const Face *face) {
    Mat3x3 F = deformation_gradient<WS>(face);
    Vec3 l = eigen_values(F.t()*F);
    double s0 = sqrt(l[0]) - 1, s1 = sqrt(l[1]) - 1;
    double tens = clamp(1e2*s0, 0., 0.5), comp = clamp(-1e2*s1, 0., 0.5);
    return Vec3(1-tens, (1-tens)*(1-comp), (1-comp));
}

Vec3 plasticity_color (const Face *face) {
    double s = norm_F(face->Sp_bend)/1000;
    double d = face->damage;
    s = min(s/2, 0.5);
    d = min(d/2, 0.5);
    return Vec3(1-s, (1-s)*(1-d), (1-d));
    // return colormap(trace(face->S_plastic)/500);
}

Vec3 origami_color (const Mat3x3& M) {
    double H = trace(M);
    return 0.9*Vec3(1 + H, 1 - abs(H)/2, 1 - H);
}

inline double matrix_mag(const Mat3x3& M) {
	Vec3 l = eigen_values(M);    
    return sqrt(sq(l[0])+sq(l[1])+sq(l[2])) * sgn(l[0]+l[1]+l[2]);
}

// http://www.sron.nl/~pault/colourschemes.pdf
// [-1,1]
Vec3 red_blue_colorscheme(double v) {
    v=clamp(0.5*(v+1.0),0.0,1.0);
    double v2=sq(v),v3=v2*v,v4=v3*v,v5=v4*v;
    return Vec3 (0.237-2.13*v + 26.92*v2-65.5*v3+63.5*v4-22.36*v5, 
                 sq((0.572+1.524*v-1.811*v2)/(1-0.291*v+0.1574*v2)),
                 1.0/(1.579-4.03*v+12.92*v2-31.4*v3+48.6*v4-23.36*v5));
}

Vec3 debug_color (Face *face, const Vert* vert) {
    double sc = 1.0/display_modes[display_mode].scale;
    switch(display_mode) {
        case 0: { // sigma        	
            compute_ms_data(face);
            Mat3x3 F = deformation_gradient<WS>(face);
    		Mat3x3 G = (F.t()*F - Mat3x3(1)) * 0.5;
    		Mat3x3 sigma = material_model(face, G);
			return red_blue_colorscheme(sc * matrix_mag(sigma));}
		case 1: { // sigma_bend
            Mat3x3 F_bend = Mat3x3(1);
            for (int i=0; i<3; i++) 
                F_bend += face->v[i]->node->curvature * (0.5/3.0 * face->material->fracture_bend_thickness);
            Mat3x3 G_bend = (F_bend.t()*F_bend - Mat3x3(1)) * 0.5;
            Mat3x3 sigma_bend = material_model(face, G_bend);
            return red_blue_colorscheme(sc * matrix_mag(sigma_bend));
        }
        case 2: { // sep strength
			//return red_blue_colorscheme(sc * separation_strength(vert->node,0,false));
            return red_blue_colorscheme(sc * vert->node->sep);
		}
        case 3: {// Sp_str
            double frob = 0;
            Mat3x3 S = face->Sp_str - Mat3x3(1);
            frob += norm2(S.col(0)) + norm2(S.col(1)) + norm2(S.col(2));
            return red_blue_colorscheme(sc * frob);
        }
        case 4: {// Sp_bend
            /*double frob = 0;
            Mat3x3 S = face->Sp_bend;
            frob += norm2(S.col(0)) + norm2(S.col(1)) + norm2(S.col(2));
            return red_blue_colorscheme(sc * frob);*/
            double v = abs(face->adje[0]->theta_ideal) +
                       abs(face->adje[1]->theta_ideal) +
                       abs(face->adje[2]->theta_ideal);
            return red_blue_colorscheme(sc * v);
        }
        case 5: {
            // Stretching energy
            double energy = 0;
            if (face -> wear != nullptr) {
                energy = face -> wear->cur_strain_energy;
            }
            energy /= face -> a; // per area
            return red_blue_colorscheme(sc * energy - 1);
        }
        case 6: {
            // Accumulated energy
            double energy = 0;
            if (face -> wear != nullptr) {
                energy = face -> wear->acc_strain_energy;
            }
            energy /= face -> a; // per area
            return red_blue_colorscheme(sc * energy - 1);
        }
        case 7:
        case 8: {
            double energy = 0;
            int work_idx = display_mode - 7;
            for (size_t i = 0; i < 3; i++) {
                if (face->v[i]->node->wear != nullptr) {
                    energy += face->v[i]->node->wear->f_work[work_idx];
                }
            }
            energy /= face -> a;
            return red_blue_colorscheme(sc * energy - 1);
        }
        default:
        	return Vec3(0);
        case 9:
        case 10: {
            double energy = 0;
            int work_idx = display_mode - 9;
            for (size_t i = 0; i < 3; i++) {
                if (face->v[i]->node->wear != nullptr) {
                    energy += face->v[i]->node->wear->f_power[work_idx];
                }
            }
            energy /= face -> a;
            return red_blue_colorscheme(sc * energy - 1);
        }
        case 11: {
            double load = 0;
            for (size_t i = 0; i < 3; i++) {
                if (face->v[i]->node->wear != nullptr) {
                    Vec3 n = face->v[i]->node->wear->N[ClothBody];
                    load += norm(n);
                }
            }
            load /= face -> a;
            return red_blue_colorscheme(sc * load - 1);
        }
    }
}

void draw_mesh_ms (Mesh &mesh, bool set_color=false) {
    if (set_color)
        glDisable(GL_COLOR_MATERIAL);
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < mesh.faces.size(); i++) {
        // if (i % 256 == 0) {
        //     glEnd();
        //     glBegin(GL_TRIANGLES);
        // }
        Face *face = mesh.faces[i];

        if (set_color) {
            auto frt = debug_color(face, face->v[0]);
            auto bak = frt;
            float front[4] = {(float)frt[0], (float)frt[1], (float)frt[2], 1},
                  back[4] = {(float)bak[0], (float)bak[1], (float)bak[2], 1};
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front);
            glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back);
        }
        normal(normal<MS>(face));
        for (int v = 0; v < 3; v++) {
            vertex(face->v[v]->u);
        }
    }

    glEnd();
    if (set_color)
        glEnable(GL_COLOR_MATERIAL);


    // Draw seam and boundary edges, it looks messy and let's skip it
    // glLineWidth(2);
    // glColor3d(0,0,0);
    // glBegin(GL_LINES);
    // for (size_t i=0; i<mesh.edges.size(); i++) {
    //     if (is_seam_or_boundary(mesh.edges[i])) {
    //         vertex(mesh.edges[i]->n[0]->verts[0]->u);
    //         vertex(mesh.edges[i]->n[1]->verts[0]->u);
    //     }
    // }
    // glEnd();
    // glLineWidth(1);

    // Draw preserved vertices, we don't need it.
    // glPointSize(5);
    // color(Vec3(1,0,0));
    // glBegin(GL_POINTS);
    // for (size_t i = 0; i < mesh.verts.size(); i++) {
    // 	if (mesh.verts[i]->node->preserve)
    // 	vertex(mesh.verts[i]->u);
    // }
    // glEnd();
}

void draw_meshes_ms (bool set_color=false) {
    for (size_t m = 0; m < sim.cloth_meshes.size(); m++)
        draw_mesh_ms(*sim.cloth_meshes[m], set_color);
}

void shrink_face (const Face *face, double shrink_factor, double shrink_max,
                  Vec3 u[3]) {
    Vec3 u0 = face->v[0]->u, u1 = face->v[1]->u, u2 = face->v[2]->u;
    double a = face->a;
    double l = max(norm(u0 - u1), max(norm(u1 - u2), norm(u2 - u0)));
    double h = 2*a/l;
    double dh = min(h*shrink_factor, shrink_max);
    for (int v = 0; v < 3; v++) {
        Vec3 e1 = normalize(face->v[NEXT(v)]->u - face->v[v]->u),
             e2 = normalize(face->v[PREV(v)]->u - face->v[v]->u);
        Vec3 du = (e1 + e2)*dh / norm(cross(e1, e2));
        u[v] = face->v[v]->u + du;
    }
}

void draw_meshes_ms_fancy () {
    double shrink_factor = 0.1, shrink_max = 0.5e-3;
    for (size_t m = 0; m < sim.cloth_meshes.size(); m++) {
        const Mesh &mesh = *sim.cloth_meshes[m];
        glBegin(GL_TRIANGLES);
        glColor3f(0.5,0.5,0.5);
        for (int f = 0; f < (int)mesh.faces.size(); f++) {
            const Face *face = mesh.faces[f];
            for (int v = 0; v < 3; v++)
                vertex(face->v[v]->u);
        }
        glEnd();
        glBegin(GL_TRIANGLES);
        for (int f = 0; f < (int)mesh.faces.size(); f++) {
            const Face *face = mesh.faces[f];
            color(origami_color(face->Sp_bend)/1000.0);
            glColor3f(0.9,0.9,0.9);
            Vec3 u[3];
            shrink_face(face, shrink_factor, shrink_max, u);
            for (int v = 0; v < 3; v++)
                vertex(u[v]);
        }
        glEnd();
    }
}

void draw_mesh_ps (const Mesh &mesh, bool set_color=false) {
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < (int)mesh.faces.size(); i++) {
        Face *face = mesh.faces[i];
        // if (i % 256 == 0) {
        //     glEnd();
        //     glBegin(GL_TRIANGLES);
        // }
        normal(normal<PS>(face));
        for (int v = 0; v < 3; v++)
            vertex(face->v[v]->node->y);
    }
    glEnd();
}

void draw_meshes_ps (bool set_color=false) {
    for (size_t m = 0; m < sim.cloth_meshes.size(); m++)
        draw_mesh_ps(*sim.cloth_meshes[m], set_color);
}

template <Space s>
void draw_annotation(Annotation& a) {
	// glDisable(GL_COLOR_MATERIAL);
	glColor3f(a.color[0],a.color[1],a.color[2]);
	glLineWidth(2);
	glPointSize(6);
	if (a.face) {
		glBegin(GL_TRIANGLES);
		for (int i=0; i<3; i++)
			vertex(pos<s>(a.face->v[i]));
		glEnd();
	} else if (a.edge) {
		glBegin(GL_LINES);
		vertex(pos<s>(a.edge->n[0]->verts[0]));
		vertex(pos<s>(a.edge->n[1]->verts[0]));
		glEnd();
	} else if (a.node) {
	    if (a.node -> verts.size()) {
	        glBegin(GL_POINTS);
	        vertex(pos<s>(a.node->verts[0]));
	        glEnd();
	    }
	} else {
		if (norm(a.dir) > 0) {
			glBegin(GL_LINES);
			vertex(a.pos);
			vertex(a.pos + a.dir);
			glEnd();
		} else {	
			glBegin(GL_POINTS);	
			vertex(a.pos);
			glEnd();
		}
	}
	glLineWidth(1);
    glPointSize(1);
	// glEnable(GL_COLOR_MATERIAL);
}

template <Space s>
void draw_mesh (const Mesh &mesh, bool set_color) {
    if (set_color)
        glDisable(GL_COLOR_MATERIAL);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < (int)mesh.faces.size(); i++) {
        Face *face = mesh.faces[i];
        // if (i % 256 == 0) {
        //     glEnd();
        //     glBegin(GL_TRIANGLES);
        // }
        if (set_color) {
            Vec3 frt, bak;
            // int c = find((Mesh*)&mesh, sim.cloth_meshes);
            // static const float phi = (1+sqrt(5))/2;
            // double hue = c*(2 - phi)*2*M_PI; // golden angle
            // hue = -0.6*M_PI + hue; // looks better this way :/
            // //if (face->label % 2 == 1) hue += M_PI;
            // static Vec3 a = Vec3(0.92, -0.39, 0), b = Vec3(0.05, 0.12, -0.99);
            // Vec3 frt = Vec3(0.7,0.7,0.7) + (a*cos(hue) + b*sin(hue))*0.3,
            //      bak = frt*0.5 + Vec3(0.5,0.5,0.5);

            frt = debug_color(face, face->v[0]);
            bak = frt * 0.5 + Vec3(0.5, 0.5, 0.5);
            float front[4] = {(float)frt[0], (float)frt[1], (float)frt[2], 1},
                  back[4] = {(float)bak[0], (float)bak[1], (float)bak[2], 1};
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front);
            glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back);
        }
        normal(normal<s>(face));
        for (int v = 0; v < 3; v++)
            vertex(pos<s>(face->v[v]));
    }
    glEnd();
    if (set_color)
        glEnable(GL_COLOR_MATERIAL);
}

template <Space s>
void draw_meshes (bool set_color=false) {
    for (size_t m = 0; m < sim.cloth_meshes.size(); m++)
        draw_mesh<s>(*sim.cloth_meshes[m], set_color);
}

template <Space s>
void draw_seam_or_boundary_edges () {
    glColor3f(0,0,0);
    glBegin(GL_LINES);
    for (size_t m = 0; m < sim.cloth_meshes.size(); m++) {
        const Mesh &mesh = *sim.cloth_meshes[m];
        for (int e = 0; e < (int)mesh.edges.size(); e++) {
            const Edge *edge = mesh.edges[e];
            if (!is_seam_or_boundary(edge))
                continue;
            vertex(pos<s>(edge->n[0]));
            vertex(pos<s>(edge->n[1]));
        }
    }
    glEnd();
}

void draw_node_vels () {
    double dt = 0.01;
    glBegin(GL_LINES);
    for (size_t m = 0; m < sim.cloth_meshes.size(); m++) {
        const Mesh &mesh = *sim.cloth_meshes[m];
        for (int n = 0; n < (int)mesh.nodes.size(); n++) {
            const Node *node = mesh.nodes[n];
            glColor3d(0,0,1);
            vertex(node->x);
            vertex(node->x + dt*node->v);
            glColor3d(1,0,0);
            vertex(node->x);
            vertex(node->x - dt*node->v);
        }
    }
    for (size_t o = 0; o < sim.obstacles.size(); o++) {
        const Mesh &mesh = sim.obstacles[o] -> get_mesh();
        for (int n = 0; n < (int)mesh.nodes.size(); n++) {
            const Node *node = mesh.nodes[n];
            glColor3d(0,0,1);
            vertex(node->x);
            vertex(node->x + dt*node->v);
            glColor3d(1,0,0);
            vertex(node->x);
            vertex(node->x - dt*node->v);
        }
    }
    glEnd();
}

void draw_node_accels () {
    double dt2 = 1e-6;
    glBegin(GL_LINES);
    for (size_t m = 0; m < sim.cloth_meshes.size(); m++) {
        const Mesh &mesh = *sim.cloth_meshes[m];
        for (int n = 0; n < (int)mesh.nodes.size(); n++) {
            const Node *node = mesh.nodes[n];
            glColor3d(0,0,1);
            vertex(node->x);
            vertex(node->x + dt2*node->acceleration);
            glColor3d(1,0,0);
            vertex(node->x);
            vertex(node->x - dt2*node->acceleration);
        }
    }
    glEnd();
}

void directional_light (int i, const Vec3 &dir, const Vec3 &dif) {
    float diffuse[4] = {(float)dif[0], (float)dif[1], (float)dif[2], 1};
    float position[4] = {(float)dir[0], (float)dir[1], (float)dir[2], 0};
    glEnable(GL_LIGHT0+i);
    glLightfv(GL_LIGHT0+i, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0+i, GL_POSITION, position);
}

void ambient_light (const Vec3 &a) {
    float ambient[4] = {(float)a[0], (float)a[1], (float)a[2], (float)1};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
}

double aspect_ratio () {
    int width = 1280, height = 720;
    // If using GLFW, get actual window size
    if (GLFWwindow* win = glfwGetCurrentContext()) {
        glfwGetWindowSize(win, &width, &height);
    }
    return (double)width/(double)height;
}

void apply_view (const Pane &view) {
    glTranslatef(view.offset[0], view.offset[1], 0);
    glScalef(view.scale, view.scale, view.scale);
    glRotatef(view.roll, 0, 0, 1);
    glRotatef(view.lat, 1,0,0);
    glRotatef(view.lon, 0,1,0);    
    glTranslatef(view.center[0], view.center[1], view.center[2]);    
}

template<Space s>
void init_view(Pane& view) {
    if (view.initialized) 
        return;
    
    Vec3 c(0);
    int n=0;    
    for (size_t m=0; m<sim.cloth_meshes.size(); m++) {
        const vector<Vert*>& verts = sim.cloth_meshes[m]->verts;
        for (size_t i=0; i<verts.size(); i++) {
            c += pos<s>(verts[i]);
            n++;
        }            
    }
    view.center = Vec3(0);//-c / ((double)n);
    view.initialized = true;
}

void basic_gl_setup(double aspect) {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1,1);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, aspect, 0.001, 1000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -1);
}

void display_material (double aspect) {
    init_view<MS>(Pane::material());
    // glClearColor(1,1,1,1);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    basic_gl_setup(aspect);
    // draw_meshes_ms_fancy();
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    glColor3d(0.9,0.9,0.9);
    directional_light(0, Vec3(0,0,1), Vec3(0.5,0.5,0.5));
    ambient_light(Vec3(0.5));
    apply_view(Pane::material());
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    draw_meshes_ms(true);
    glColor4d(0,0,0, 0.2);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    draw_meshes_ms();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    for (size_t i=0; i<Annotation::list.size(); i++)
        draw_annotation<MS>(Annotation::list[i]);
}

void display_plastic (double aspect) {
    if (!Pane::plastic().enabled)
        return;
    init_view<PS>(Pane::plastic());
    // glClearColor(1,1,1,1);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    basic_gl_setup(aspect);
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    directional_light(0, Vec3(0,0,1), Vec3(0.5,0.5,0.5));
    ambient_light(Vec3(0.5));
    apply_view(Pane::plastic());
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    draw_meshes<PS>(true);
    glColor4d(0,0,0, 0.2);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    draw_meshes<PS>();
    draw_seam_or_boundary_edges<PS>();
}

void display_world (double aspect) {
    init_view<WS>(Pane::world());
    basic_gl_setup(aspect);
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    directional_light(0, Vec3(0,0,1), Vec3(0.5,0.5,0.5));
    ambient_light(Vec3(0.5));
    apply_view(Pane::world());
    if (!::HideClothes) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        draw_meshes<WS>(true);
    }
    glEnable(GL_CULL_FACE);
    glColor3f(0.8,0.8,0.8);
    for(int o = 0; o < (int)sim.obstacles.size(); o++)
        draw_mesh<WS>(sim.obstacles[o] -> get_mesh());
    glDisable(GL_CULL_FACE);
    glColor4d(0,0,0, 0.2);
    if (!::HideGrid) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        draw_meshes<WS>();
        draw_seam_or_boundary_edges<WS>();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        for (size_t i=0; i<Annotation::list.size(); i++)
            draw_annotation<WS>(Annotation::list[i]);
    }

    char title[1024];
    // sprintf(title, "%s frame %d time %.3f  debug: %s [scale %g]",outprefix.c_str(), sim.frame, sim.time,
    //     display_modes[display_mode].name.c_str(), display_modes[display_mode].scale);

    snprintf(title, 1024, "%s frame %d time %.3f  FPS: %.3f debug: %s [scale %g]",outprefix.c_str(), sim.frame, sim.time,
        fps.last == 0 ? 0 : 1 / fps.last, display_modes[display_mode].name.c_str(), display_modes[display_mode].scale);
    glfwSetWindowTitle(glfwGetCurrentContext(), title);
}

// Helper to compute aspect ratio for a pane
static double pane_aspect_ratio(int x0, int x1, int height) {
    return (double)(x1 - x0) / (double)height;
}

template<Space s>
void align_pane_rotation(int pane_index, Mesh *cloth) {
    char config_backup_name[256];
    snprintf(config_backup_name, 256, "%s/%s", inprefix.c_str(), "conf.json");

    Json::Value json;
    Json::Reader reader;
    ifstream file(config_backup_name);
    bool parsingSuccessful = reader.parse(file, json);
    if(parsingSuccessful && json.isMember("extra") && json["extra"].isMember("material_rotation")) {
        Pane::panes[pane_index].roll = -json["extra"]["material_rotation"].asFloat();
    }
    else {
        // Solving the principle direction of the cloth mesh with PCA is abandoned
        // double angle = std::abs(find_principle_direction<s>(cloth)) / M_PI * 180;
        // constexpr int rounding_threshold = 5; // Round to nearest multiple of 3 degrees
        // int rounded = static_cast<int>(std::round(angle / rounding_threshold) * rounding_threshold); // Round to nearest even number
        // Pane::panes[pane_index].roll = rounded - 90;
    }
    std::cout << "[] Material pane rotation set to " << Pane::panes[pane_index].roll << " degrees." << std::endl;
    file.close();
}

void align_material_pane_rotation() {
    align_pane_rotation<MS>(0, sim.cloth_meshes[0]);
}

void set_pane_viewport(int pane_index, int fb_width, int fb_height) {
    if (!Pane::panes[pane_index].enabled) return;

    int npanes = 0;
    int j = 0;
    for (int i = 0; i < 3; i++)
        if (Pane::panes[i].enabled) {
            npanes++;
            if (i < pane_index) j++;
        }
    int x0 = fb_width*j/npanes, x1 = fb_width*(j+1)/npanes;
    glViewport(x0, 0, x1-x0, fb_height);
}


void screen_to_framebuffer(int x, int y, int &xx, int &yy) {
    int win_width, win_height, fb_width, fb_height;
    GLFWwindow* win = glfwGetCurrentContext();
    glfwGetWindowSize(win, &win_width, &win_height);
    glfwGetFramebufferSize(win, &fb_width, &fb_height);
    // Convert window (logical) coordinates to framebuffer (pixel) coordinates
    double scale_x = (double)fb_width / win_width;
    double scale_y = (double)fb_height / win_height;
    int x_fb = (int)(x * scale_x);
    int y_fb = (int)(y * scale_y);
    int y_flipped = fb_height - y_fb;
    xx = x_fb;
    yy = y_flipped;
}

struct MouseState {
    bool down;
    int x, y;
    int down_x, down_y;
    enum {ROTATE, TRANSLATE, SCALE, LASSO} func;
    bool lasso_mode = false;
    std::vector<std::pair<int, int>> lasso_path;

    void add_lasso_point(int x, int y) {
        int xx, yy;
        screen_to_framebuffer(x, y, xx, yy);
        lasso_path.emplace_back(xx, yy);
    }
} mouse_state;

void zoom (bool in, int mouse_x) {
    Pane* pane = Pane::current(mouse_x);
    if (!pane) return;
    if (in)
        pane->scale *= 1.2;
    else
        pane->scale /= 1.2;
    redisplay();
}

// click on element for debug information
void select_element(int x, int y, int button, int mods) {
    int xx, yy;
    screen_to_framebuffer(x, y, xx, yy);
    Pane* pane = Pane::current(x);
    double modelview[16], project[16];
    int viewport[4];
    Vec3 p0,p1;
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, project);
    glGetIntegerv(GL_VIEWPORT, viewport);
    // get 3d ray
    gluUnProject(xx, yy, 0, modelview, project, viewport, &p0[0], &p0[1], &p0[2]);
    gluUnProject(xx, yy, 1, modelview, project, viewport, &p1[0], &p1[1], &p1[2]);
    Face *face = 0; Vert *vert = 0;
    double maxZ = 1e100;
    double minD = 0.03;
    for (size_t c=0; c<sim.cloth_meshes.size(); c++) {
        Mesh& mesh = *sim.cloth_meshes[c];
        if (button == GLFW_MOUSE_BUTTON_LEFT && !(mods & GLFW_MOD_SHIFT)) {
            for (size_t t=0; t<mesh.faces.size(); t++) {
                Face* cur = mesh.faces[t];
                double z;
                if (triangle_ray_test(pane->pos(cur->v[0]), pane->pos(cur->v[1]),
                                      pane->pos(cur->v[2]), p0, p1-p0, z)) {
                    if (fabs(z)<maxZ) {
                        maxZ = fabs(z);
                        face = cur;
                    }
                }
            }
        } else if(button == GLFW_MOUSE_BUTTON_RIGHT || (button == GLFW_MOUSE_BUTTON_LEFT && (mods & GLFW_MOD_SHIFT))) {
            for (size_t v=0; v<mesh.verts.size(); v++) {
                Vert* cur = mesh.verts[v];
                Vec3 ax = pane->pos(cur);
                double d = norm(cross(ax-p0,ax-p1))/norm(p1-p0);
                if (d < minD) {
                    minD = d;
                    vert = cur;
                }
            }
        }
	}
	Annotation::list.clear();
	if (face) {
		Annotation::add(face);
        cout << face << " index " << face->index << endl;
        map<Node*,Plane> planes;
        compute_face_sizing(face->v[0]->node->mesh->parent->remeshing,face, planes, true);
        Vec3 eig = eigen_values(face->sigma);
        cout << "max sigma " << eig[0] << " (toughness " << face->material->toughness << ")" << endl;        
        cout << "aspect " << aspect(face) << endl;
        cout << "Sp_str " << face->Sp_str << endl;
        cout << "theta ideal " << face->adje[0]->theta_ideal << " " << face->adje[1]->theta_ideal << " " << face->adje[2]->theta_ideal << endl;
	    if (face -> wear != nullptr) cout << "wear: " << face -> wear->cur_strain_energy << endl;
	}
	if (vert) {
		Annotation::add(vert->node);
		cout << vert << " index " << vert->index << endl;
		cout << "preserve " << vert->node->preserve << " flags " << vert->node->flag << " label " 
             << vert->node->label << endl;
        cout << "sep " << separation_strength(vert->node,0,true) << endl;
        cout << "sizing " << vert->sizing << endl;
        cout << "x " << vert->node->x << endl;
        cout << "u " << vert->u << endl;
        cout << "vel " << vert->node->v << endl;
        cout << "acc " << vert->node->acceleration << endl;
        for (size_t i=0; i<vert->adjf.size(); i++)
            cout << vert->adjf[i] << " ";
        cout << endl;
	    if (vert -> node -> wear != nullptr) {
            cout << "accumulated frictional work: " << vert -> node -> wear->f_work << endl;
        } else {
            cout << "no wear data for this node" << endl;
        }
	}
	redisplay();
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    int x = static_cast<int>(xpos);
    int y = static_cast<int>(ypos);
    mouse_state.down = (action == GLFW_PRESS);
    mouse_state.x = x;
    mouse_state.y = y;

    if (button == GLFW_MOUSE_BUTTON_LEFT && (mods & GLFW_MOD_ALT) && action == GLFW_PRESS) {
        mouse_state.lasso_mode = true;
        mouse_state.func = MouseState::LASSO;
        mouse_state.lasso_path.clear();
        mouse_state.add_lasso_point(x, y);
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE && mouse_state.lasso_mode) {
        mouse_state.add_lasso_point(x, y);
        select_lasso_region(mouse_state.lasso_path); // To be implemented
        mouse_state.lasso_mode = false;
        mouse_state.func = MouseState::ROTATE;
        return;
    }

    if (action == GLFW_RELEASE && abs(mouse_state.down_x-x) < 2 && abs(mouse_state.down_y-y) < 2 &&
        (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT))
        select_element(x, y, button, mods);
    mouse_state.down_x = x;
    mouse_state.down_y = y;
    Pane* pane = Pane::current(x);
    if (!pane) return;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE || (button == GLFW_MOUSE_BUTTON_LEFT && (mods & GLFW_MOD_SHIFT))) {
        mouse_state.func = MouseState::TRANSLATE;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouse_state.func = MouseState::ROTATE;
    }
    // Mouse wheel handled in scroll callback
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    int x = static_cast<int>(xpos);
    int y = static_cast<int>(ypos);
    int y_flipped = height - y;
    if (mouse_state.lasso_mode && mouse_state.down) {
        mouse_state.add_lasso_point(x, y);
        mouse_state.x = x;
        mouse_state.y = y;
        return;
    }
    if (!mouse_state.down)
        return;
    Pane* pane = Pane::current(x);
    if (!pane) return;
    if (mouse_state.func == MouseState::ROTATE) {
        double speed = 0.25;
        pane->lon += (x - mouse_state.x)*speed;
        pane->lat += (y - mouse_state.y)*speed;
    } else if (mouse_state.func == MouseState::TRANSLATE) {
        double speed = 1e-3;
        pane->offset[0] += (x - mouse_state.x)*speed;
        pane->offset[1] -= (y - mouse_state.y)*speed;
    }
    mouse_state.x = x;
    mouse_state.y = y;
    redisplay();
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    int x = static_cast<int>(xpos);
    int y = static_cast<int>(ypos);
    int y_flipped = height - y;
    Pane* pane = Pane::current(x);
    if (!pane) return;
    double oldScale = pane->scale;
    if (yoffset > 0) {
        pane->scale *= 1.02;
    } else if (yoffset < 0) {
        pane->scale /= 1.02;
    }
    double s = pane->scale/oldScale;
    Vec2 mvVec = Vec2((double)x/width-0.5, (double)-y_flipped/height+0.5);
    pane->offset = s * pane->offset + (1-s)*aspect_ratio()*mvVec;
    redisplay();
}

// Helper: project a 3D point to framebuffer pixel coordinates (retina-aware, pane-aware)
bool project_to_framebuffer_pane(const Vec3& pos, int& x_fb, int& y_fb) {
    double modelview[16], projection[16];
    int viewport[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetIntegerv(GL_VIEWPORT, viewport);
    double winX, winY, winZ;
    if (!gluProject(pos[0], pos[1], pos[2], modelview, projection, viewport, &winX, &winY, &winZ))
        return false;
    x_fb = (int)(winX);
    y_fb = (int)(winY);
    return true;
}

bool point_in_polygon(int x, int y, const std::vector<std::pair<int, int>>& polygon) {
    // Ray-casting algorithm to check if point (x, y) is inside the polygon
    bool inside = false;
    size_t n = polygon.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((polygon[i].second > y) != (polygon[j].second > y)) &&
            (x < (polygon[j].first - polygon[i].first) * (y - polygon[i].second) / (polygon[j].second - polygon[i].second) + polygon[i].first)) {
            inside = !inside;
        }
    }
    return inside;
}

template<Space S>
void select_lasso_region(const std::vector<std::pair<int, int>>& lasso_path) {
    if (lasso_path.size() < 3) return;

    double total_damage = 0;

    std::vector<int> face_inx;

    for (size_t c = 0; c < sim.cloth_meshes.size(); c++) {
        Mesh& mesh = *sim.cloth_meshes[c];
        // Select areas (faces) if all their nodes are inside
        for (Face *face : mesh.faces) {
            bool all_inside = true;
            for (int v = 0; v < 3; v++) {
                int x_fb, y_fb;
                Vec3 p = pos<S>(face->v[v]);
                if (!project_to_framebuffer_pane(p, x_fb, y_fb)) {
                    all_inside = false;
                    continue;
                }
                if (point_in_polygon(x_fb, y_fb, lasso_path)) {
                    Annotation::add(face -> v[v] -> node);
                    total_damage += face -> v[v] -> node -> wear ? face -> v[v] -> node -> wear -> f_work[ClothBody] : 0;
                }
                else all_inside = false;
            }
            if (all_inside) {
                Annotation::add(face);
                face_inx.push_back(face -> index);
            }
        }
    }

    std::cout << "Total damage in lasso region: " << total_damage << std::endl;
    if (face_inx.empty()) {
        std::cout << "No faces selected in lasso region." << std::endl;
    } else {
        std::cout << "Selected faces: ";
        for (int index : face_inx) {
            std::cout << index << ", ";
        }
        std::cout << std::endl;
    }
}

// Lasso selection: select all nodes and areas within the lasso region, pane-aware
void select_lasso_region(const std::vector<std::pair<int, int>>& lasso_path) {
    if (lasso_path.size() < 3) return;
    int fb_width, fb_height;
    GLFWwindow* win = glfwGetCurrentContext();
    glfwGetFramebufferSize(win, &fb_width, &fb_height);

    int npanes = 0;
    for (int i = 0; i < 3; i++)
        if (Pane::panes[i].enabled)
            npanes++;

    double aspect = (double)fb_width / (double)fb_height / npanes;


    Annotation::list.clear();
    for (int i = 0; i < 3; i++) {
        if (!Pane::panes[i].enabled) continue;
        set_pane_viewport(i, fb_width, fb_height);
        basic_gl_setup(aspect);
        if (i == 0) {
            // Material pane
            apply_view(Pane::material());
            select_lasso_region<MS>(lasso_path);
        } else if (i == 1) {
            // Plastic pane
            apply_view(Pane::plastic());
            select_lasso_region<PS>(lasso_path);
        } else if (i == 2) {
            // World pane
            apply_view(Pane::world());
            select_lasso_region<WS>(lasso_path);
        }
    }
    redisplay();
}

// Draw the lasso path (in framebuffer pixel coordinates, overlay)
void draw_lasso_path(const std::vector<std::pair<int, int>>& lasso_path) {
    if (lasso_path.size() < 2) return;
    int fb_width, fb_height;
    GLFWwindow* win = glfwGetCurrentContext();
    glfwGetFramebufferSize(win, &fb_width, &fb_height);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, fb_width, 0, fb_height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glColor3f(1, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINE_STRIP);
    for (const auto& p : lasso_path) {
        glVertex2i(p.first, p.second);
    }
    glEnd();
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void redisplay () {
    int width = 1280, height = 720;
    int fb_width = width, fb_height = height;
    GLFWwindow* win = glfwGetCurrentContext();
    glfwGetWindowSize(win, &width, &height);
    glfwGetFramebufferSize(win, &fb_width, &fb_height);

    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int npanes = 0;
    for (int i = 0; i < 3; i++)
        if (Pane::panes[i].enabled)
            npanes++;

    double aspect = (double)fb_width / (double)fb_height / npanes;

    for (int i = 0; i < 3; i++) {
        if (!Pane::panes[i].enabled) continue;
        set_pane_viewport(i, fb_width, fb_height);
        if (i == 0) display_material(aspect);
        else if (i == 1) display_plastic(aspect);
        else if (i == 2) display_world(aspect);
    }

    // Draw lasso path as overlay (always on top)
    if (mouse_state.lasso_mode && !mouse_state.lasso_path.empty()) {
        glViewport(0, 0, fb_width, fb_height); // Fullscreen overlay
        draw_lasso_path(mouse_state.lasso_path);
    }
}

void display_mode_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
        ::display_mode = key - GLFW_KEY_0;
    } else if (key == GLFW_KEY_Q) {
        ::display_mode = 10;
    } else if (key == GLFW_KEY_W) {
        ::display_mode = 11;
    }else if (key == GLFW_KEY_EQUAL) {
        if (display_mode >= 0 && display_mode < display_modes.size())
            ::display_modes[display_mode].scale *= 1.2;
    } else if (key == GLFW_KEY_MINUS) {
        if (display_mode >= 0 && display_mode < display_modes.size())
            ::display_modes[display_mode].scale *= 0.8;
    }
    else if (key == GLFW_KEY_H) ::HideClothes = !::HideClothes;
    else if (key == GLFW_KEY_G) ::HideGrid = !::HideGrid;
}

void display_view_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (key == GLFW_KEY_R) align_pane_rotation<MS>(0, sim.cloth_meshes[0]);
}

static std::vector<std::function<void(GLFWwindow*, int, int, int, int)>> key_callbacks;
void key_callback_dispatcher(GLFWwindow* window, int key, int scancode, int action, int mods) {
    for (auto& cb : key_callbacks) {
        cb(window, key, scancode, action, mods);
    }
}

void add_key_callback(std::function<void(GLFWwindow*, int, int, int, int)> cb)
{
    key_callbacks.push_back(cb);
}


void prepare_callbacks() {
    GLFWwindow* window = glfwGetCurrentContext();
    key_callbacks.push_back(display_mode_key_callback);
    key_callbacks.push_back(display_view_key_callback);
    glfwSetKeyCallback(window, key_callback_dispatcher);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
}


void wait_key () {
    redisplay();
    stepDebug = true;
    GLFWwindow* window = glfwGetCurrentContext();
    while (stepDebug && window && !glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // Optionally, add a small sleep here to avoid busy-waiting
    }
    Annotation::list.clear();
}

#endif // NO_OPENGL
