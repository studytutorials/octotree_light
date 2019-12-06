/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <cstdint>

#include <Eigen/Dense>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif



template<typename T> struct gl;

template<> struct gl<uchar3> {
	static const int format = GL_RGB;
	static const int type = GL_UNSIGNED_BYTE;
};

template<> struct gl<float> {
	static const int format = GL_LUMINANCE;
	static const int type = GL_FLOAT;
};

template<> struct gl<uint8_t> {
	static const int format = GL_LUMINANCE;
	static const int type = GL_UNSIGNED_BYTE;
};

template<> struct gl<uint16_t> {
	static const int format = GL_LUMINANCE;
	static const int type = GL_UNSIGNED_SHORT;
};

template<> struct gl<uint32_t> {
	static const int format = GL_RGBA;
	static const int type = GL_UNSIGNED_BYTE;
};



template<typename T>
void drawit(const T*               scene,
            const Eigen::Vector2i& size) {

  static Eigen::Vector2i last_size (0, 0);
  if (last_size.x() != size.x() || last_size.y() != size.y()) {
    int argc = 1;
    char* argv = (char*) "supereight";
    glutInit(&argc, &argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(size.x(), size.y());
    glutCreateWindow("supereight display");

    last_size = size;
  }

  glClear(GL_COLOR_BUFFER_BIT);

  if (scene != nullptr) {
    glRasterPos2i(-1, 1);
    glPixelZoom(1, -1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, size.x());
    glDrawPixels(size.x(), size.y(), gl<T>::format, gl<T>::type, scene);
  }

  glutSwapBuffers();
}



template<typename A, typename B, typename C, typename D>
void drawthem(const A* scene_1, const Eigen::Vector2i& size_1,
              const B* scene_2, const Eigen::Vector2i& size_2,
              const C* scene_3, const Eigen::Vector2i& size_3,
              const D* scene_4, const Eigen::Vector2i& size_4) {

  constexpr int rows = 2;
  constexpr int cols = 2;
  const int col_width  = size_2.x();
  const int row_height = size_2.y();
  const int width  = cols * col_width;
  const int height = rows * row_height;

  static Eigen::Vector2i last_size (0, 0);
  if (last_size.x() != size_2.x() || last_size.y() != size_2.y()) {
    int argc = 1;
    char* argv = (char*) "supereight";
    glutInit(&argc, &argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("supereight display");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, (GLdouble) width, 0.0, (GLdouble) height);
    glMatrixMode(GL_MODELVIEW);

    last_size = size_2;
  }

  glClear(GL_COLOR_BUFFER_BIT);

  if (scene_1 != nullptr) {
    glRasterPos2i(0 * col_width, 2 * row_height);
    glPixelZoom((float)  col_width  / size_1.x(),
                (float) -row_height / size_1.y());
    glDrawPixels(size_1.x(), size_1.y(), gl<A>::format, gl<A>::type, scene_1);
  }

  if (scene_2 != nullptr) {
    glRasterPos2i(1 * col_width, 2 * row_height);
    glPixelZoom((float)  col_width  / size_2.x(),
                (float) -row_height / size_2.y());
    glDrawPixels(size_2.x(), size_2.y(), gl<B>::format, gl<B>::type, scene_2);
  }

  if (scene_3 != nullptr) {
    glRasterPos2i(0 * col_width, 1 * row_height);
    glPixelZoom((float)  col_width  / size_3.x(),
                (float) -row_height / size_3.y());
    glDrawPixels(size_3.x(), size_3.y(), gl<C>::format, gl<C>::type, scene_3);
  }

  if (scene_4 != nullptr) {
    glRasterPos2i(1 * col_width, 1 * row_height);
    glPixelZoom((float)  col_width  / size_4.x(),
                (float) -row_height / size_4.y());
    glDrawPixels(size_4.x(), size_4.y(), gl<D>::format, gl<D>::type, scene_4);
  }

  glutSwapBuffers();
}

