/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <algorithm>
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

  const Eigen::Vector2i content_size (size);

  // Create a GLUT window if one does not already exist.
  if (glutGetWindow() == 0) {
    int argc = 1;
    char* argv = (char*) "supereight";
    glutInit(&argc, &argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(content_size.x(), content_size.y());
    glutCreateWindow("supereight display");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, content_size.x());

    // Change raster coordinates from [-1, 1] to [0, 1].
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
  }

  // Get the window size and the scaling factor to scale the content to the
  // window.
  const Eigen::Vector2i window_size = Eigen::Vector2i(
      glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  const float width_factor  = (float) window_size.x() / content_size.x();
  const float height_factor = (float) window_size.y() / content_size.y();
  const float factor = std::min(width_factor, height_factor);

  glViewport(0, 0, window_size.x(), window_size.y());

  glClear(GL_COLOR_BUFFER_BIT);

  if (scene != nullptr) {
    // Draw the image at the top left.
    glRasterPos2i(0, 1);
    // Scale the image and flip it up-down.
    glPixelZoom(factor, -factor);
    glDrawPixels(size.x(), size.y(), gl<T>::format, gl<T>::type, scene);
  }

  glutSwapBuffers();
}



template<typename A, typename B, typename C, typename D>
void drawthem(const A* scene_1, const Eigen::Vector2i& size_1,
              const B* scene_2, const Eigen::Vector2i& size_2,
              const C* scene_3, const Eigen::Vector2i& size_3,
              const D* scene_4, const Eigen::Vector2i& size_4) {

  const Eigen::Vector2i grid_size (2, 2);
  const Eigen::Vector2i image_size (size_2);
  const Eigen::Vector2i content_size = grid_size.cwiseProduct(image_size);
  const Eigen::Vector2f grid_step (1.f / grid_size.x(), 1.f / grid_size.y());

  // Create a GLUT window if one does not already exist.
  if (glutGetWindow() == 0) {
    int argc = 1;
    char* argv = (char*) "supereight";
    glutInit(&argc, &argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(content_size.x(), content_size.y());
    glutCreateWindow("supereight display");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Change raster coordinates from [-1, 1] to [0, 1].
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
  }

  // Get the window size and the scaling factor to scale the content to the
  // window.
  const Eigen::Vector2i window_size = Eigen::Vector2i(
      glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  const float width_factor  = (float) window_size.x() / content_size.x();
  const float height_factor = (float) window_size.y() / content_size.y();
  const float factor = std::min(width_factor, height_factor);

  glViewport(0, 0, window_size.x(), window_size.y());

  glClear(GL_COLOR_BUFFER_BIT);

  if (scene_1 != nullptr) {
    glRasterPos2f(0 * grid_step.x(), 2 * grid_step.y());
    glPixelZoom(factor, -factor);
    glDrawPixels(size_1.x(), size_1.y(), gl<A>::format, gl<A>::type, scene_1);
  }

  if (scene_2 != nullptr) {
    glRasterPos2f(1 * grid_step.x(), 2 * grid_step.y());
    glPixelZoom(factor, -factor);
    glDrawPixels(size_2.x(), size_2.y(), gl<B>::format, gl<B>::type, scene_2);
  }

  if (scene_3 != nullptr) {
    glRasterPos2f(0 * grid_step.x(), 1 * grid_step.y());
    glPixelZoom(factor, -factor);
    glDrawPixels(size_3.x(), size_3.y(), gl<C>::format, gl<C>::type, scene_3);
  }

  if (scene_4 != nullptr) {
    glRasterPos2f(1 * grid_step.x(), 1 * grid_step.y());
    glPixelZoom(factor, -factor);
    glDrawPixels(size_4.x(), size_4.y(), gl<D>::format, gl<D>::type, scene_4);
  }

  glutSwapBuffers();
}

