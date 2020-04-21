/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 Copyright 2016 Emanuele Vespa, Imperial College London

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors
 may be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _COMMONS_
#define _COMMONS_

#if defined(__GNUC__)
// circumvent packaging problems in gcc 4.7.0
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

// need c headers for __int128 and uint16_t
#include <limits.h>
#endif
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <iterator>
#include <set>
#include <vector>

// Internal dependencies
#include "se/utils/math_utils.h"
#include "se/constant_parameters.h"

//External dependencies
#undef isnan
#undef isfinite
////////////////////////// MATh STUFF //////////////////////
//

#define INVALID -2
// DATA TYPE
//

inline
bool is_file(std::string path) {
	struct stat buf;
	stat(path.c_str(), &buf);
	return S_ISREG(buf.st_mode);
}

template<typename T>
std::string NumberToString(T Number, int width = 6) {
	std::ostringstream ss;
	ss << std::setfill('0') << std::setw(width) << Number;
	return ss.str();
}

template<typename T>
void read_input(std::string inputfile, T * in) {
	size_t isize;
	std::ifstream file(inputfile.c_str(),
			std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()) {
		isize = file.tellg();
		file.seekg(0, std::ios::beg);
		file.read((char*) in, isize);
		file.close();
	} else {
		std::cout << "File opening failed : " << inputfile << std::endl;
		exit(1);
	}
}

inline void gs2rgb(double h, unsigned char rgba[4]) {
	double v = 0;
	double r = 0, g = 0, b = 0;
	v = 0.75;
	if (v > 0) {
		double m;
		double sv;
		int sextant;
		double fract, vsf, mid1, mid2;
		m = 0.25;
		sv = 0.6667;
		h *= 6.0;
		sextant = (int) h;
		fract = h - sextant;
		vsf = v * sv * fract;
		mid1 = m + vsf;
		mid2 = v - vsf;
		switch (sextant) {
		case 0:
			r = v;
			g = mid1;
			b = m;
			break;
		case 1:
			r = mid2;
			g = v;
			b = m;
			break;
		case 2:
			r = m;
			g = v;
			b = mid1;
			break;
		case 3:
			r = m;
			g = mid2;
			b = v;
			break;
		case 4:
			r = mid1;
			g = m;
			b = v;
			break;
		case 5:
			r = v;
			g = m;
			b = mid2;
			break;
		default:
			r = 0;
			g = 0;
			b = 0;
			break;
		}
	}
	rgba[0] = r * 255;
	rgba[1] = g * 255;
	rgba[2] = b * 255;
	rgba[3] = 255; // Only for padding purposes
}

typedef struct Triangle {
  Eigen::Vector3f vertexes[3];
  Eigen::Vector3f vnormals[3];
  Eigen::Vector3f normal;
  float color;
  float surface_area;

  Triangle(){
    vertexes[0] = Eigen::Vector3f::Constant(0);
    vertexes[1] = Eigen::Vector3f::Constant(0);
    vertexes[2] = Eigen::Vector3f::Constant(0);
    normal = Eigen::Vector3f::Constant(0);
    surface_area = -1.f;
  }

  inline bool iszero(const Eigen::Vector3f& v){
    return !(v.array() == 0).all();
  }

  inline bool valid(){
    return !(iszero(vertexes[0]) && iszero(vertexes[1]) && iszero(vertexes[2]));
  }

  inline void compute_normal(){
    normal = (vertexes[1] - vertexes[0]).cross(vertexes[2] - vertexes[1]);
  }

  inline void compute_boundingbox(Eigen::Vector3f& minV, Eigen::Vector3f& maxV) const {
    minV = vertexes[0];
    maxV = vertexes[0];
    minV = minV.cwiseMin(vertexes[0]);
    minV = minV.cwiseMin(vertexes[1]);
    minV = minV.cwiseMin(vertexes[2]);
    maxV = maxV.cwiseMax(vertexes[0]);
    maxV = maxV.cwiseMax(vertexes[1]);
    maxV = maxV.cwiseMax(vertexes[2]);
  }

  inline float area() {
    // Use the cached value if available
    if(surface_area > 0) return surface_area;
    Eigen::Vector3f a = vertexes[1] - vertexes[0];
    Eigen::Vector3f b = vertexes[2] - vertexes[1];
    Eigen::Vector3f v = a.cross(b);
    surface_area = v.norm() / 2;
    return surface_area;
  }

  Eigen::Vector3f* uniform_sample(int num){

    Eigen::Vector3f* points = new Eigen::Vector3f[num];
    for(int i = 0; i < num; ++i){
      float u = ((float)rand()) / (float)RAND_MAX;
      float v = ((float)rand()) / (float)RAND_MAX;
      if(u + v > 1){
        u = 1 - u;
        v = 1 - v;
      }
      float w = 1 - (u + v);
      points[i] = u * vertexes[0] + v * vertexes[1] + w * vertexes[2];
    }

    return points;
  }

  Eigen::Vector3f * uniform_sample(int num, unsigned int& seed) const {

    Eigen::Vector3f * points = new Eigen::Vector3f[num];
    for(int i = 0; i < num; ++i){
      float u = ((float)rand_r(&seed)) / (float)RAND_MAX;
      float v = ((float)rand_r(&seed)) / (float)RAND_MAX;
      if(u + v > 1){
        u = 1 - u;
        v = 1 - v;
      }
      float w = 1 - (u + v);
      points[i] = u * vertexes[0] + v * vertexes[1] + w * vertexes[2];
    }
    return points;
  }

} Triangle;

struct TrackData {
	int result;
	float error;
	float J[6];
};

inline Eigen::Matrix4f getCameraMatrix(const Eigen::Vector4f& k) {
  Eigen::Matrix4f K;
	K << k.x(), 0, k.z(), 0,
       0, k.y(), k.w(), 0,
       0, 0, 1, 0,
       0, 0, 0, 1;
	return K;
}

inline Eigen::Matrix4f getInverseCameraMatrix(const Eigen::Vector4f& k) {
  Eigen::Matrix4f invK;
  invK << 1.0f / k.x(), 0, -k.z() / k.x(), 0,
       0, 1.0f / k.y(), -k.w() / k.y(), 0,
       0, 0, 1, 0,
       0, 0, 0, 1;
	return invK;
}

//std::ostream& operator<<(std::ostream& os, const uint3 val) {
//    os << val.x << ", " << val.y << ", " << val.z;
//    return os;
//}

static const float epsilon = 0.0000001;

inline void compareTrackData(std::string str, TrackData* l, TrackData* r,
		unsigned int size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].error - r[i].error) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.error =  " << l[i].error << std::endl;
			std::cout << "r.error =  " << r[i].error << std::endl;
		}

		if (std::abs(l[i].result - r[i].result) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.result =  " << l[i].result << std::endl;
			std::cout << "r.result =  " << r[i].result << std::endl;
		}
	}
}

inline void compareFloat(std::string str, float* l, float * r, unsigned int size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i] - r[i]) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l =  " << l[i] << std::endl;
			std::cout << "r =  " << r[i] << std::endl;
		}
	}
}

template<typename T>
void writefile(std::string prefix, int idx, T* data, unsigned int size) {

	std::string filename = prefix + NumberToString(idx);
	FILE* pFile = fopen(filename.c_str(), "wb");

	if (!pFile) {
		std::cout << "File opening failed : " << filename << std::endl;
		exit(1);
	}

	size_t write_cnt = fwrite(data, sizeof(T), size, pFile);

	std::cout << "File " << filename << " of size " << write_cnt << std::endl;

	fclose(pFile);
}

inline void writeVtkMesh(const char*                  filename,
                         const std::vector<Triangle>& mesh,
                         const Eigen::Matrix4f&       T_WM,
                         const float*                 point_data = nullptr,
                         const float*                 cell_data = nullptr){
  std::stringstream ss_points_W;
  std::stringstream ss_polygons;
  std::stringstream ss_point_data;
  std::stringstream ss_cell_data;
  int point_count = 0;
  int triangle_count = 0;
  bool has_point_data = point_data != nullptr;
  bool has_cell_data = cell_data != nullptr;

  for(unsigned int i = 0; i < mesh.size(); ++i ){
    const Triangle& triangle_M = mesh[i];

    Eigen::Vector3f vertex_0_W = (T_WM * triangle_M.vertexes[0].homogeneous()).head(3);
    Eigen::Vector3f vertex_1_W = (T_WM * triangle_M.vertexes[1].homogeneous()).head(3);
    Eigen::Vector3f vertex_2_W = (T_WM * triangle_M.vertexes[2].homogeneous()).head(3);

    ss_points_W << vertex_0_W.x() << " "
                << vertex_0_W.y() << " "
                << vertex_0_W.z() << std::endl;

    ss_points_W << vertex_1_W.x() << " "
                << vertex_1_W.y() << " "
                << vertex_1_W.z() << std::endl;

    ss_points_W << vertex_2_W.x() << " "
                << vertex_2_W.y() << " "
                << vertex_2_W.z() << std::endl;

    ss_polygons << "3 " << point_count << " " << point_count+1 <<
      " " << point_count+2 << std::endl;

    if(has_point_data){
      ss_point_data << point_data[i*3] << std::endl;
      ss_point_data << point_data[i*3 + 1] << std::endl;
      ss_point_data << point_data[i*3 + 2] << std::endl;
    }

    if(has_cell_data){
      ss_cell_data << cell_data[i] << std::endl;
    }

    point_count +=3;
    triangle_count++;
  }

  std::ofstream f;
  f.open(filename);
  f << "# vtk DataFile Version 1.0" << std::endl;
  f << "vtk mesh generated from KFusion" << std::endl;
  f << "ASCII" << std::endl;
  f << "DATASET POLYDATA" << std::endl;

  f << "POINTS " << point_count << " FLOAT" << std::endl;
  f << ss_points_W.str();

  f << "POLYGONS " << triangle_count << " " << triangle_count * 4 << std::endl;
  f << ss_polygons.str() << std::endl;
  if(has_point_data){
    f << "POINT_DATA " << point_count << std::endl;
    f << "SCALARS vertex_scalars float 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    f << ss_point_data.str();
  }

  if(has_cell_data){
    f << "CELL_DATA " << triangle_count << std::endl;
    f << "SCALARS cell_scalars float 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    f << ss_cell_data.str();
  }
  f.close();
}

inline void writeObjMesh(const char* filename,
                         const std::vector<Triangle>& mesh){
  std::stringstream points_M;
  std::stringstream faces;
  int point_count = 0;
  int face_count = 0;

  for(unsigned int i = 0; i < mesh.size(); i++){
    const Triangle& triangle_M = mesh[i];
    points_M << "v " << triangle_M.vertexes[0].x() << " "
                     << triangle_M.vertexes[0].y() << " "
                     << triangle_M.vertexes[0].z() << std::endl;
    points_M << "v " << triangle_M.vertexes[1].x() << " "
                     << triangle_M.vertexes[1].y() << " "
                     << triangle_M.vertexes[1].z() << std::endl;
    points_M << "v " << triangle_M.vertexes[2].x() << " "
                     << triangle_M.vertexes[2].y() << " "
                     << triangle_M.vertexes[2].z() << std::endl;

    faces  << "f " << (face_count*3)+1 << " " << (face_count*3)+2
           << " " << (face_count*3)+3 << std::endl;

    point_count +=3;
    face_count += 1;
  }

  std::ofstream f(filename);
  f << "# OBJ file format with ext .obj" << std::endl;
  f << "# vertex count = " << point_count << std::endl;
  f << "# face count = " << face_count << std::endl;
  f << points_M.str();
  f << faces.str();
  f.close();
  std::cout << "Written " << face_count << " faces and " << point_count
            << " points" << std::endl;
}
#endif
