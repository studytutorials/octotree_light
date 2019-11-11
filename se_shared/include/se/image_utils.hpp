#ifndef __IMAGE_UTILS_HPP
#define __IMAGE_UTILS_HPP

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <lodepng.h>



/**
 * Pack the individual RGBA channels into a single 32-bit unsigned integer.
 *
 * The uint32_t is stored in little-endian order in all common CPUs so the
 * alpha channel is stored in the MSB and the red channel in the LSB.
 * Red:   bits  0-7
 * Green: bits  8-15
 * Blue:  bits 16-23
 * Alpha: bits 24-31
 *
 * \param[in] r The value of the red channel.
 * \param[in] g The value of the green channel.
 * \param[in] b The value of the blue channel.
 * \param[in] a The value of the alpha channel.
 * \return The 32-bit unsigned integer RGBA value.
 */
static inline uint32_t to_rgba(const uint8_t r,
                               const uint8_t g,
                               const uint8_t b,
                               const uint8_t a) {

  return (a << 24) + (b << 16) + (g <<  8) +  r;
}



/**
 * Pack a color stored in an Eigen vector into a single 32-bit unsigned
 * integer. It is assumed that the R, G, B and A channels are stored in the x,
 * y, z and w members respectively. Their values are assumed to be in the range
 * [0, 1].
 *
 * \param[in] color The input color as an Eigen Vector.
 * \return The 32-bit unsigned integer RGBA value.
 */
static inline uint32_t to_rgba(const Eigen::Vector4f& color) {

  return (static_cast<uint8_t>(color.w() * 255) << 24)
       + (static_cast<uint8_t>(color.z() * 255) << 16)
       + (static_cast<uint8_t>(color.y() * 255) <<  8)
       +  static_cast<uint8_t>(color.x() * 255);
}



/**
 * Pack a color stored in an Eigen vector into a single 32-bit unsigned
 * integer. It is assumed that the R, G and B channels are stored in the x, y
 * and z members respectively. Their values are assumed to be in the range [0,
 * 1]. The alpha channel is assumed to be completely opaque, e.g. 1.
 *
 * \param[in] color The input color as an Eigen Vector.
 * \return The 32-bit unsigned integer RGBA value.
 */
static inline uint32_t to_rgba(const Eigen::Vector3f& color) {

  return (                                 0xFF << 24)
       + (static_cast<uint8_t>(color.z() * 255) << 16)
       + (static_cast<uint8_t>(color.y() * 255) <<  8)
       +  static_cast<uint8_t>(color.x() * 255);
}



/**
 * Pack a color stored in an Eigen vector into a single 32-bit unsigned
 * integer. It is assumed that the R, G, B and A channels are stored in the x,
 * y, z and w members respectively. Their values are assumed to be in the range
 * [0x00, 0xFF].
 *
 * \param[in] color The input color as an Eigen Vector.
 * \return The 32-bit unsigned integer RGBA value.
 */
static inline uint32_t to_rgba(const Eigen::Vector4i& color) {

  return (color.w() << 24)
       + (color.z() << 16)
       + (color.y() <<  8)
       +  color.x();
}



/**
 * Pack a color stored in an Eigen vector into a single 32-bit unsigned
 * integer. It is assumed that the R, G and B channels are stored in the x, y
 * and z members respectively. Their values are assumed to be in the range
 * [0x00, 0xFF]. The alpha channel is assumed to be completely opaque, e.g.
 * 0xFF.
 *
 * \param[in] color The input color as an Eigen Vector.
 * \return The 32-bit unsigned integer RGBA value.
 */
static inline uint32_t to_rgba(const Eigen::Vector3i& color) {

  return (     0xFF << 24)
       + (color.z() << 16)
       + (color.y() <<  8)
       +  color.x();
}



/**
 * Get the value of the red channel from a 32-bit packed RGBA value.
 *
 * \param[in] rgba The 32-bit packed RGBA value.
 * \return The value of the red channel.
 */
static inline uint8_t r_from_rgba(const uint32_t rgba) {
  return (uint8_t) rgba;
}



/**
 * Get the value of the green channel from a 32-bit packed RGBA value.
 *
 * \param[in] rgba The 32-bit packed RGBA value.
 * \return The value of the green channel.
 */
static inline uint8_t g_from_rgba(const uint32_t rgba) {
  return (uint8_t) (rgba >> 8);
}



/**
 * Get the value of the blue channel from a 32-bit packed RGBA value.
 *
 * \param[in] rgba The 32-bit packed RGBA value.
 * \return The value of the blue channel.
 */
static inline uint8_t b_from_rgba(const uint32_t rgba) {
  return (uint8_t) (rgba >> 16);
}



/**
 * Get the value of the alpha channel from a 32-bit packed RGBA value.
 *
 * \param[in] rgba The 32-bit packed RGBA value.
 * \return The value of the alpha channel.
 */
static inline uint8_t a_from_rgba(const uint32_t rgba) {
  return (uint8_t) (rgba >> 24);
}



/**
 * Blend two RGBA colors based on the value of the blending parameter alpha.
 * Returns a color alpha * rgba_1 + (1 - alpha) * rgba_2. The values of
 * alpha are assumed to be in the range [0, 1].
 *
 * \note Code from https://stackoverflow.com/a/12016968
 *
 * \param[in] rgba_1 A 32-bit packed RGBA value.
 * \param[in] rgba_2 A 32-bit packed RGBA value.
 * \param[in] alpha The value of the blending parameter.
 * \return The 32-bit RGBA value of the blended color.
 *
 * \warning Swapping the rgba_1 with rgba_2 while keeping the same value for
 * alpha will not always produce the same result.
 */
static inline uint32_t blend(const uint32_t rgba_1,
                             const uint32_t rgba_2,
                             const float    alpha) {

  const uint8_t r = static_cast<uint8_t>(
      round(alpha * r_from_rgba(rgba_1) + (1 - alpha) * r_from_rgba(rgba_2)));
  const uint8_t g = static_cast<uint8_t>(
      round(alpha * g_from_rgba(rgba_1) + (1 - alpha) * g_from_rgba(rgba_2)));
  const uint8_t b = static_cast<uint8_t>(
      round(alpha * b_from_rgba(rgba_1) + (1 - alpha) * b_from_rgba(rgba_2)));
  const uint8_t a = static_cast<uint8_t>(
      round(alpha * a_from_rgba(rgba_1) + (1 - alpha) * a_from_rgba(rgba_2)));

  return to_rgba(r, g, b, a);
}



/**
 * Save a depth image with depth values in millimeters to a PNG.
 *
 * \param[in] depth Pointer to the 16-bit image data.
 * \param[in] depth_size Size of the depth image in pixels (width and height).
 * \param[in] filename The name of the PNG file to create.
 * \return 0 on success, nonzero on error.
 */
static int save_depth_png(const uint16_t*        depth,
                          const Eigen::Vector2i& depth_size,
                          const std::string&     filename) {

  // Allocate a new image buffer to use for changing the image data from little
  // endian (used in x86 and ARM CPUs) to big endian order (used in PNG).
  const size_t num_pixels = depth_size.x() * depth_size.y();
  uint16_t* depth_big_endian = new uint16_t[num_pixels];
#pragma omp parallel for
  for (size_t i = 0; i < num_pixels; ++i) {
    // Swap the byte order.
    const uint16_t pixel = depth[i];
    const uint16_t low_byte = pixel & 0x00FF;
    const uint16_t high_byte = (pixel & 0xFF00) >> 8;
    depth_big_endian[i] = low_byte << 8 | high_byte;
  }

  // Save the image to file.
  const unsigned ret = lodepng_encode_file(
      filename.c_str(),
      reinterpret_cast<const unsigned char*>(depth_big_endian),
      depth_size.x(),
      depth_size.y(),
      LCT_GREY,
      16);

  delete depth_big_endian;
  return ret;
}



/**
 * Load a PNG depth image into a buffer with depth values in millimeters.
 *
 * \param[in] depth Pointer to the loaded 16-bit image data.
 * \param[in] depth_size Size of the depth image in pixels (width and height).
 * \param[in] filename The name of the PNG file to load.
 * \return 0 on success, nonzero on error.
 *
 * \warning The memory for the image buffer is allocated inside this function.
 * free(*depth) must be called to free the memory. width * height *
 * sizeof(uint16_t) bytes are allocated.
 */
static int load_depth_png(uint16_t**         depth,
                          Eigen::Vector2i&   depth_size,
                          const std::string& filename) {

  // Load the image.
  const unsigned ret = lodepng_decode_file(
      reinterpret_cast<unsigned char**>(depth),
      reinterpret_cast<unsigned int*>(&(depth_size.x())),
      reinterpret_cast<unsigned int*>(&(depth_size.y())),
      filename.c_str(),
      LCT_GREY,
      16);

  // Change the image data from little endian (used in x86 and ARM CPUs) to big
  // endian order (used in PNG).
  const size_t num_pixels = depth_size.x() * depth_size.y();
#pragma omp parallel for
  for (size_t i = 0; i < num_pixels; ++i) {
    // Swap the byte order.
    const uint16_t pixel = (*depth)[i];
    const uint16_t low_byte = pixel & 0x00FF;
    const uint16_t high_byte = (pixel & 0xFF00) >> 8;
    (*depth)[i] = low_byte << 8 | high_byte;
  }

  return ret;
}



/**
 * Save a depth image with depth values in millimeters to a P2 PGM.
 *
 * \note For documentation on the structure of P2 PGM images see here
 * https://en.wikipedia.org/wiki/Netpbm_format
 *
 * \param[in] depth Pointer to the 16-bit image data.
 * \param[in] depth_size Size of the depth image in pixels (width and height).
 * \param[in] filename The name of the PGM file to create.
 * \return 0 on success, nonzero on error.
 */
static int save_depth_pgm(const uint16_t*        depth,
                          const Eigen::Vector2i& depth_size,
                          const std::string&     filename) {

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  // Write the PGM header.
  file << "P2\n";
  file << depth_size.x() << " " << depth_size.y() << "\n";
  file << UINT16_MAX << "\n";

  // Write the image data.
  for (int y = 0; y < depth_size.y(); y++) {
    for (int x = 0; x < depth_size.x(); x++) {
      const int pixel = x + y * depth_size.x();
      file << depth[pixel];
      // Do not add a whitespace after the last element of a row.
      if (x < depth_size.x() - 1) {
        file << " ";
      }
    }
    // Add a newline at the end of each row.
    file << "\n";
  }

  file.close();

  return 0;
}



/**
 * Load a P2 PGM depth image into a buffer with depth values in millimeters.
 *
 * \param[in] depth Pointer to the loaded 16-bit image data.
 * \param[in] depth_size Size of the depth image in pixels (width and height).
 * \param[in] filename The name of the PGM file to load.
 * \return 0 on success, nonzero on error.
 *
 * \warning The memory for the image buffer is allocated inside this function.
 * free(*depth) must be called to free the memory. width * height *
 * sizeof(uint16_t) bytes are allocated.
 */
static int load_depth_pgm(uint16_t**         depth,
                          Eigen::Vector2i&   depth_size,
                          const std::string& filename) {

  // Open the file for reading.
  std::ifstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to read file " << filename << "\n";
    return 1;
  }

  // Read the file format.
  std::string pgm_format;
  std::getline(file, pgm_format);
  if (pgm_format != "P2") {
    std::cerr << "Invalid PGM format: " << pgm_format << "\n";
    return 1;
  }

  // Read the image size and allocate memory for the image.
  file >> depth_size.x() >> depth_size.y();
  const size_t depth_size_pixels = depth_size.x() * depth_size.y();
  *depth = static_cast<uint16_t*>(malloc(depth_size_pixels * sizeof(uint16_t)));

  // Read the maximum pixel value.
  size_t max_value;
  file >> max_value;
  if (max_value > UINT16_MAX) {
    std::cerr << "Invalid maximum depth value " << max_value
        << " > " << UINT16_MAX << "\n";
    return 1;
  }

  // Read the image data. Do not perform any scaling since in our cases the
  // pixel values represent distances.
  for (size_t p = 0; p < depth_size_pixels; ++p) {
    file >> (*depth)[p];
  }

  file.close();

  return 0;
}


#endif

