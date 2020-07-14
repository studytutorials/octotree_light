#include <iostream>
#include <vector>

#include <lodepng.h>



int main(int argc, char** argv) {
  constexpr int w = 120;
  constexpr int h = 60;
  std::vector<unsigned char> raw_image (w * h * 4, 0);

  std::vector<unsigned char> png_image;
  const unsigned result = lodepng::encode(png_image, raw_image.data(), w, h);

  std::cout << "Encoded empty image\n";
}

