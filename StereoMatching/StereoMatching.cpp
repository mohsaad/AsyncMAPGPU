#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define cimg_display 0
#define cimg_use_jpeg
#define cimg_use_png
//#define cimg_use_openmp
#include "CImg.h"

int main(int, char**) {
	std::cout << "Test..." << std::endl;

	std::string fnimL("Data/stereo-pairs/tsukuba/imL.png");
	std::string fnimR("Data/stereo-pairs/tsukuba/imR.png");

	cimg_library::CImg<unsigned char> imL(fnimL.c_str());
	imL.resize(imL.width()/2, imL.height()/2, 1, 3, 1);
	cimg_library::CImg<unsigned char> imR(fnimR.c_str());
	imR.resize(imR.width() / 2, imR.height() / 2, 1, 3, 1);

	int dispRange = 16;
	int winWidth = 3;
	int winHeight = 3;

	std::vector<float> dataVolume((imL.width() - dispRange + 1)*imL.height()*dispRange, 0.0f);

	for (int x = dispRange - 1; x < imL.width(); ++x) {
		for (int y = 0; y < imL.height(); ++y) {
			for (int d = 0; d < dispRange; ++d) {
				float value = 0.0f;
				float contrib = 0.0f;

				for (int w = x-winWidth/2; w < x - winWidth/2 + winWidth; ++w) {
					for (int h = y - winHeight/2; h < y - winHeight/2 + winHeight; ++h) {
						for (int c = 0; c < 3; ++c) {
							if (w - d >= 0 && w < imL.width() && h >= 0 && h < imL.height()) {
								value += fabs(float(imL(w, h, c)) - float(imR(w - d, h, c)));
								++contrib;
							}
						}
					}
				}

				dataVolume[d + dispRange*y + (x-dispRange+1)*dispRange*imL.height()] = value/contrib;
			}
		}
	}

	std::vector<int> Info{ dispRange, imL.height(), imL.width() - dispRange + 1 };

	std::ofstream ofs("CostVolume_16.dat", std::ios_base::binary | std::ios_base::out);
	if (ofs.is_open()) {
		ofs.write((char*)&Info[0], Info.size()*sizeof(int));
		ofs.write((char*)&dataVolume[0], dataVolume.size()*sizeof(float));
		ofs.close();
	} else {
		std::cout << "Error writing file." << std::endl;
	}
	

	return 0;
}