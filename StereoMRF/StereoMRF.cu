#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <algorithm>

#include "../libAsyncRMP/Region.h"

struct Parameters {
	int MRFWidth;
	int numThreads;
	int WaitingTime;
	int numIterations;
};

template <typename T>
int PerformInference(std::vector<int>& info, std::vector<T>& volume, const Parameters& params) {
	MPGraph<T, int> g;
	g.AddVariables(std::vector<int>(info[1] * info[2], info[0]));
	std::vector<typename MPGraph<T, int>::PotentialID> pots;

	T* volumePtr = &volume[0];
	for (int x = 0; x < info[2]; ++x) {
		for (int y = 0; y < info[1]; ++y) {
			pots.push_back(g.AddPotential(typename MPGraph<T, int>::PotentialVector(volumePtr, info[0])));
			volumePtr += info[0];
		}
	}
	std::vector<T> pairPot(info[0] * info[0], T(0));
	for (int p1 = 0; p1 < info[0]; ++p1) {
		for (int p2 = 0; p2 < info[0]; ++p2) {
			pairPot[p1*info[0] + p2] = T(10)*(T(info[0]) - 1 - fabs(T(p1) - T(p2)));
			//pairPot[p1*info[0] + p2] = T(5)*(T(info[0]) - 1 - fabs(T(p1) - T(p2)));
		}
	}
	typename MPGraph<T, int>::PotentialID pairPotID = g.AddPotential(typename MPGraph<T, int>::PotentialVector(&pairPot[0], info[0]*info[0]));

	std::vector<typename MPGraph<T, int>::RegionID> regs;
	for (int k = 0; k < info[1] * info[2]; ++k) {
		regs.push_back(g.AddRegion(1.0f, std::vector<int>{k}, pots[k]));
	}

	
	for (int x = 0; x < info[2]; ++x) {
		for (int y = 0; y < info[1]; ++y) {
			for (int dx = 0; dx < params.MRFWidth; ++dx) {
				for (int dy = 0; dy < params.MRFWidth; ++dy) {
					if (y + dy < info[1] && x + dx < info[2] && !(dx == 0 && dy == 0)) {
						typename MPGraph<T, int>::RegionID regID = g.AddRegion(T(1), std::vector<int>{y + x*info[1], y + dy + (x+dx)*info[1]}, pairPotID);
						g.AddConnection(regs[y + x*info[1]], regID);
						g.AddConnection(regs[y + dy + (x + dx)*info[1]], regID);
					}
				}
			}
		}
	}
	/*for (int x = 0; x < info[2]; ++x) {
		for (int y = 0; y < info[1]; ++y) {
			if (y < info[1] - 1) {
				typename MPGraph<T, int>::RegionID regID = g.AddRegion(T(1), std::vector<int>{y + x*info[1], y + 1 + x*info[1]}, pairPotID);
				g.AddConnection(regs[y + x*info[1]], regID);
				g.AddConnection(regs[y + 1 + x*info[1]], regID);
			}
			if (x < info[2] - 1) {
				typename MPGraph<T, int>::RegionID regID = g.AddRegion(T(1), std::vector<int>{y + x*info[1], y + (x + 1)*info[1]}, pairPotID);
				g.AddConnection(regs[y + x*info[1]], regID);
				g.AddConnection(regs[y + (x + 1)*info[1]], regID);
			}
		}
	}*/

	g.AllocateMessageMemory();
        g.CoptMessageMemory();


	CPrecisionTimer CTmr;
	CTmr.Start();
	//AsyncRMP<T, int> ARMP;
	//AsyncRMPThread<T, int> ARMP;
	CudaAsyncRMP<T, int> ARMP;
        
        ARMP.CudaRunMP(g, T(1.0), params.numIterations, params.numThreads, params.WaitingTime);
	std::cout << CTmr.Stop() << std::endl;

	T* belPtr = NULL;
	size_t BelSize = ARMP.GetBeliefs(g, 1.0f, &belPtr, true);

	std::ofstream ofs("Beliefs.dat", std::ios_base::binary | std::ios_base::out);
	ofs.write((char*)belPtr, sizeof(T)*BelSize);
	ofs.close();

	g.DeleteBeliefs();
	return 0;
}

int main(int argc, char** argv) {
	std::cout << "Test..." << std::endl;

#ifdef _MSC_VER
	std::string fn("..\\StereoMatching\\CostVolume.dat");
#else
	std::string fn("../data/CostVolume.dat");
#endif

	std::ifstream ifs(fn.c_str(), std::ios_base::binary | std::ios_base::in);
	if (!ifs.is_open()) {
		std::cout << "Error openening file." << std::endl;
		return 0;
	}
	std::vector<int> info(3, -1);
	ifs.read((char*)&info[0], 3 * sizeof(int));
	size_t sz = info[0] * info[1] * info[2];
	std::vector<float> volume(sz, 0.0f);
	ifs.read((char*)&volume[0], sz * sizeof(float));
	ifs.close();
	std::transform(volume.begin(), volume.end(), volume.begin(), std::negate<float>());

	Parameters params;
	params.MRFWidth = 2;
	params.numIterations = 50;
	params.numThreads = 1;
	params.WaitingTime = 100;
	if (argc > 1) {
		params.MRFWidth = std::atoi(argv[1]);
		if (argc > 2) {
			params.numIterations = std::atoi(argv[2]);
			if (argc > 3) {
				params.numThreads = std::atoi(argv[3]);
				if (argc > 4) {
					params.WaitingTime = std::atoi(argv[4]);
				}
			}
		}
	}

	std::cout << "MRFWidth:     " << params.MRFWidth << std::endl;
	std::cout << "Iterations:   " << params.numIterations << std::endl;
	std::cout << "Threads:      " << params.numThreads << std::endl;
	std::cout << "Waiting Time: " << params.WaitingTime << std::endl;

	//PerformInference<float>(info, volume);

	std::vector<double> volumeDouble(sz, 0.0);
	std::copy(volume.begin(), volume.end(), volumeDouble.begin());
	PerformInference<double>(info, volumeDouble, params);

	return 0;
}
