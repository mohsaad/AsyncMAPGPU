#include <iostream>

#include "../gpu/Region.h"


int main(int, char**) {
	std::cout << "Test..." << std::endl;

	double dataUnary[2]{0.0, 1.0};
	//double dataUnaryStrong[2]{1.0, 3.0};
	double dataPair[4]{1.0, 0.0, 0.0, 1.0};
	//double dataPairAsym[4]{3.5, 0.5, 0.2, 1.0};

	MPGraph<double, int> g;
	g.AddVariables({ 2, 2, 2 });
	std::vector<MPGraph<double, int>::PotentialID> pots;
	pots.push_back(g.AddPotential(MPGraph<double, int>::PotentialVector(dataUnary, 2)));
	//pots.push_back(g.AddPotential(MPGraph<double, int>::PotentialVector(dataUnary, 2)));
	//pots.push_back(g.AddPotential(MPGraph<double, int>::PotentialVector(dataUnary, 2)));
	pots.push_back(g.AddPotential(MPGraph<double, int>::PotentialVector(dataPair, 4)));
	//pots.push_back(g.AddPotential(MPGraph<double, int>::PotentialVector(dataPairAsym, 4)));

	std::vector<MPGraph<double, int>::RegionID> regs;
	for (int k = 0; k < 3; ++k) {
		regs.push_back(g.AddRegion(1.0, std::vector<int>{k}, pots[0]));
	}
	regs.push_back(g.AddRegion(1.0, { 0, 1 }, pots[1]));
	regs.push_back(g.AddRegion(1.0, { 1, 2 }, pots[1]));
	regs.push_back(g.AddRegion(1.0, { 0, 2 }, pots[1]));

	g.AddConnection(regs[0], regs[3]);
	g.AddConnection(regs[0], regs[5]);
	g.AddConnection(regs[1], regs[3]);
	g.AddConnection(regs[1], regs[4]);
	g.AddConnection(regs[2], regs[4]);
	g.AddConnection(regs[2], regs[5]);

	g.AllocateMessageMemory();

	//AsyncRMP<float, int> ARMP;
	//ARMP.RunMP(g, 1.0f);

	CPrecisionTimer CTmr;
	CTmr.Start();
	CudaAsyncRMPThread<double, int> ARMP;
	ARMP.CudaRunMP(g, 1.0, 100, 10, 100);
	std::cout << CTmr.Stop() << std::endl;

	return 0;
}
