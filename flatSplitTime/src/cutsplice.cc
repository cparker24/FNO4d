// This file does the following:
//    input: root file with a TTree with a vector<vector<vector<vector<float>>>> of data
//       optional: max time steps, default 60
//       optional: max events per file, default 1000
//     ouput: file named [input_file]_flat_[nevents]_[nFeatures]_[nXY]_[nTime].root 
//            has a TTree with a flat array of data per event
//            will make multiple files if necessary

// corsen the event to lower resolution in x and y by a factor of 4 and increate time out toward end

#include <iostream>
#include <time.h>
#include <string>

#include <chrono>
#include <thread>

#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TRandom3.h"

#include "TH1D.h"
#include "TProfile.h"

using namespace std;

int main(int argc, char** argv)
{
    // the constants:
    const int nXY = 30;
    const int ntau = 110;
    const int nF = 3;
    const int nSizeOut = nF*nXY*nXY*tau_steps;

    // Parse command line arguments
    std::string file_name = "coarse_0_10_10K.root";
    std::string output_file_name = "cutsplice_0_10_10K_110step.root";

    // const int tau_steps = 110;
    // const int hard_offset=11;
    // if (hard_offset > 0) cout << " USING HARD OFFSET: will make only one set per event of tau_steps="<<tau_steps<<endl;

    if (argc > 1) { 
        file_name = argv[1];
    }
    if (argc > 2) {
        output_file_name = argv[2];
    }

    std::cout << "Input file: " << file_name << std::endl;
    std::cout << "Output file: " << output_file_name << std::endl;
    std::cout << "Tau steps: " << tau_steps << std::endl;

    TFile* outFile = new TFile(output_file_name.c_str(), "RECREATE");
    TTree* out_tree = new TTree("t", "Flat data tree");


    const int nXY = 30;
    const int ntau = 11;
    const int nF = 3;
    const int nSizeOut = nF*nXY*nXY*tau_steps;
    float data[nSizeOut]; // assuming tau_steps <= 100, adjust as needed
    out_tree->Branch("coarse_res", data, Form("data[%d]/F", nSizeOut));
    int b_itau;
    out_tree->Branch("itau_start", &b_itau, "itau/I");

    // Open input file and get trees
    TFile* inFile = TFile::Open(file_name.c_str(), "READ");
    if (!inFile || inFile->IsZombie()) {
        std::cerr << "Error opening input file." << std::endl;
        return 1;
    }

    TTree* inTree = dynamic_cast<TTree*>(inFile->Get("t"));
    TTree* freezeTree = dynamic_cast<TTree*>(inFile->Get("freezetree"));
    if (!inTree || !freezeTree) {
        std::cerr << "Error: Could not find required trees in input file." << std::endl;
        inFile->Close();
        return 1;
    }

    // Set up branches
    float user_res[297000];
    float nsteps = 0;

    inTree->SetBranchAddress("user_res", user_res);
    freezeTree->SetBranchAddress("nsteps", &nsteps);

    //offsets in the input
    const int o_iF = nXY * nXY * ntau;
    const int o_iX = nXY * ntau;
    const int o_iY = ntau;

    //offsets in the output
    const int o_oF = nXY*nXY*tau_steps;
    const int o_oX = nXY*tau_steps;
    const int o_oY = tau_steps;

    TRandom3 randGen(0); // Seed with 0 for reproducibility

    // Loop over entries synchronously
    Long64_t nEntries = std::min(inTree->GetEntries(), freezeTree->GetEntries());
    for (Long64_t i = 0; i < nEntries; ++i) {
        if (i%1000 == 0) cout << " processed " << i << " events" << endl;

        // if (i>100) break;

        // cout << "i: " << i << " tau_steps: " << nsteps << endl;
        inTree->GetEntry(i);
        freezeTree->GetEntry(i);

        int offset = randGen.Integer(hard_offset > 0 ? hard_offset : tau_steps);
        int nsets  = (hard_offset > 0 ? 1 : (nsteps - offset)/tau_steps);
        // cout << " offset: " << offset << " in " << nsteps << " for nsets: " << nsets << endl;
        // cout << " size " << nSizeOut << endl;

        for (int iset=0; iset<nsets; ++iset) {
            for (int iX=0;iX<nXY;++iX) {
                int inX = iX*o_iX;
                int outX = iX*o_oX;
                for (int iY=0;iY<nXY;++iY) {
                    int inY = iY*o_iY;
                    int outY = iY*o_oY;
                    for (int iT=0; iT<tau_steps; ++iT) {
                        int itau = iset*tau_steps+iT+offset;
                        float tau = itau/10.+0.5;
                        float e = user_res[0*o_iF+inX+inY+itau];
                        data[0*o_oF+outX+outY+itau] = tau * pow(e,0.75);
                        data[1*o_oF+outX+outY+itau] = user_res[1*o_iF+inX+inY+itau];
                        data[2*o_oF+outX+outY+itau] = user_res[2*o_iF+inX+inY+itau];
                    }
                }
            }
            b_itau = iset*tau_steps+offset;
            out_tree->Fill();
        }
    }

    outFile->Write();
    outFile->Close();
    inFile->Close();
}
