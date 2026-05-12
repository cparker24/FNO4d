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

#include "TProfile.h"
#include "TPad.h"

using namespace std;

int main(int argc, char** argv)
{
    // Parse command line arguments
    std::string file_name = "corse_0_10_10K.root";
    std::string output_file_name = "profiles.root";
    const int tau_steps = 120;

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

    const int nXY = 30;
    const int ntau = 120;
    const int nF = 3;
    const int nSizeOut = nF*nXY*nXY*tau_steps;

    float taumax = ntau/10.+0.5;
    float taumin = 0.5;
    TProfile* p_e = new TProfile("e","max e; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);
    TProfile* p_ep75 = new TProfile("ep75","max e^{0.75}; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);
    TProfile* p_taue = new TProfile("taue","max #tau#times{}e; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);
    TProfile* p_tauep75 = new TProfile("tauep75","max #tau#times{}e^{0.75}; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);

    TProfile* pm_e = new TProfile("e_mean","mean e; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);
    TProfile* pm_ep75 = new TProfile("ep75_mean","mean e^{0.75}; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);
    TProfile* pm_taue = new TProfile("taue_mean","mean #tau#times{}e; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);
    TProfile* pm_tauep75 = new TProfile("tauep75_mean","mean #tau#times{}e^{0.75}; #tau [fm/c]; mean [GeV/fm^3]", ntau, taumin, taumax);

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
    float user_res[324000];
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

        // cout << "i: " << i << " tau_steps: " << nsteps << endl;
        if (i>1000) break;
        inTree->GetEntry(i);
        freezeTree->GetEntry(i);

        int offset = 0; //randGen.Integer(tau_steps);
        int nsets = 1; //(nsteps - offset)/tau_steps;
        // cout << " offset: " << offset << " in " << nsteps << " for nsets: " << nsets << endl;
        // cout << " size " << nSizeOut << endl;

        for (int iset=0; iset<nsets; ++iset) {
            for (int iT=0; iT<tau_steps; ++iT) {
                int itau = iset*tau_steps+iT+offset;
                float tau = itau/10.+0.5;

                float maxe = 0;
                float sume = 0;

                float maxep75 = 0;
                float sumep75 = 0;

                float maxetau = 0;
                float sumetau = 0;

                float maxep75tau = 0;
                float sumep75tau = 0;

                for (int iX=0;iX<nXY;++iX) {
                    int inX = iX*o_iX;
                    int outX = iX*o_oX;
                    for (int iY=0;iY<nXY;++iY) {
                        int inY = iY*o_iY;
                        int outY = iY*o_oY;

                        float e = user_res[0*o_iF+inX+inY+itau];
                        if (e > maxe) maxe = e;
                        // if (iT==0 && e>10) cout << " iT " << iT << " and e " << e << " so maxe " << maxe << endl;
                        sume += e;

                        float ep75 = pow(e,0.75);
                        if (ep75 > maxep75) maxep75 = ep75;
                        sumep75 += ep75;

                        float taue = tau * e;
                        if (taue > maxetau) maxetau = taue;
                        sumetau += taue;

                        float tauep75 = tau * ep75;
                        if (tauep75 > maxep75tau) maxep75tau = tauep75;
                        sumep75tau += tauep75;
                    }
                }
            if (maxe == 0.) break;

            pm_e->Fill(tau, sume / (nXY * nXY));
            pm_ep75->Fill(tau, sumep75 / (nXY * nXY));
            pm_taue->Fill(tau, sumetau / (nXY * nXY));
            pm_tauep75->Fill(tau, sumep75tau / (nXY * nXY));

            if (maxe == 0) cout << " tau " << tau << " and maxe " << maxe << endl;
            p_e->Fill(tau, maxe);
            p_ep75->Fill(tau, maxep75);
            p_taue->Fill(tau, maxetau);
            p_tauep75->Fill(tau, maxep75tau);
            }
        }
    }
    p_e->Draw();
    outFile->cd();
    gPad->Print("p_e.pdf");

    p_ep75->Draw();
    outFile->cd();
    gPad->Print("p_ep75.pdf");

    p_taue->Draw();
    outFile->cd();
    gPad->Print("p_taue.pdf");

    p_tauep75->Draw();
    outFile->cd();
    gPad->Print("p_tauep75.pdf");

    pm_e->Draw();
    outFile->cd();
    gPad->Print("pm_e.pdf");

    pm_ep75->Draw();
    outFile->cd();
    gPad->Print("pm_ep75.pdf");

    pm_taue->Draw();
    outFile->cd();
    gPad->Print("pm_taue.pdf");

    pm_tauep75->Draw();
    outFile->cd();
    gPad->Print("pm_tauep75.pdf");
    outFile->Write();
    outFile->Close();
    inFile->Close();
}
